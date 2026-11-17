"""아찔한 순간 기록 API (영상 파일 모드 저장·목록, 실시간 /live). 로직은 nearmiss.py·live_recorder.py에 있다."""

import asyncio
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.fake_warning import FakeWarningSource
from app.live_recorder import LiveRecorder
from app.models import Clip, Device, Drive, Event
from app.nearmiss import NEARMISS_CFG, ROOT, EventIn, record_event

# FastAPI 권장 방식: 요청마다 DB 세션을 주입받는 타입
SessionDep = Annotated[Session, Depends(get_session)]

# 로그인 대신 앱이 처음 만든 기기 UUID를 모든 요청의 X-Device-Id 헤더로 보낸다.
# 다른 기기의 주행·이벤트·클립은 "없는 것"(404)으로 보이게 해서 남의 기록을 못 보게 한다.
# 헤더가 없거나 UUID 모양이 아니면 FastAPI가 422로 막는다.
DeviceIdDep = Annotated[uuid.UUID, Header(alias="X-Device-Id")]

app = FastAPI(title="teum")

# 브라우저는 다른 주소(Vite 5173)에서 오는 요청을 막으므로, .env의 CORS_ORIGINS만 허용한다
# (db.py를 import할 때 load_dotenv()가 이미 실행됨). 쉼표로 여러 개 적을 수 있다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class DriveIn(BaseModel):
    source: Literal["live", "upload"]
    video_name: str | None = None


class EventOut(BaseModel):
    id: int
    drive_id: int
    occurred_at: datetime
    offset_s: float
    type: str
    level: str
    min_ttc_s: float | None
    distance_m: float | None
    speed_kmh: float | None
    duration_s: float | None
    clip_key: str | None


def get_own_drive(session: Session, drive_id: int, device_id: uuid.UUID) -> Drive:
    """이 기기의 주행만 돌려준다. 없거나 다른 기기 것이면 똑같이 404.

    사용처: 5단계(기록). 주행 번호를 받는 API가 처리 전에 부른다.
    입력: session DB 세션, drive_id 주행 번호, device_id 요청 헤더의 기기 UUID
    출력: Drive
    주의: 403이 아니라 404로 답한다 (남의 주행 번호가 있는지조차 알려주지 않으려고).
    """
    drive = session.get(Drive, drive_id)
    if drive is None or drive.device_id != device_id:
        raise HTTPException(status_code=404, detail="drive not found")
    return drive


@app.post("/drives")
def create_drive(body: DriveIn, device_id: DeviceIdDep, session: SessionDep) -> dict:
    """주행(또는 영상 파일 분석) 1건을 시작한다. 처음 보는 기기면 기기도 등록한다."""
    if session.get(Device, device_id) is None:
        session.add(Device(id=device_id))
    drive = Drive(device_id=device_id, source=body.source, video_name=body.video_name)
    session.add(drive)
    session.commit()
    return {"drive_id": drive.id}


@app.post("/drives/{drive_id}/events")
def create_event(
    drive_id: int, body: EventIn, device_id: DeviceIdDep, session: SessionDep
) -> dict:
    """아찔한 순간을 저장한다. 같은 종류가 merge_gap_s 안에 또 오면 기존 건에 합친다."""
    get_own_drive(session, drive_id, device_id)
    event, merged = record_event(session, drive_id, body)
    return {"event_id": event.id, "merged": merged}


@app.get("/events")
def list_events(
    device_id: DeviceIdDep, session: SessionDep, limit: int = 50
) -> list[EventOut]:
    """요청한 기기의 아찔한 순간 목록 (최근 것부터). 목록 화면에서 쓴다."""
    events = session.scalars(
        select(Event)
        .join(Drive)
        .where(Drive.device_id == device_id)
        .order_by(Event.occurred_at.desc())
        .limit(limit)
    ).all()
    return [
        EventOut(
            id=e.id,
            drive_id=e.drive_id,
            occurred_at=e.occurred_at,
            offset_s=e.offset_s,
            type=e.type,
            level=e.level,
            min_ttc_s=e.min_ttc_s,
            distance_m=e.distance_m,
            speed_kmh=e.speed_kmh,
            duration_s=e.duration_s,
            clip_key=e.clip.object_key if e.clip else None,
        )
        for e in events
    ]


@app.get("/events/{event_id}/clip")
def get_clip(
    event_id: int, device_id: DeviceIdDep, session: SessionDep
) -> FileResponse:
    """이벤트의 클립 영상 파일을 돌려준다. 앱 목록에서 눌렀을 때 재생용.

    다른 기기의 이벤트면 클립이 없는 것과 똑같이 404를 준다.
    """
    clip = session.scalars(
        select(Clip)
        .join(Event)
        .join(Drive)
        .where(Clip.event_id == event_id, Drive.device_id == device_id)
    ).first()
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return FileResponse(ROOT / clip.object_key, media_type="video/mp4")


@app.websocket("/live")
async def live(websocket: WebSocket, device_id: Annotated[uuid.UUID, Query()]) -> None:
    """실시간 주행. 폰이 JPEG 프레임을 보내면 경고 판단 결과(JSON)를 돌려보내고, 아찔한 순간은 클립으로 저장한다.

    사용처: 5단계(표시·기록). 앱 주행 화면이 ws://서버/live?device_id=<UUID>로 연결한다.
    입력: 바이너리 메시지 1개 = JPEG 프레임 1장
    출력: 연결 직후 {"drive_id"}, 이후 프레임마다 {"t_s", "warnings"}
    주의: 브라우저 WebSocket은 헤더를 못 붙여서 기기 UUID를 X-Device-Id 대신 주소(query)로 받는다.
          없거나 UUID 모양이 아니면 FastAPI가 연결을 거절한다(1008).
          DB·클립 저장은 LiveRecorder가 뒤에서 하므로 이 루프는 기다리지 않는다.
    """
    await websocket.accept()
    recorder = LiveRecorder(device_id)
    drive_id = await asyncio.to_thread(recorder.start)
    # 연결마다 새로 만든다. 진짜 경고 판단으로 바꿔도 YOLO·추적 상태는 연결마다 새로 불러온다(11/5 결정)
    judge = FakeWarningSource(NEARMISS_CFG["live_fake_warning_at_s"])
    await websocket.send_json({"drive_id": drive_id})

    # 촬영 시각 = 서버가 받은 순간 (Windows에서 monotonic은 해상도가 약 16ms라 perf_counter 사용)
    t0 = time.perf_counter()
    try:
        while True:
            jpg = await websocket.receive_bytes()
            t_s = time.perf_counter() - t0
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            warnings = judge(frame, t_s)
            recorder.submit(jpg, t_s, warnings)
            await websocket.send_json(
                {"t_s": t_s, "warnings": [asdict(w) for w in warnings]}
            )
    except WebSocketDisconnect:
        pass
    finally:
        # 남은 클립 저장·주행 종료 (기다리는 동안 다른 연결이 멈추지 않게 스레드에서)
        await asyncio.to_thread(recorder.close)
