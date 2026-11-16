"""아찔한 순간 기록 API (1차: 영상 파일 모드 저장·목록). 로직은 nearmiss.py에 있다."""

import os
import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Clip, Device, Drive, Event
from app.nearmiss import ROOT, EventIn, record_event

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
