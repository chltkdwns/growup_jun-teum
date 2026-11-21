"""아찔한 순간 기록 API (영상 업로드 분석, 저장·목록, 실시간 /live). 로직은 nearmiss.py·live_recorder.py·upload_analyzer.py에 있다."""

import asyncio
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
import yaml
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_session
from app.live_overlay import live_overlay
from app.live_recorder import LiveRecorder
from app.models import Clip, Device, Drive, Event
from app.nearmiss import CONFIG_PATH, NEARMISS_CFG, ROOT, EventIn, record_event
from app.pipeline import PerceptionPipeline
from app.risk.warning_source import PipelineWarningSource
from app.upload_analyzer import UploadProgress, run_upload_job

# FastAPI 권장 방식: 요청마다 DB 세션을 주입받는 타입
SessionDep = Annotated[Session, Depends(get_session)]

# 로그인 대신 앱이 처음 만든 기기 UUID를 모든 요청의 X-Device-Id 헤더로 보낸다.
# 다른 기기의 주행·이벤트·클립은 "없는 것"(404)으로 보이게 해서 남의 기록을 못 보게 한다.
# 헤더가 없거나 UUID 모양이 아니면 FastAPI가 422로 막는다.
DeviceIdDep = Annotated[uuid.UUID, Header(alias="X-Device-Id")]

app = FastAPI(title="teum")

# 분석 파이프라인은 config.yaml 전체(camera, lane, detect ...)가 필요하다
CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
UPLOAD_DIR = ROOT / NEARMISS_CFG["upload_dir"]
# 영상 분석은 한 번에 하나씩 (GPU가 하나라 동시에 돌리면 서로 느려짐). 나머지는 queued로 기다린다
UPLOAD_POOL = ThreadPoolExecutor(max_workers=1)
# 주행 번호 → 분석 진행 상황 (메모리에만 있음, 서버를 다시 켜면 사라짐)
UPLOAD_JOBS: dict[int, UploadProgress] = {}

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


class AnalysisOut(BaseModel):
    status: Literal["queued", "running", "done", "failed"]
    frames_done: int
    frames_total: int | None
    events: int  # 지금까지 저장된 아찔한 순간 수
    error: str | None


class DriveOut(BaseModel):
    id: int
    source: str  # live | upload
    video_name: str | None
    started_at: datetime
    ended_at: datetime | None
    # 주행 길이 (초). 실시간만 (끝 - 시작). upload는 끝-시작이 영상 길이가 아니라 분석 시간이라 None
    duration_s: float | None
    distance_km: float | None
    events: int  # 아찔한 순간 수
    danger_events: int  # 그중 위험 단계 수
    worst_ttc_s: float | None  # 이 주행에서 가장 낮았던 TTC


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


def make_judge() -> PipelineWarningSource:
    """영상 1개 또는 실시간 연결 1개에 쓸 경고 판단(분석 → 거리·TTC → 경고 단계)을 만든다.

    사용처: 4단계(경고 판단) 자리. 업로드 분석 스레드(run_upload_job)는 영상마다, /live는 연결마다 한 번 부른다.
    입력: 없음
    출력: PipelineWarningSource (frame, t_s → 경고 목록). 내 속도는 만든 뒤 speed_kmh에 대입한다
    주의: 추적·TTC 기록이 영상마다 달라야 해서 매번 새로 만든다 (YOLO도 다시 불러옴, 11/5 결정).
          YOLO 가중치 경로가 backend 기준이라 서버는 backend 폴더에서 켜야 한다.
    """
    return PipelineWarningSource(PerceptionPipeline(CFG), CFG["risk"])


@app.post("/drives/upload")
def upload_video(file: UploadFile, device_id: DeviceIdDep, session: SessionDep) -> dict:
    """블랙박스 영상을 받아 주행(upload)을 만들고, 분석은 뒤에서 시작한다. 분석이 끝나길 기다리지 않는다.

    사용처: 5단계(기록). 앱의 영상 올리기 화면. 이후 GET /drives/{id}/analysis로 진행 상황을 본다.
    입력: file 영상 파일 (multipart/form-data, 이름 "file")
    출력: {"drive_id", "status": "queued"}
    주의: 파일은 원래 이름이 아니라 "주행번호.확장자"로 저장한다 (이름 겹침·경로 조작 방지). 원래 이름은 video_name에.
    """
    if session.get(Device, device_id) is None:
        session.add(Device(id=device_id))
    drive = Drive(device_id=device_id, source="upload", video_name=file.filename)
    session.add(drive)
    session.commit()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_path = UPLOAD_DIR / f"{drive.id}{Path(file.filename or '').suffix}"
    # 블랙박스 영상은 100MB가 넘을 수 있어서 메모리에 다 올리지 않고 조금씩 복사한다
    with video_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    progress = UploadProgress()
    UPLOAD_JOBS[drive.id] = progress
    UPLOAD_POOL.submit(
        run_upload_job, drive.id, video_path, make_judge, progress
    )
    # progress.status는 맡기자마자 running으로 바뀔 수 있어서, 받은 시점의 상태(queued)를 그대로 답한다
    return {"drive_id": drive.id, "status": "queued"}


@app.get("/drives/{drive_id}/analysis")
def get_analysis(
    drive_id: int, device_id: DeviceIdDep, session: SessionDep
) -> AnalysisOut:
    """올린 영상의 분석 진행 상황과 지금까지 저장된 아찔한 순간 수를 돌려준다. 앱이 몇 초마다 묻는다.

    주의: 진행 상황은 메모리에 있다. 서버를 다시 켜서 기록이 없으면, 주행이 끝났으면(ended_at) done,
          아니면 분석 도중 꺼진 것이라 failed로 답한다.
    """
    drive = get_own_drive(session, drive_id, device_id)
    if drive.source != "upload":
        raise HTTPException(status_code=404, detail="upload drive not found")
    n_events = session.scalar(
        select(func.count()).select_from(Event).where(Event.drive_id == drive_id)
    )
    progress = UPLOAD_JOBS.get(drive_id)
    if progress is None:
        done = drive.ended_at is not None
        progress = UploadProgress(
            status="done" if done else "failed",
            frames_done=drive.frames or 0,
            error=None if done else "서버가 다시 켜져서 분석이 중단됨",
        )
    return AnalysisOut(
        status=progress.status,
        frames_done=progress.frames_done,
        frames_total=progress.frames_total,
        events=n_events,
        error=progress.error,
    )


@app.post("/drives/{drive_id}/events")
def create_event(
    drive_id: int, body: EventIn, device_id: DeviceIdDep, session: SessionDep
) -> dict:
    """아찔한 순간을 저장한다. 같은 종류가 merge_gap_s 안에 또 오면 기존 건에 합친다."""
    get_own_drive(session, drive_id, device_id)
    event, merged = record_event(session, drive_id, body)
    return {"event_id": event.id, "merged": merged}


@app.get("/drives")
def list_drives(
    device_id: DeviceIdDep, session: SessionDep, limit: int = 30
) -> list[DriveOut]:
    """요청한 기기의 주행 목록과 주행별 요약 (최근 것부터). 주행 분석 화면에서 쓴다.

    사용처: 5단계(표시). 앱 '주행 분석' 화면.
    입력: device_id 요청 헤더의 기기 UUID, limit 최대 개수
    출력: DriveOut 목록
    주의: schema.sql의 drive_summary 뷰와 같은 계산을 기기 조건을 붙여 SQLAlchemy로 한다
          (뷰는 models.py에 클래스가 없어서). 아찔한 순간이 없는 주행도 0건으로 나온다(LEFT JOIN).
    """
    rows = session.execute(
        select(
            Drive,
            func.count(Event.id),
            func.count(Event.id).filter(Event.level == "danger"),
            func.min(Event.min_ttc_s),
        )
        .outerjoin(Event)
        .where(Drive.device_id == device_id)
        .group_by(Drive.id)
        .order_by(Drive.started_at.desc())
        .limit(limit)
    ).all()
    return [
        DriveOut(
            id=d.id,
            source=d.source,
            video_name=d.video_name,
            started_at=d.started_at,
            ended_at=d.ended_at,
            duration_s=(
                (d.ended_at - d.started_at).total_seconds()
                if d.source == "live" and d.ended_at
                else None
            ),
            distance_km=d.distance_km,
            events=n,
            danger_events=n_danger,
            worst_ttc_s=worst,
        )
        for d, n, n_danger, worst in rows
    ]


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
    입력: 바이너리 메시지 1개 = JPEG 프레임 1장. 글자 메시지 {"speed_kmh": 내 속도(km/h, 모르면 null)} = 폰 GPS 속도
          (속도가 바뀔 때만 보낸다. 안 오면 안전거리 판단만 건너뛴다)
    출력: 연결 직후 {"drive_id"}, 이후 프레임마다 {"t_s", "warnings", "boxes", "lanes"} (boxes·lanes는 live_overlay 참고)
    주의: 브라우저 WebSocket은 헤더를 못 붙여서 기기 UUID를 X-Device-Id 대신 주소(query)로 받는다.
          없거나 UUID 모양이 아니면 FastAPI가 연결을 거절한다(1008).
          DB·클립 저장은 LiveRecorder가 뒤에서 하므로 이 루프는 기다리지 않는다.
    """
    await websocket.accept()
    recorder = LiveRecorder(device_id)
    drive_id = await asyncio.to_thread(recorder.start)
    # 연결마다 새로 만든다 (YOLO·추적 상태는 연결마다 새로 불러온다, 11/5 결정). 불러오는 동안 다른 연결이 멈추지 않게 스레드에서
    judge = await asyncio.to_thread(make_judge)
    await websocket.send_json({"drive_id": drive_id})

    # 촬영 시각 = 서버가 받은 순간 (Windows에서 monotonic은 해상도가 약 16ms라 perf_counter 사용)
    t0 = time.perf_counter()
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                # 속도는 답하지 않고 다음 프레임 판단부터 반영한다
                judge.speed_kmh = json.loads(msg["text"])["speed_kmh"]
                continue
            jpg = msg["bytes"]
            t_s = time.perf_counter() - t0
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            # YOLO 분석은 프레임당 수십 ms라, 그동안 다른 연결이 멈추지 않게 스레드에서 돌린다
            warnings = await asyncio.to_thread(judge, frame, t_s)
            recorder.submit(jpg, t_s, warnings)
            await websocket.send_json(
                {
                    "t_s": t_s,
                    "warnings": [asdict(w) for w in warnings],
                    **live_overlay(judge.last_result),
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        # 남은 클립 저장·주행 종료 (기다리는 동안 다른 연결이 멈추지 않게 스레드에서)
        await asyncio.to_thread(recorder.close)
