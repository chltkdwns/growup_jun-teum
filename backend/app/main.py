"""아찔한 순간 기록 API (1차: 영상 파일 모드 저장·목록). 로직은 nearmiss.py에 있다."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Clip, Device, Drive, Event
from app.nearmiss import ROOT, EventIn, record_event

# FastAPI 권장 방식: 요청마다 DB 세션을 주입받는 타입
SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI(title="teum")


class DriveIn(BaseModel):
    device_id: uuid.UUID
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


@app.post("/drives")
def create_drive(body: DriveIn, session: SessionDep) -> dict:
    """주행(또는 영상 파일 분석) 1건을 시작한다. 처음 보는 기기면 기기도 등록한다."""
    if session.get(Device, body.device_id) is None:
        session.add(Device(id=body.device_id))
    drive = Drive(
        device_id=body.device_id, source=body.source, video_name=body.video_name
    )
    session.add(drive)
    session.commit()
    return {"drive_id": drive.id}


@app.post("/drives/{drive_id}/events")
def create_event(drive_id: int, body: EventIn, session: SessionDep) -> dict:
    """아찔한 순간을 저장한다. 같은 종류가 merge_gap_s 안에 또 오면 기존 건에 합친다."""
    if session.get(Drive, drive_id) is None:
        raise HTTPException(status_code=404, detail="drive not found")
    event, merged = record_event(session, drive_id, body)
    return {"event_id": event.id, "merged": merged}


@app.get("/events")
def list_events(
    device_id: uuid.UUID, session: SessionDep, limit: int = 50
) -> list[EventOut]:
    """한 기기의 아찔한 순간 목록 (최근 것부터). 목록 화면에서 쓴다."""
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
def get_clip(event_id: int, session: SessionDep) -> FileResponse:
    """이벤트의 클립 영상 파일을 돌려준다. 앱 목록에서 눌렀을 때 재생용."""
    clip = session.scalars(select(Clip).where(Clip.event_id == event_id)).first()
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return FileResponse(ROOT / clip.object_key, media_type="video/mp4")
