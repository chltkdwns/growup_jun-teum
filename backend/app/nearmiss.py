"""아찔한 순간 기록 로직. API(main.py)와 영상 분석 루프가 같은 함수를 쓴다 (FastAPI를 import하지 않음)."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.clip_recorder import SavedClip
from app.models import Clip, Event

ROOT = Path(__file__).resolve().parents[2]  # 레포 루트 (클립 경로의 기준)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"  # backend/config.yaml
NEARMISS_CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["nearmiss"]

# 단계 순서: 합칠 때 더 높은 단계를 남긴다
LEVEL_RANK = {"warning": 1, "danger": 2}

EventType = Literal["lead", "cut_in", "brake", "lane_departure", "rear", "two_wheeler"]
Level = Literal["warning", "danger"]


class EventIn(BaseModel):
    """경고 판단이 넘겨주는 아찔한 순간 1건."""

    occurred_at: datetime
    offset_s: float
    type: EventType
    level: Level
    min_ttc_s: float | None = None
    distance_m: float | None = None
    lateral_m: float | None = None
    speed_kmh: float | None = None
    lat: float | None = None
    lon: float | None = None
    target_class: str | None = None
    duration_s: float | None = None
    detail: dict = {}


def find_merge_target(
    last: Event | None, new: EventIn, merge_gap_s: float
) -> Event | None:
    """새 이벤트를 직전 이벤트에 합칠지 정한다.

    사용처: 5단계(기록). record_event가 저장 전에 호출한다.
    입력: last 같은 주행·같은 종류의 가장 최근 이벤트(없으면 None), new 새 이벤트,
          merge_gap_s 합치는 간격(초, config: nearmiss.merge_gap_s)
    출력: 합칠 대상 Event, 새로 만들어야 하면 None
    주의: 시간은 프레임 수가 아니라 촬영 시각(occurred_at) 기준이다.
    """
    if last is None:
        return None
    # 직전 이벤트가 끝난 시각부터 잰다 (경고가 길게 이어진 경우를 위해)
    last_end_s = last.occurred_at.timestamp() + (last.duration_s or 0.0)
    gap_s = new.occurred_at.timestamp() - last_end_s
    return last if gap_s <= merge_gap_s else None


def record_event(session: Session, drive_id: int, body: EventIn) -> tuple[Event, bool]:
    """아찔한 순간을 저장한다. 같은 종류가 merge_gap_s 안에 또 오면 기존 건에 합친다.

    사용처: 5단계(기록). POST /drives/{id}/events와 영상 분석 루프.
    입력: session DB 세션, drive_id 주행 번호, body 이벤트
    출력: (저장되거나 합쳐진 Event, 합쳤는지 여부)
    주의: 커밋까지 한다. 반환된 event.id로 ClipRecorder.trigger를 부른다.
    """
    last = session.scalars(
        select(Event)
        .where(Event.drive_id == drive_id, Event.type == body.type)
        .order_by(Event.occurred_at.desc())
        .limit(1)
    ).first()
    target = find_merge_target(last, body, NEARMISS_CFG["merge_gap_s"])

    if target is None:
        event = Event(drive_id=drive_id, **body.model_dump())
        session.add(event)
        session.commit()
        return event, False

    # 합치기: 더 위험한 값(높은 단계, 낮은 TTC)을 남기고 지속 시간을 늘린다
    if LEVEL_RANK[body.level] > LEVEL_RANK[target.level]:
        target.level = body.level
    if body.min_ttc_s is not None and (
        target.min_ttc_s is None or body.min_ttc_s < target.min_ttc_s
    ):
        target.min_ttc_s = body.min_ttc_s
        target.distance_m = body.distance_m
    target.duration_s = (
        body.occurred_at.timestamp()
        - target.occurred_at.timestamp()
        + (body.duration_s or 0.0)
    )
    session.commit()
    return target, True


def save_clip(session: Session, clip: SavedClip) -> Clip:
    """저장이 끝난 클립 파일을 clips 표에 기록한다.

    사용처: 5단계(기록). ClipRecorder.push()/flush()가 SavedClip을 돌려줄 때마다.
    입력: session DB 세션, clip ClipRecorder가 만든 클립 정보
    출력: 저장된 Clip 행
    주의: object_key는 레포 루트 기준 경로다 (나중에 S3로 가면 버킷 안 경로로 바뀜).
    """
    row = Clip(
        event_id=clip.event_id,
        storage="local",
        object_key=clip.path.resolve().relative_to(ROOT).as_posix(),
        pre_s=clip.pre_s,
        post_s=clip.post_s,
        fps=clip.fps,
        width=clip.width,
        height=clip.height,
        size_bytes=clip.path.stat().st_size,
        status="ready",
    )
    session.add(row)
    session.commit()
    return row
