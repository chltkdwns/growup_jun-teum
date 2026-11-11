"""DB 표 4개를 파이썬 클래스로 옮긴 것. 표 자체는 schema.sql로 만든다(여기서 만들지 않음)."""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Device(Base):
    """앱 설치 1개 = 1행. 로그인 대신 UUID로 구분한다."""

    __tablename__ = "devices"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    nickname: Mapped[str | None] = mapped_column(Text)
    camera_cfg: Mapped[str] = mapped_column(Text, server_default="default")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Drive(Base):
    """주행 1회 또는 영상 파일 1개 = 1행."""

    __tablename__ = "drives"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    device_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("devices.id"))
    source: Mapped[str] = mapped_column(Text)  # 'live' | 'upload'
    video_name: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    ended_at: Mapped[datetime | None]
    distance_km: Mapped[float | None]
    avg_speed_kmh: Mapped[float | None]
    max_speed_kmh: Mapped[float | None]
    frames: Mapped[int | None]
    avg_latency_ms: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    events: Mapped[list["Event"]] = relationship(back_populates="drive")


class Event(Base):
    """아찔한 순간 1건. 경고(TTC 3초) 이상일 때만 저장한다(docs/SPEC.md)."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    drive_id: Mapped[int] = mapped_column(ForeignKey("drives.id"))
    occurred_at: Mapped[datetime]
    offset_s: Mapped[float]
    type: Mapped[str] = mapped_column(
        Text
    )  # lead, cut_in, brake, lane_departure, rear, two_wheeler
    level: Mapped[str] = mapped_column(Text)  # 'warning' | 'danger'
    min_ttc_s: Mapped[float | None]
    distance_m: Mapped[float | None]
    lateral_m: Mapped[float | None]
    speed_kmh: Mapped[float | None]
    lat: Mapped[float | None]
    lon: Mapped[float | None]
    target_class: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None]
    detail: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    drive: Mapped[Drive] = relationship(back_populates="events")
    clip: Mapped["Clip | None"] = relationship(back_populates="event")


class Clip(Base):
    """이벤트 영상 1개. 파일은 디스크/S3에 두고 여기엔 위치만 적는다."""

    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), unique=True)
    storage: Mapped[str] = mapped_column(Text, server_default="local")
    object_key: Mapped[str] = mapped_column(Text)
    thumb_key: Mapped[str | None] = mapped_column(Text)
    pre_s: Mapped[float]
    post_s: Mapped[float]
    fps: Mapped[float | None]
    width: Mapped[int | None]
    height: Mapped[int | None]
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    event: Mapped[Event] = relationship(back_populates="clip")
