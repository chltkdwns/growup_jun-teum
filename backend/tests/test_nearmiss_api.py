"""아찔한 순간 API 테스트. .env의 DATABASE_URL(로컬 teum DB)에 실제로 쓰고, 끝나면 지운다."""

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.clip_recorder import ClipRecorder
from app.db import SessionLocal
from app.main import app
from app.models import Device, Event
from app.nearmiss import EventIn, find_merge_target

client = TestClient(app)
T0 = datetime(2026, 11, 11, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def device_id():
    """테스트용 기기를 쓰고, 끝나면 기기째 지운다 (주행·이벤트는 CASCADE로 같이 지워짐)."""
    did = uuid.uuid4()
    yield did
    with SessionLocal() as s:
        dev = s.get(Device, did)
        if dev:
            s.delete(dev)
            s.commit()


def _event(
    sec: float, level: str = "warning", ttc: float = 2.5, type_: str = "lead"
) -> dict:
    return {
        "occurred_at": (T0 + timedelta(seconds=sec)).isoformat(),
        "offset_s": sec,
        "type": type_,
        "level": level,
        "min_ttc_s": ttc,
    }


def test_merge_boundary():
    """정확히 merge_gap_s 차이면 합치고, 넘으면 새 건이다."""
    last = Event(occurred_at=T0, duration_s=0.0)
    at_gap = EventIn(**_event(5.0))
    over_gap = EventIn(**_event(5.1))
    assert find_merge_target(last, at_gap, 5.0) is last
    assert find_merge_target(last, over_gap, 5.0) is None
    assert find_merge_target(None, at_gap, 5.0) is None


def test_save_merge_and_list(device_id):
    drive_id = client.post(
        "/drives",
        json={"device_id": str(device_id), "source": "upload", "video_name": "a.avi"},
    ).json()["drive_id"]

    r1 = client.post(
        f"/drives/{drive_id}/events", json=_event(0, "warning", 2.5)
    ).json()
    r2 = client.post(f"/drives/{drive_id}/events", json=_event(2, "danger", 1.2)).json()
    r3 = client.post(
        f"/drives/{drive_id}/events", json=_event(3, "warning", 2.0, "cut_in")
    ).json()
    r4 = client.post(
        f"/drives/{drive_id}/events", json=_event(20, "warning", 2.8)
    ).json()

    assert r1["merged"] is False
    assert r2 == {"event_id": r1["event_id"], "merged": True}  # 2초 뒤 같은 종류 → 합침
    assert r3["merged"] is False  # 종류가 다르면 따로
    assert r4["merged"] is False  # 간격이 길면 따로

    events = client.get("/events", params={"device_id": str(device_id)}).json()
    assert len(events) == 3
    merged = next(e for e in events if e["id"] == r1["event_id"])
    assert merged["level"] == "danger"  # 더 높은 단계가 남음
    assert merged["min_ttc_s"] == pytest.approx(1.2)
    assert merged["duration_s"] == pytest.approx(2.0)


def test_bad_level_rejected(device_id):
    drive_id = client.post(
        "/drives", json={"device_id": str(device_id), "source": "upload"}
    ).json()["drive_id"]
    r = client.post(f"/drives/{drive_id}/events", json=_event(0, level="caution"))
    assert r.status_code == 422  # 주의 단계는 기록 대상이 아님


def test_clip_saved_and_served(device_id):
    """이벤트 → 클립 저장 → clips 표 기록 → 목록에 clip_key, 파일 API로 받기까지."""
    from app.nearmiss import ROOT, save_clip

    drive_id = client.post(
        "/drives", json={"device_id": str(device_id), "source": "upload"}
    ).json()["drive_id"]
    event_id = client.post(f"/drives/{drive_id}/events", json=_event(1.0)).json()[
        "event_id"
    ]

    clip_dir = ROOT / "outputs" / "clips" / "_test"
    rec = ClipRecorder(clip_dir, pre_s=1.0, post_s=1.0)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    saved = []
    for i in range(31):  # 10fps로 3초
        t_s = i / 10
        if i == 10:
            rec.trigger(event_id, t_s, f"{drive_id}/{event_id}")
        saved += rec.push(frame, t_s)
    assert len(saved) == 1

    with SessionLocal() as s:
        row = save_clip(s, saved[0])
        key = row.object_key
    try:
        assert key == f"outputs/clips/_test/{drive_id}/{event_id}.mp4"
        events = client.get("/events", params={"device_id": str(device_id)}).json()
        assert events[0]["clip_key"] == key
        r = client.get(f"/events/{event_id}/clip")
        assert r.status_code == 200
        assert r.headers["content-type"] == "video/mp4"
    finally:
        saved[0].path.unlink()
