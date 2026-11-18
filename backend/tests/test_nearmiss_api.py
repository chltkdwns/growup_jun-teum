"""아찔한 순간 API 테스트. .env의 DATABASE_URL(로컬 teum DB)에 실제로 쓰고, 끝나면 지운다."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv

# PostgreSQL을 안 깐 팀원도 pytest가 돌게, DB 접속 정보가 없으면 이 파일만 건너뛴다
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
if not os.getenv("DATABASE_URL"):
    pytest.skip("DATABASE_URL 없음 (DB 테스트 건너뜀)", allow_module_level=True)

# 아래 import는 DB가 필요해서 건너뛰기 판단 뒤에 둔다
from fastapi.testclient import TestClient

from app.clip_recorder import ClipRecorder
from app.db import SessionLocal
from app.main import app
from app.models import Device, Event
from app.nearmiss import EventIn, find_merge_target

client = TestClient(app)
T0 = datetime(2026, 11, 11, 10, 0, 0, tzinfo=timezone.utc)


def _cleanup_device(did: uuid.UUID) -> None:
    """테스트 기기를 지운다 (주행·이벤트는 CASCADE로 같이 지워짐)."""
    with SessionLocal() as s:
        dev = s.get(Device, did)
        if dev:
            s.delete(dev)
            s.commit()


@pytest.fixture
def device_id():
    """테스트용 기기를 쓰고, 끝나면 기기째 지운다."""
    did = uuid.uuid4()
    yield did
    _cleanup_device(did)


@pytest.fixture
def other_device_id():
    """남의 기록을 못 보는지 확인할 때 쓰는 두 번째 기기."""
    did = uuid.uuid4()
    yield did
    _cleanup_device(did)


def _h(did: uuid.UUID) -> dict:
    """앱이 모든 요청에 붙이는 기기 헤더."""
    return {"X-Device-Id": str(did)}


def _new_drive(did: uuid.UUID, **body) -> int:
    return client.post(
        "/drives", json={"source": "upload", **body}, headers=_h(did)
    ).json()["drive_id"]


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
    drive_id = _new_drive(device_id, video_name="a.avi")
    url, h = f"/drives/{drive_id}/events", _h(device_id)

    r1 = client.post(url, json=_event(0, "warning", 2.5), headers=h).json()
    r2 = client.post(url, json=_event(2, "danger", 1.2), headers=h).json()
    r3 = client.post(url, json=_event(3, "warning", 2.0, "cut_in"), headers=h).json()
    r4 = client.post(url, json=_event(20, "warning", 2.8), headers=h).json()

    assert r1["merged"] is False
    assert r2 == {"event_id": r1["event_id"], "merged": True}  # 2초 뒤 같은 종류 → 합침
    assert r3["merged"] is False  # 종류가 다르면 따로
    assert r4["merged"] is False  # 간격이 길면 따로

    events = client.get("/events", headers=h).json()
    assert len(events) == 3
    merged = next(e for e in events if e["id"] == r1["event_id"])
    assert merged["level"] == "danger"  # 더 높은 단계가 남음
    assert merged["min_ttc_s"] == pytest.approx(1.2)
    assert merged["duration_s"] == pytest.approx(2.0)


def test_bad_level_rejected(device_id):
    drive_id = _new_drive(device_id)
    r = client.post(
        f"/drives/{drive_id}/events",
        json=_event(0, level="caution"),
        headers=_h(device_id),
    )
    assert r.status_code == 422  # 주의 단계는 기록 대상이 아님


def test_missing_device_header_rejected():
    """기기 헤더가 없으면 목록·주행·클립 모두 거절한다."""
    assert client.get("/events").status_code == 422
    assert client.post("/drives", json={"source": "upload"}).status_code == 422
    assert client.get("/events/1/clip").status_code == 422
    bad = {"X-Device-Id": "not-a-uuid"}
    assert client.get("/events", headers=bad).status_code == 422


def test_other_device_blocked(device_id, other_device_id):
    """다른 기기는 내 주행에 이벤트를 못 넣고, 내 목록도 못 본다."""
    drive_id = _new_drive(device_id)
    mine = client.post(
        f"/drives/{drive_id}/events", json=_event(0), headers=_h(device_id)
    )
    assert mine.status_code == 200

    r = client.post(
        f"/drives/{drive_id}/events", json=_event(1), headers=_h(other_device_id)
    )
    assert r.status_code == 404  # 남의 주행은 없는 주행처럼
    assert client.get("/events", headers=_h(other_device_id)).json() == []
    assert len(client.get("/events", headers=_h(device_id)).json()) == 1


def test_clip_saved_and_served(device_id, other_device_id):
    """이벤트 → 클립 저장 → clips 표 기록 → 목록에 clip_key, 파일 API로 받기까지.
    다른 기기는 같은 이벤트 번호로 클립을 받을 수 없다."""
    from app.nearmiss import ROOT, save_clip

    drive_id = _new_drive(device_id)
    event_id = client.post(
        f"/drives/{drive_id}/events", json=_event(1.0), headers=_h(device_id)
    ).json()["event_id"]

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
        events = client.get("/events", headers=_h(device_id)).json()
        assert events[0]["clip_key"] == key
        r = client.get(f"/events/{event_id}/clip", headers=_h(device_id))
        assert r.status_code == 200
        assert r.headers["content-type"] == "video/mp4"
        # 이벤트 번호를 알아도 다른 기기면 클립이 없는 것처럼 404
        r = client.get(f"/events/{event_id}/clip", headers=_h(other_device_id))
        assert r.status_code == 404
    finally:
        saved[0].path.unlink()


def test_list_drives_summary(device_id, other_device_id):
    """GET /drives: 주행마다 아찔한 순간 수·위험 수·최저 TTC. 이벤트 없는 주행은 0건, 남의 주행은 안 보인다."""
    empty_id = _new_drive(device_id)
    busy_id = _new_drive(device_id)
    for sec, level, ttc in [
        (0, "warning", 2.5),
        (20, "danger", 1.2),
    ]:  # 5초 넘게 떨어져 2건
        client.post(
            f"/drives/{busy_id}/events",
            json=_event(sec, level, ttc),
            headers=_h(device_id),
        )
    _new_drive(other_device_id)

    drives = {d["id"]: d for d in client.get("/drives", headers=_h(device_id)).json()}
    assert set(drives) == {empty_id, busy_id}
    assert drives[empty_id]["events"] == 0
    assert drives[empty_id]["worst_ttc_s"] is None
    assert drives[busy_id]["events"] == 2
    assert drives[busy_id]["danger_events"] == 1
    assert drives[busy_id]["worst_ttc_s"] == pytest.approx(1.2)
    assert drives[busy_id]["duration_s"] is None  # upload는 영상 길이를 모른다
