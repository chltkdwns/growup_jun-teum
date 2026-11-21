"""실시간 기록 테스트. .env의 DATABASE_URL(로컬 teum DB)에 실제로 쓰고, 끝나면 지운다."""

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

# DB 접속 정보가 없으면 이 파일만 건너뛴다 (test_nearmiss_api.py와 같은 방식)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
if not os.getenv("DATABASE_URL"):
    pytest.skip("DATABASE_URL 없음 (DB 테스트 건너뜀)", allow_module_level=True)

from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from app import main
from app.db import SessionLocal
from app.live_recorder import LiveRecorder
from app.main import app
from app.models import Clip, Device, Drive, Event
from app.nearmiss import ROOT
from app.schemas import NearMissWarning

client = TestClient(app)
FPS = 10
JPG = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))[1].tobytes()
CLIP_DIR = ROOT / "outputs" / "clips" / "_test_live"


@pytest.fixture
def device_id():
    """테스트 기기를 쓰고, 끝나면 기기째(주행·이벤트·클립 행 CASCADE) 지운다. 클립 파일도 지운다."""
    did = uuid.uuid4()
    yield did
    with SessionLocal() as s:
        keys = s.scalars(
            select(Clip.object_key)
            .join(Event)
            .join(Drive)
            .where(Drive.device_id == did)
        )
        for key in keys:
            (ROOT / key).unlink(missing_ok=True)
        dev = s.get(Device, did)
        if dev:
            s.delete(dev)
            s.commit()


def _drive_with_events(drive_id: int) -> tuple[Drive, list[Event]]:
    with SessionLocal() as s:
        drive = s.get(Drive, drive_id)
        events = list(drive.events)
        for e in events:
            _ = e.clip  # 세션 닫기 전에 읽어 둔다
        s.expunge_all()
        return drive, events


def test_live_recorder_saves_merges_and_filters(device_id):
    """10fps 3초 주행: 0.5초 주의(기록 안 함), 1.0초 위험, 1.5초 같은 종류 경고(합침).
    → 이벤트 1건, 클립 0.0~2.5초(앞 1초 설정이지만 합쳐서 뒤가 늘어남)."""
    rec = LiveRecorder(device_id, clip_dir=CLIP_DIR, pre_s=1.0, post_s=1.0)
    drive_id = rec.start()
    warnings = {
        5: [NearMissWarning(t_s=0.5, type="lead", level="caution", ttc_s=3.5)],
        10: [NearMissWarning(t_s=1.0, type="lead", level="danger", ttc_s=1.2)],
        15: [NearMissWarning(t_s=1.5, type="lead", level="warning", ttc_s=2.5)],
    }
    for i in range(31):
        rec.submit(JPG, i / FPS, warnings.get(i, []))
    rec.close()

    drive, events = _drive_with_events(drive_id)
    assert drive.source == "live"
    assert drive.ended_at is not None
    assert drive.frames == 31
    assert len(events) == 1  # 주의는 빠지고, 1.5초 경고는 1.0초 건에 합쳐짐
    e = events[0]
    assert e.level == "danger"
    assert e.min_ttc_s == pytest.approx(1.2)
    assert e.clip is not None
    assert e.clip.pre_s == pytest.approx(1.0)
    assert e.clip.post_s == pytest.approx(1.5)  # 1.5초 + 1초 = 2.5초까지
    assert (ROOT / e.clip.object_key).exists()


def test_live_recorder_flushes_on_close(device_id):
    """경고 직후 연결이 끊기면 있는 만큼이라도 저장한다."""
    rec = LiveRecorder(device_id, clip_dir=CLIP_DIR, pre_s=1.0, post_s=5.0)
    drive_id = rec.start()
    for i in range(11):
        w = [NearMissWarning(t_s=0.8, type="brake", level="warning")] if i == 8 else []
        rec.submit(JPG, i / FPS, w)
    rec.close()
    _, events = _drive_with_events(drive_id)
    assert len(events) == 1
    assert events[0].clip.post_s == pytest.approx(0.2)


class _FirstFrameDangerJudge:
    """첫 프레임에만 위험 경고 1개를 내는 가짜 판단 (프레임 내용은 안 봄)."""

    def __init__(self) -> None:
        self.calls = 0
        self.speed_kmh = None
        self.last_result = None  # 서버가 화면 표시용 박스·차선을 읽는 곳 (가짜라 비워 둠)
        self.speeds_seen: list[float | None] = []  # 프레임마다 그때의 내 속도 (속도 메시지 시험용)

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]:
        self.calls += 1
        self.speeds_seen.append(self.speed_kmh)
        if self.calls > 1:
            return []
        return [NearMissWarning(t_s=t_s, type="lead", level="danger", ttc_s=1.3)]


def test_ws_creates_live_drive_and_answers(device_id, monkeypatch):
    """/live: 연결하면 주행(live)이 생기고, 프레임마다 결과를 받고, 끊으면 주행이 끝난다.
    YOLO 없이 돌도록 첫 프레임에만 위험을 내는 판단으로 바꿔 끼우면, 이벤트가 저장되고 끊을 때 클립이 남는다."""
    monkeypatch.setattr(main, "make_judge", _FirstFrameDangerJudge)
    with client.websocket_connect(f"/live?device_id={device_id}") as ws:
        drive_id = ws.receive_json()["drive_id"]
        ws.send_bytes(JPG)
        first = ws.receive_json()
        ws.send_bytes(JPG)
        second = ws.receive_json()
    assert len(first["warnings"]) == 1
    assert first["warnings"][0]["level"] == "danger"
    assert second["warnings"] == []
    assert second["t_s"] > first["t_s"]

    drive, events = _drive_with_events(drive_id)
    assert drive.device_id == device_id
    assert drive.source == "live"
    assert drive.frames == 2
    assert drive.ended_at is not None
    assert len(events) == 1
    assert events[0].clip is not None


def test_ws_speed_message_reaches_judge(device_id, monkeypatch):
    """글자 메시지 {"speed_kmh"}는 답 없이 판단의 내 속도로 들어가고, 다음 프레임부터 쓰인다."""
    judge = _FirstFrameDangerJudge()
    monkeypatch.setattr(main, "make_judge", lambda: judge)
    with client.websocket_connect(f"/live?device_id={device_id}") as ws:
        ws.receive_json()  # drive_id
        ws.send_bytes(JPG)
        ws.receive_json()
        ws.send_text('{"speed_kmh": 87.5}')
        ws.send_bytes(JPG)
        ws.receive_json()  # 속도 메시지에는 답이 없어서, 바로 다음 답이 두 번째 프레임 것이다
        ws.send_text('{"speed_kmh": null}')
        ws.send_bytes(JPG)
        ws.receive_json()
    assert judge.speeds_seen == [None, 87.5, None]


def test_ws_requires_device_id():
    """기기 UUID가 없거나 이상하면 연결을 받지 않는다."""
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/live"):
        pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/live?device_id=not-a-uuid"):
            pass
