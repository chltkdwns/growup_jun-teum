"""영상 업로드 분석 테스트. .env의 DATABASE_URL(로컬 teum DB)에 실제로 쓰고, 끝나면 지운다.

YOLO 가중치 없이 돌도록 경고 판단은 _ScriptedJudge(프레임을 안 봄)로 바꿔 끼운다.
"""

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

# DB 접속 정보가 없으면 이 파일만 건너뛴다 (test_live.py와 같은 방식)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
if not os.getenv("DATABASE_URL"):
    pytest.skip("DATABASE_URL 없음 (DB 테스트 건너뜀)", allow_module_level=True)

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import main
from app.db import SessionLocal
from app.models import Clip, Device, Drive, Event
from app.nearmiss import ROOT
from app.schemas import NearMissWarning
from app.upload_analyzer import UploadProgress, run_upload_job

client = TestClient(main.app)
FPS = 10
N_FRAMES = 30  # 3초짜리 영상
CLIP_DIR = ROOT / "outputs" / "clips" / "_test_upload"


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


@pytest.fixture
def video_path(tmp_path) -> Path:
    """10fps 3초짜리 작은 AVI (MJPG). 프레임마다 밝기를 달리한다."""
    path = tmp_path / "drive.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


class _ScriptedJudge:
    """정해 둔 프레임 시각에 정해 둔 경고를 내는 가짜 판단 (주의·위험·합침을 한 번에 시험)."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]:
        self.times.append(t_s)
        script = {
            0.5: NearMissWarning(t_s=t_s, type="lead", level="caution", ttc_s=3.5),
            1.0: NearMissWarning(t_s=t_s, type="lead", level="danger", ttc_s=1.2),
            1.5: NearMissWarning(t_s=t_s, type="lead", level="warning", ttc_s=2.5),
        }
        w = script.get(round(t_s, 1))
        return [w] if w else []


class _NoWarningJudge:
    """경고를 하나도 내지 않는 가짜 판단 (기록 권한만 시험할 때)."""

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]:
        return []


def _new_upload_drive(did: uuid.UUID) -> int:
    with SessionLocal() as s:
        s.add(Device(id=did))
        drive = Drive(device_id=did, source="upload", video_name="drive.avi")
        s.add(drive)
        s.commit()
        return drive.id


def _drive_with_events(drive_id: int) -> tuple[Drive, list[Event]]:
    with SessionLocal() as s:
        drive = s.get(Drive, drive_id)
        events = list(drive.events)
        for e in events:
            _ = e.clip  # 세션 닫기 전에 읽어 둔다
        s.expunge_all()
        return drive, events


def test_upload_job_saves_merges_and_filters(device_id, video_path):
    """주의(0.5초)는 빠지고, 위험(1.0초)과 같은 종류 경고(1.5초)는 1건으로 합쳐진다.
    클립은 앞 1초(0.0초부터) ~ 1.5초 + 뒤 1초 = 2.5초까지."""
    drive_id = _new_upload_drive(device_id)
    judge = _ScriptedJudge()
    progress = UploadProgress()
    run_upload_job(
        drive_id,
        video_path,
        lambda: judge,
        progress,
        clip_dir=CLIP_DIR,
        pre_s=1.0,
        post_s=1.0,
    )

    assert progress.status == "done", progress.error
    assert progress.frames_done == N_FRAMES
    # 촬영 시각은 프레임 수가 아니라 영상 속 시각이고, 늘어나기만 한다
    assert judge.times[0] == pytest.approx(0.0)
    assert judge.times[-1] == pytest.approx((N_FRAMES - 1) / FPS)
    assert all(b > a for a, b in zip(judge.times, judge.times[1:]))

    drive, events = _drive_with_events(drive_id)
    assert drive.ended_at is not None
    assert drive.frames == N_FRAMES
    assert len(events) == 1
    e = events[0]
    assert e.level == "danger"
    assert e.offset_s == pytest.approx(1.0)
    assert e.min_ttc_s == pytest.approx(1.2)
    assert e.clip.pre_s == pytest.approx(1.0)
    assert e.clip.post_s == pytest.approx(1.5)
    assert (ROOT / e.clip.object_key).exists()


def test_upload_job_fails_on_broken_file(device_id, tmp_path):
    """영상이 아닌 파일이면 멈추지 않고 failed와 이유를 남긴다. 주행은 끝나지 않은 채로 둔다."""
    drive_id = _new_upload_drive(device_id)
    bad = tmp_path / "not_video.avi"
    bad.write_bytes(b"not a video")
    progress = UploadProgress()
    run_upload_job(drive_id, bad, lambda: _ScriptedJudge(), progress, clip_dir=CLIP_DIR)
    assert progress.status == "failed"
    assert progress.error
    drive, _ = _drive_with_events(drive_id)
    assert drive.ended_at is None


def test_upload_api_runs_in_background(device_id, video_path, tmp_path, monkeypatch):
    """POST /drives/upload는 바로 답하고(queued), 분석이 끝나면 GET .../analysis가 done과 이벤트 수를 준다."""
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "make_judge", _ScriptedJudge)
    headers = {"X-Device-Id": str(device_id)}
    with video_path.open("rb") as f:
        res = client.post(
            "/drives/upload", files={"file": ("my drive.avi", f)}, headers=headers
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"
    drive_id = body["drive_id"]
    assert (
        tmp_path / "uploads" / f"{drive_id}.avi"
    ).exists()  # 원래 이름이 아니라 주행 번호로 저장

    # 분석 일꾼은 1명이라, 빈 일을 하나 더 맡겨 끝나길 기다리면 앞의 분석도 끝나 있다
    main.UPLOAD_POOL.submit(lambda: None).result()

    res = client.get(f"/drives/{drive_id}/analysis", headers=headers)
    assert res.status_code == 200
    a = res.json()
    assert a["status"] == "done", a["error"]
    assert a["frames_done"] == N_FRAMES
    assert a["events"] == 1
    drive, events = _drive_with_events(drive_id)
    assert drive.video_name == "my drive.avi"
    assert events[0].clip is not None


def test_analysis_hides_other_devices(device_id, video_path, tmp_path, monkeypatch):
    """다른 기기는 남의 분석 상황을 404로 본다. 기기 헤더가 없으면 422."""
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "make_judge", _NoWarningJudge)
    with video_path.open("rb") as f:
        res = client.post(
            "/drives/upload",
            files={"file": ("drive.avi", f)},
            headers={"X-Device-Id": str(device_id)},
        )
    drive_id = res.json()["drive_id"]
    main.UPLOAD_POOL.submit(lambda: None).result()

    other = {"X-Device-Id": str(uuid.uuid4())}
    assert client.get(f"/drives/{drive_id}/analysis", headers=other).status_code == 404
    assert client.get(f"/drives/{drive_id}/analysis").status_code == 422
    with video_path.open("rb") as f:
        assert (
            client.post("/drives/upload", files={"file": ("x.avi", f)}).status_code
            == 422
        )
