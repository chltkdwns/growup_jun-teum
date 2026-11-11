"""ClipRecorder 테스트. 10fps 가짜 프레임으로 구간 계산을 확인한다."""

import cv2
import numpy as np
import pytest

from app.clip_recorder import ClipRecorder

FPS = 10
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def _run(rec: ClipRecorder, until_s: float, triggers: dict[float, int]) -> list:
    """0초부터 until_s까지 프레임을 넣고, triggers의 시각에 trigger를 부른다."""
    saved = []
    for i in range(int(until_s * FPS) + 1):
        t_s = i / FPS
        if t_s in triggers:
            rec.trigger(triggers[t_s], t_s, f"d/{triggers[t_s]}")
        saved += rec.push(FRAME, t_s)
    return saved


def test_pre_and_post(tmp_path):
    """12초 경고 → 7초~17초가 저장된다."""
    rec = ClipRecorder(tmp_path, pre_s=5.0, post_s=5.0)
    saved = _run(rec, 30.0, {12.0: 7})
    assert len(saved) == 1
    clip = saved[0]
    assert clip.pre_s == pytest.approx(5.0)
    assert clip.post_s == pytest.approx(5.0)
    cap = cv2.VideoCapture(str(clip.path))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 10 * FPS + 1


def test_merged_event_extends(tmp_path):
    """같은 이벤트가 15초에 다시 오면 끝이 20초로 늘어난다."""
    rec = ClipRecorder(tmp_path, pre_s=5.0, post_s=5.0)
    saved = _run(rec, 30.0, {12.0: 7, 15.0: 7})
    assert len(saved) == 1
    assert saved[0].post_s == pytest.approx(8.0)  # 12초 기준 20초까지


def test_start_of_video(tmp_path):
    """영상 2초 지점 경고면 앞부분은 0초부터 2초만 담긴다."""
    rec = ClipRecorder(tmp_path, pre_s=5.0, post_s=5.0)
    saved = _run(rec, 10.0, {2.0: 1})
    assert saved[0].pre_s == pytest.approx(2.0)


def test_flush_at_end(tmp_path):
    """영상이 경고 직후 끝나면 flush로 있는 만큼 저장한다."""
    rec = ClipRecorder(tmp_path, pre_s=5.0, post_s=5.0)
    assert _run(rec, 10.0, {8.0: 3}) == []
    saved = rec.flush()
    assert saved[0].post_s == pytest.approx(2.0)
    assert saved[0].path.exists()
