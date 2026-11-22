"""live_overlay: 분석 결과 → 앱이 그릴 박스·차선 (DB·YOLO 없이)."""

import json

import numpy as np

from app.live_overlay import live_overlay


def test_marks_front_and_converts_numpy():
    """앞차만 is_front이고, numpy 숫자도 JSON으로 보낼 수 있게 바뀐다."""
    dets = [
        {"track_id": 3, "cls_name": "car", "box_xyxy_px": np.array([10, 20, 110, 90], dtype=np.float32)},
        {"track_id": 7, "cls_name": "motorcycle", "box_xyxy_px": (200.5, 30.0, 240.0, 80.0)},
    ]
    result = {
        "dets": dets,
        "front": {"track_id": 3},
        "lanes": {"left": (np.int64(100), 360, 280, 200), "right": None},
    }
    out = live_overlay(result)
    json.dumps(out)  # numpy가 남아 있으면 여기서 실패
    assert [b["is_front"] for b in out["boxes"]] == [True, False]
    assert out["boxes"][0]["box_xyxy_px"] == [10.0, 20.0, 110.0, 90.0]
    assert out["lanes"] == {"left": [100, 360, 280, 200], "right": None}


def test_no_front_and_no_track_id():
    """앞차가 없으면 아무 박스도 앞차가 아니다. 추적 ID가 없는 박스도 앞차로 보지 않는다."""
    result = {
        "dets": [{"track_id": None, "cls_name": "car", "box_xyxy_px": (0, 0, 1, 1)}],
        "front": None,
        "lanes": {"left": None, "right": None},
    }
    assert live_overlay(result)["boxes"][0]["is_front"] is False


def test_before_first_result():
    """분석 결과가 아직 없으면 빈 값."""
    assert live_overlay(None) == {"boxes": [], "lanes": {"left": None, "right": None}}
