"""PerceptionPipeline 테스트: 가짜 YOLO 모델과 가짜 도로 이미지로 순서·출력 키를 확인한다 (가중치 없이 실행)."""
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.pipeline import PerceptionPipeline

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))
W, H = 1280, 720


class _T:
    """요약: ultralytics 텐서 흉내 (.cpu().numpy(), .int().tolist())."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def int(self):
        return self

    def tolist(self):
        return self.a.astype(int).tolist()


class _FakeTrackModel:
    """요약: 매 프레임 같은 차 1대(track_id 7, 내 차선 가운데)를 돌려주는 가짜 추적 모델."""

    def __init__(self):
        self.calls = 0

    def track(self, img, **kwargs):
        self.calls += 1
        boxes = type("B", (), {"xyxy": _T([[560, 480, 720, 560]]), "cls": _T([2]), "conf": _T([0.9]), "id": _T([7])})()
        return [type("R", (), {"boxes": boxes})()]


def _road() -> np.ndarray:
    """요약: 흰 차선 두 줄이 그려진 가짜 도로 (test_lane_detect와 같은 모양)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.line(img, (300, 700), (600, 430), (255, 255, 255), 8)
    cv2.line(img, (980, 700), (680, 430), (255, 255, 255), 8)
    return img


def test_pipeline_output_keys_and_front():
    """요약: 한 프레임 결과에 묶음 키가 다 있고, 내 차선 가운데 차가 앞차로 잡히는지."""
    model = _FakeTrackModel()
    pipe = PerceptionPipeline(CFG, model=model)
    res = pipe.process(_road(), 0.0)
    assert set(res) == {"dets", "lanes", "offset", "lane_change", "front", "cut_in"}  # schemas.PerceptionResult
    assert res["lanes"]["left"] is not None and res["lanes"]["right"] is not None
    assert res["offset"]["lane_status"] == "ok"
    assert res["front"]["track_id"] == 7 and res["front"]["front_changed"] is True
    assert res["cut_in"] is None
    assert model.calls == 1


def test_pipeline_keeps_state_between_frames():
    """요약: 두 번째 프레임에서는 같은 앞차가 유지되고(front_changed False) 상태가 이어지는지."""
    pipe = PerceptionPipeline(CFG, model=_FakeTrackModel())
    pipe.process(_road(), 0.0)
    res = pipe.process(_road(), 0.1)
    assert res["front"]["track_id"] == 7 and res["front"]["front_changed"] is False
