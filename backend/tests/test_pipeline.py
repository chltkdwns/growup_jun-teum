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


class _FakeBrakeModel:
    """요약: 매 프레임 앞차와 같은 자리에 "브레이크등 켜짐"(확신도 0.9) 박스 1개를 돌려주는 가짜 브레이크등 모델."""

    def __init__(self):
        self.calls = 0

    def predict(self, img, **kwargs):
        self.calls += 1
        boxes = type("B", (), {"xyxy": _T([[560, 480, 720, 560]]), "cls": _T([1]), "conf": _T([0.9])})()
        return [type("R", (), {"boxes": boxes, "names": {0: "car_BrakeOff", 1: "car_BrakeOn"}})()]


def _cfg_with(section: str, **values) -> dict:
    """요약: CFG를 복사해 한 섹션의 값만 바꾼다 (원본 CFG는 그대로)."""
    return {**CFG, section: {**CFG[section], **values}}


def _road() -> np.ndarray:
    """요약: 흰 차선 두 줄이 그려진 가짜 도로 (test_lane_detect와 같은 모양)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.line(img, (300, 700), (600, 430), (255, 255, 255), 8)
    cv2.line(img, (980, 700), (680, 430), (255, 255, 255), 8)
    return img


def test_pipeline_output_keys_and_front():
    """요약: 한 프레임 결과에 묶음 키가 다 있고, 내 차선 가운데 차가 앞차로 잡히는지."""
    model, brake_model = _FakeTrackModel(), _FakeBrakeModel()
    pipe = PerceptionPipeline(CFG, model=model, brake_model=brake_model)
    res = pipe.process(_road(), 0.0)
    assert set(res) == {"dets", "lanes", "offset", "lane_change", "front", "brake", "cut_in"}  # schemas.PerceptionResult
    assert res["lanes"]["left"] is not None and res["lanes"]["right"] is not None
    assert res["offset"]["lane_status"] == "ok"
    assert res["front"]["track_id"] == 7 and res["front"]["front_changed"] is True
    assert res["cut_in"] is None
    # 가짜 도로는 빨간 등이 없어서 모델은 켜짐이라도 미등 걸러내기로 최종 꺼짐
    assert res["brake"]["model_on"] is True and res["brake"]["brake_score"] == 0.9 and res["brake"]["brake_on"] is False
    assert model.calls == 1 and brake_model.calls == 1


def test_pipeline_keeps_state_between_frames():
    """요약: 두 번째 프레임에서는 같은 앞차가 유지되고(front_changed False) 상태가 이어지는지."""
    pipe = PerceptionPipeline(CFG, model=_FakeTrackModel(), brake_model=_FakeBrakeModel())
    pipe.process(_road(), 0.0)
    res = pipe.process(_road(), 0.1)
    assert res["front"]["track_id"] == 7 and res["front"]["front_changed"] is False


def test_pipeline_without_brake_weights_gives_none():
    """요약: 브레이크등 모델도 없고 가중치 파일도 없으면 브레이크등 단계만 꺼지고 나머지는 그대로 나온다."""
    cfg = _cfg_with("brake", weights_path="weights/__no_such_brake__.pt")
    res = PerceptionPipeline(cfg, model=_FakeTrackModel()).process(_road(), 0.0)
    assert res["brake"] is None and res["front"]["track_id"] == 7


def test_brake_runs_every_n_frames_and_reuses_last_result():
    """요약: brake_every_n_frames=3이면 0·3번째 프레임에서만 모델을 돌리고, 사이 프레임은 직전 결과를 그대로 쓴다."""
    brake_model = _FakeBrakeModel()
    pipe = PerceptionPipeline(_cfg_with("pipeline", brake_every_n_frames=3), model=_FakeTrackModel(), brake_model=brake_model)
    results = [pipe.process(_road(), t)["brake"] for t in (0.0, 0.1, 0.2, 0.3)]
    assert brake_model.calls == 2
    assert results[1] is results[0] and results[2] is results[0] and results[3] is not results[0]
