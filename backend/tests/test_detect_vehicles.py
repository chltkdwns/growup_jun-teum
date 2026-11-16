"""차량 탐지 테스트: 가짜 YOLO 결과로 변환·필터·보닛 자르기를 확인한다 (가중치 없이 실행)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.detect.vehicles import (
    ClassVoter,
    detect_vehicles,
    load_detector,
    to_detections,
)

BACKEND = Path(__file__).parents[1]
CFG = yaml.safe_load((BACKEND / "config.yaml").read_text(encoding="utf-8"))

XYXY = np.array([[10, 20, 110, 90], [200, 50, 260, 120], [300, 60, 340, 100], [400, 70, 450, 130]], dtype=float)
CLS = np.array([2, 0, 3, 7])  # car, person(버려야 함), motorcycle, truck
SCORES = np.array([0.9, 0.95, 0.6, 0.2])  # truck은 확신도가 낮아 버려야 함


def test_keeps_only_vehicle_classes_above_score():
    """요약: 사람(0)과 확신도 낮은 truck은 버리고 car·motorcycle만 남는지."""
    dets = to_detections(XYXY, CLS, SCORES, None, 1.5, score_min=0.25)
    assert [d["cls_name"] for d in dets] == ["car", "motorcycle"]


def test_output_keys_and_types():
    """요약: 출력 키·타입이 약속한 형식인지 (track_id, cls_name, score, box_xyxy_px, t_s)."""
    det = to_detections(XYXY, CLS, SCORES, [7, 8, 9, 10], 1.5, score_min=0.25)[0]
    assert det == {"track_id": 7, "cls_name": "car", "score": 0.9, "box_xyxy_px": (10.0, 20.0, 110.0, 90.0), "t_s": 1.5}
    assert isinstance(det["score"], float) and isinstance(det["box_xyxy_px"][0], float)


def test_track_id_none_when_not_tracking():
    """요약: 추적을 끄면 track_id가 None."""
    assert to_detections(XYXY, CLS, SCORES, None, 0.0, score_min=0.25)[0]["track_id"] is None


class _FakeBoxes:
    """요약: ultralytics Boxes 흉내 (xyxy·cls·conf·id 와 .cpu().numpy())."""

    class _T:
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

    def __init__(self):
        self.xyxy, self.cls, self.conf, self.id = self._T(XYXY), self._T(CLS), self._T(SCORES), None


class _FakeModel:
    """요약: 받은 이미지 크기를 기록하고 가짜 결과를 돌려주는 모델."""

    def __init__(self):
        self.seen_shape = None
        self.seen_kwargs = None

    def predict(self, img, **kwargs):
        self.seen_shape = img.shape
        self.seen_kwargs = kwargs
        return [type("R", (), {"boxes": _FakeBoxes()})()]


def test_bonnet_is_cut_before_detection():
    """요약: 모델에는 보닛 아래(bonnet_ratio 0.85 → 720px 중 612px까지)만 들어가는지."""
    model = _FakeModel()
    cfg_detect = dict(CFG["detect"], track=False)
    dets = detect_vehicles(np.zeros((720, 1280, 3), np.uint8), model, CFG["camera"], cfg_detect, 0.0)
    assert model.seen_shape[:2] == (612, 1280)
    assert len(dets) == 2


def test_agnostic_nms_passed_to_model():
    """요약: config의 agnostic_nms 값이 YOLO 호출에 그대로 전달되는지 (클래스 달라도 겹친 박스 합치기)."""
    model = _FakeModel()
    cfg_detect = dict(CFG["detect"], track=False, agnostic_nms=True)
    detect_vehicles(np.zeros((720, 1280, 3), np.uint8), model, CFG["camera"], cfg_detect, 0.0)
    assert model.seen_kwargs["agnostic_nms"] is True


@pytest.mark.skipif(not (BACKEND / CFG["detect"]["weights_path"]).is_file(), reason="YOLO 가중치 없음")
def test_real_model_runs_on_blank_frame():
    """요약: 실제 가중치로 빈 프레임을 돌리면 오류 없이 빈 목록 (통합 확인, 가중치 있을 때만)."""
    cfg_detect = dict(CFG["detect"], weights_path=str(BACKEND / CFG["detect"]["weights_path"]), track=False)
    model = load_detector(cfg_detect)
    assert detect_vehicles(np.zeros((720, 1280, 3), np.uint8), model, CFG["camera"], cfg_detect, 0.0) == []


VOTE_CFG = {"class_vote_window_s": 1.0}


def _d(tid, cls_name: str, score: float = 0.9) -> dict:
    """요약: cls_name만 다른 가짜 탐지."""
    return {"track_id": tid, "cls_name": cls_name, "score": score, "box_xyxy_px": (0, 0, 10, 10), "t_s": 0.0}


def test_class_vote_keeps_majority():
    """요약: car 판정이 쌓인 차가 한 프레임 truck으로 나와도 car로 유지."""
    v = ClassVoter(VOTE_CFG)
    for t in (0.0, 0.1, 0.2):
        v.update([_d(1, "car")], t)
    assert v.update([_d(1, "truck")], 0.3)[0]["cls_name"] == "car"


def test_class_vote_weighted_by_score():
    """요약: 다수결은 확신도 합으로 한다 (확신도 낮은 판정 여러 번 < 확신도 높은 판정 한 번)."""
    v = ClassVoter(VOTE_CFG)
    v.update([_d(1, "motorcycle", 0.9)], 0.0)
    v.update([_d(1, "car", 0.3)], 0.1)
    assert v.update([_d(1, "car", 0.3)], 0.2)[0]["cls_name"] == "motorcycle"


def test_class_vote_forgets_old_votes():
    """요약: 1초보다 오래된 판정은 다수결에서 빠진다."""
    v = ClassVoter(VOTE_CFG)
    for t in (0.0, 0.1, 0.2):
        v.update([_d(1, "car")], t)
    assert v.update([_d(1, "truck")], 1.5)[0]["cls_name"] == "truck"


def test_class_vote_untracked_unchanged():
    """요약: track_id가 None이면 그대로 둔다."""
    det = _d(None, "bus")
    assert ClassVoter(VOTE_CFG).update([det], 0.0) == [det]


def test_score_min_by_class_lowers_only_that_class():
    """요약: 오토바이만 기준을 0.15로 낮추면, 확신도 0.2인 오토바이는 남고 같은 확신도의 차는 버려진다."""
    xyxy = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
    dets = to_detections(xyxy, np.array([3, 2]), np.array([0.2, 0.2]), None, 0.0, 0.25, {"motorcycle": 0.15})
    assert [d["cls_name"] for d in dets] == ["motorcycle"]


def test_yolo_conf_is_lowest_threshold():
    """요약: YOLO에는 클래스별 기준 중 가장 낮은 값을 넘겨야 낮은 확신도 오토바이를 받을 수 있다."""
    model = _FakeModel()
    cfg_detect = dict(CFG["detect"], track=False, score_min=0.25, score_min_by_class={"motorcycle": 0.15})
    detect_vehicles(np.zeros((720, 1280, 3), np.uint8), model, CFG["camera"], cfg_detect, 0.0)
    assert model.seen_kwargs["conf"] == 0.15
