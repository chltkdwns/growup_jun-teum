"""브레이크등 테스트: 가짜 모델 결과와 그림으로 그린 차로 짝짓기·미등 걸러내기·다수결을 확인한다 (가중치 없이 실행)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from app.signal.brake import (
    BrakeStateTracker,
    box_iou,
    lamp_signals,
    load_brake_model,
    match_front_brake,
    to_brake_boxes,
)

BACKEND = Path(__file__).parents[1]
CFG = yaml.safe_load((BACKEND / "config.yaml").read_text(encoding="utf-8"))["brake"]

BOX = (100.0, 100.0, 200.0, 200.0)  # 그림 속 앞차 박스
FRONT = {"track_id": 1, "box_xyxy_px": BOX}
BGR_RED = (0, 0, 255)
BGR_DIM_RED = (0, 0, 120)  # 미등처럼 어두운 빨강 (밝기 기준 red_v_min 아래)


def _car(third: bool = False, side: tuple = BGR_DIM_RED) -> np.ndarray:
    """요약: 회색 배경에 차 박스를 두고, 좌우 등(side 색)과 보조 제동등(third=True면 빨간 줄)을 그린 프레임."""
    f = np.full((300, 300, 3), 60, np.uint8)
    f[135:160, 103:127] = side  # 왼쪽 등 (박스 세로 25~65%, 가로 0~30% 안)
    f[135:160, 173:197] = side  # 오른쪽 등
    if third:
        f[110:115, 135:165] = BGR_RED  # 보조 제동등 (박스 위쪽 0~30%, 가운데 30~70% 안)
    return f


def _matched(model_on: bool = True, score: float = 0.9) -> dict:
    """요약: match_front_brake() 형식의 가짜 결과."""
    return {"box_xyxy_px": BOX, "model_on": model_on, "score": score}


def test_to_brake_boxes_reads_on_class_name():
    """요약: 클래스 이름이 car_BrakeOn일 때만 model_on이 True인지."""
    xyxy = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=float)
    out = to_brake_boxes(xyxy, np.array([0, 1]), np.array([0.8, 0.6]), {0: "car_BrakeOff", 1: "car_BrakeOn"})
    assert [b["model_on"] for b in out] == [False, True]
    assert out[1] == {"box_xyxy_px": (5.0, 5.0, 20.0, 20.0), "model_on": True, "score": 0.6}


def test_box_iou_values():
    """요약: 같은 박스는 1, 안 겹치면 0, 절반 겹치면 1/3."""
    assert box_iou(BOX, BOX) == 1.0
    assert box_iou(BOX, (300, 300, 400, 400)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_match_picks_most_overlapping_and_skips_far_boxes():
    """요약: 앞차와 가장 많이 겹치는 박스를 고르고, 기준보다 덜 겹치면 None."""
    near = {"box_xyxy_px": (105, 100, 205, 200), "model_on": True, "score": 0.9}
    other = {"box_xyxy_px": (150, 100, 250, 200), "model_on": False, "score": 0.9}
    assert match_front_brake([other, near], BOX, CFG["match_iou_min"]) is near
    assert match_front_brake([{"box_xyxy_px": (190, 190, 290, 290), "model_on": True, "score": 0.9}], BOX, 0.3) is None


def test_lamp_signals_sees_third_brake_light():
    """요약: 보조 제동등 줄을 그리면 third_red가 기준 이상, 안 그리면 0인지."""
    assert lamp_signals(_car(third=True), BOX, CFG)["third_red"] >= CFG["third_red_min"]
    assert lamp_signals(_car(third=False), BOX, CFG)["third_red"] == 0.0


def test_tail_lights_only_are_filtered_even_if_model_says_on():
    """요약: 모델은 켜짐이라도 보조 제동등도 없고 밝아지지도 않으면(미등) 꺼짐인지."""
    tr = BrakeStateTracker(CFG)
    for i in range(10):
        out = tr.update(FRONT, _matched(), _car(), i * 0.033)
    assert out["model_on"] is True and out["brake_on"] is False


def test_third_brake_light_confirms_model_on():
    """요약: 모델 켜짐 + 보조 제동등이 보이면 켜짐인지."""
    out = BrakeStateTracker(CFG).update(FRONT, _matched(), _car(third=True), 0.0)
    assert out["third_on"] is True and out["brake_on"] is True


def test_side_lamps_brightening_confirms_model_on():
    """요약: 보조 제동등이 없어도(트럭 등) 좌우 등이 평소보다 확 밝아지면 켜짐인지."""
    tr = BrakeStateTracker(CFG)
    for i in range(10):
        tr.update(FRONT, _matched(), _car(), i * 0.033)  # 평소: 어두운 미등
    for i in range(10):  # 브레이크를 밟아 좌우 등이 밝아진 상태가 다수결 시간 이상 이어짐
        out = tr.update(FRONT, _matched(), _car(side=BGR_RED), 0.35 + i * 0.033)
    assert out["lamp_rose"] is True and out["brake_on"] is True


def test_low_score_on_is_not_trusted():
    """요약: 켜짐 확신도가 on_score_min 아래면 보조 제동등이 보여도 켜짐으로 보지 않는지."""
    out = BrakeStateTracker(CFG).update(FRONT, _matched(score=CFG["on_score_min"] - 0.01), _car(third=True), 0.0)
    assert out["model_on"] is False and out["brake_on"] is False


def test_vote_uses_time_window():
    """요약: 다수결이 vote_window_s초 안의 판단만 쓰는지 (오래된 켜짐 기록은 버림)."""
    tr = BrakeStateTracker(CFG)
    tr.update(FRONT, _matched(), _car(third=True), 0.0)  # 켜짐
    out = tr.update(FRONT, _matched(model_on=False), _car(), 0.05)  # 꺼짐 1 : 켜짐 1 → 과반 아님
    assert out["brake_on"] is False
    tr2 = BrakeStateTracker(CFG)
    tr2.update(FRONT, _matched(), _car(third=True), 0.0)
    tr2.update(FRONT, _matched(), _car(third=True), 0.05)
    out = tr2.update(FRONT, _matched(model_on=False), _car(), 0.1)  # 켜짐 2 : 꺼짐 1
    assert out["brake_on"] is True
    out = tr2.update(FRONT, _matched(model_on=False), _car(), 0.1 + CFG["vote_window_s"] + 0.01)  # 켜짐 기록 만료
    assert out["brake_on"] is False


def test_front_change_resets_baseline():
    """요약: 앞차(track_id)가 바뀌면 이전 차의 평소 밝기와 비교하지 않는지."""
    tr = BrakeStateTracker(CFG)
    for i in range(10):
        tr.update(FRONT, _matched(), _car(), i * 0.033)  # 1번 차: 어두운 미등
    other = {"track_id": 2, "box_xyxy_px": BOX}
    out = tr.update(other, _matched(), _car(side=BGR_RED), 0.35)  # 2번 차는 원래 밝은 등
    assert out["lamp_rose"] is False and out["brake_on"] is False


def test_no_front_or_no_match_is_unknown():
    """요약: 앞차가 없거나 모델이 앞차를 못 찾으면 brake_on이 None(모름)인지."""
    tr = BrakeStateTracker(CFG)
    assert tr.update(None, None, _car(), 0.0)["brake_on"] is None
    assert tr.update(FRONT, None, _car(), 0.1)["brake_on"] is None


def test_load_brake_model_missing_file_stops(tmp_path):
    """요약: 가중치 파일이 없으면 자동 다운로드하지 않고 멈추는지."""
    with pytest.raises(FileNotFoundError):
        load_brake_model(dict(CFG, weights_path=str(tmp_path / "none.pt")))
