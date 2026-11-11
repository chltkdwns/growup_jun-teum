"""LaneTracker 테스트: 촬영 시각(t_s)을 직접 넣어 시간 기준 동작을 확인한다."""
from app.lane.tracker import LaneTracker

CFG_LANE = {
    "smooth_window_s": 0.5, "hold_s": 0.3, "width_window_s": 1.0, "estimate_max_s": 1.0,
    "lane_change_match_ratio": 0.15, "lane_change_cooldown_s": 1.0,
}
NONE = {"left": None, "right": None}


def _left(xb: int, xt: int) -> dict:
    """요약: 왼쪽 차선만 있는 detect_lanes() 형식 결과."""
    return {"left": (xb, 600, xt, 400), "right": None}


def test_average_within_window():
    """요약: window 안의 검출값 평균을 돌려주는지."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_left(100, 300), 0.0)
    out = tr.update(_left(110, 310), 0.1)
    assert out["left"] == (105, 600, 305, 400)
    assert out["right"] is None


def test_hold_keeps_last_value():
    """요약: 놓친 지 hold_s(0.3초) 이내면 직전 값을 유지하는지 (경계 0.3초 포함)."""
    tr = LaneTracker(CFG_LANE)
    # 시작을 0.0으로 둬야 경계가 정확히 0.3이 된다 (1.3 - 1.0 은 부동소수점으로 0.30000000000000004)
    tr.update(_left(100, 300), 0.0)
    assert tr.update(NONE, 0.2)["left"] == (100, 600, 300, 400)
    assert tr.update(NONE, 0.3)["left"] == (100, 600, 300, 400)


def test_none_after_hold():
    """요약: hold_s를 넘게 못 찾으면 None인지."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_left(100, 300), 1.0)
    assert tr.update(NONE, 1.31)["left"] is None


def test_old_value_drops_out_of_window():
    """요약: window(0.5초)보다 오래된 값은 평균에서 빠지는지."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_left(100, 300), 0.0)
    out = tr.update(_left(200, 400), 0.6)
    assert out["left"] == (200, 600, 400, 400)


def _both(xl: int, xr: int) -> dict:
    """요약: 좌우가 다 보이는 결과. 위쪽 x는 아래쪽에서 각각 +100, -100 (위 폭 = 아래 폭 - 200)."""
    return {"left": (xl, 600, xl + 100, 400), "right": (xr, 600, xr - 100, 400)}


def test_both_lanes_not_estimated():
    """요약: 좌우 다 보이면 추정하지 않음."""
    out = LaneTracker(CFG_LANE).update(_both(100, 500), 0.0)
    assert out["estimated"] == ()


def test_missing_right_estimated_from_width():
    """요약: 오른쪽만 놓치면 왼쪽 + 기억한 폭(아래 400, 위 200)으로 추정. hold보다 추정이 우선."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_both(100, 500), 0.0)
    out = tr.update({"left": (150, 600, 250, 400), "right": None}, 0.1)  # hold 시간 안이지만 추정 사용
    assert out["estimated"] == ("right",)
    left = out["left"]  # 평균 낸 왼쪽 (125, 225)
    assert out["right"] == (left[0] + 400, 600, left[2] + 200, 400)


def test_missing_left_estimated_symmetric():
    """요약: 왼쪽만 놓치면 오른쪽 - 폭으로 추정."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_both(100, 500), 0.0)
    out = tr.update({"left": None, "right": (500, 600, 400, 400)}, 0.1)
    assert out["estimated"] == ("left",)
    assert out["left"] == (100, 600, 200, 400)


def test_no_estimate_after_estimate_max():
    """요약: 두 선을 함께 본 지 estimate_max_s(1초)가 넘으면 추정하지 않고, hold도 지나 None."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_both(100, 500), 0.0)
    out = tr.update({"left": (100, 600, 200, 400), "right": None}, 1.5)
    assert out["estimated"] == () and out["right"] is None


def test_no_estimate_without_width():
    """요약: 좌우를 함께 본 적이 없으면 폭을 몰라 추정하지 않음."""
    out = LaneTracker(CFG_LANE).update({"left": (100, 600, 200, 400), "right": None}, 0.0)
    assert out["estimated"] == () and out["right"] is None


def test_width_uses_median():
    """요약: 폭 기록 중 튀는 값 하나가 있어도 중앙값(400)을 쓴다."""
    tr = LaneTracker(CFG_LANE)
    for t, xr in ((0.0, 500), (0.1, 500), (0.2, 900)):
        tr.update({"left": (100, 600, 200, 400), "right": (xr, 600, xr - 100, 400)}, t)
    out = tr.update({"left": (100, 600, 200, 400), "right": None}, 0.3)
    assert out["right"][0] == out["left"][0] + 400


def _crossed_right(tr: LaneTracker) -> dict:
    """요약: 좌 100 / 우 500 (폭 400)을 본 뒤, 0.1초에 오른쪽 선이 x=495로 왼쪽에 넘어온 상황."""
    tr.update(_both(100, 500), 0.0)
    return tr.update({"left": (495, 600, 395, 400), "right": None}, 0.1)


def test_lane_change_right_detected_and_reset():
    """요약: 오른쪽 선이 왼쪽으로 넘어오면 "right" 감지, 왼쪽은 예전 선(100)과 섞이지 않고 495 그대로."""
    out = _crossed_right(LaneTracker(CFG_LANE))
    assert out["lane_change"] == "right"
    assert out["left"] == (495, 600, 395, 400)


def test_offset_after_change_near_minus_one():
    """요약: 차로 변경 직후 내 차(정면 x=500)는 새 차로의 왼쪽 선 바로 옆이라 offset이 약 -1."""
    from app.lane.offset import lane_offset

    out = _crossed_right(LaneTracker(CFG_LANE))
    ratio = lane_offset(out, 1000, {"center_x_ratio": 0.5})["lane_offset_ratio"]
    assert -1.05 < ratio < -0.9


def test_width_kept_after_change():
    """요약: 감지 뒤에도 차선 폭(400) 기록은 남아 반대쪽을 바로 추정한다."""
    out = _crossed_right(LaneTracker(CFG_LANE))
    assert out["estimated"] == ("right",)
    assert out["right"][0] == 495 + 400


def test_lane_change_left_symmetric():
    """요약: 왼쪽 선이 오른쪽으로 넘어오면 "left" 감지."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_both(100, 500), 0.0)
    out = tr.update({"left": None, "right": (105, 600, 205, 400)}, 0.1)
    assert out["lane_change"] == "left"
    assert out["right"] == (105, 600, 205, 400)


def test_small_jitter_not_lane_change():
    """요약: 선이 조금 흔들리는 것(10px)은 차로 변경이 아님."""
    tr = LaneTracker(CFG_LANE)
    tr.update(_both(100, 500), 0.0)
    assert tr.update(_both(110, 505), 0.1)["lane_change"] is None


def test_no_repeat_during_cooldown():
    """요약: 감지 뒤 쿨다운(1초) 동안은 비슷한 상황이 와도 다시 감지하지 않음."""
    tr = LaneTracker(CFG_LANE)
    _crossed_right(tr)
    out = tr.update({"left": None, "right": (498, 600, 398, 400)}, 0.2)
    assert out["lane_change"] is None

