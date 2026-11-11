"""LaneTracker 테스트: 촬영 시각(t_s)을 직접 넣어 시간 기준 동작을 확인한다."""
from app.lane.tracker import LaneTracker

CFG_LANE = {"smooth_window_s": 0.5, "hold_s": 0.3}
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
