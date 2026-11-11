"""lane_offset 테스트: 차선 위치를 직접 넣어 경계값(정중앙, 선 위, 한쪽 없음)을 확인한다."""
import pytest

from app.lane.offset import lane_offset

W = 1000
CAM = {"center_x_ratio": 0.5}  # 정면 x = 500


def _lanes(xl: int | None, xr: int | None) -> dict:
    """요약: 아래쪽 x만 의미 있는 가짜 차선 결과."""
    return {
        "left": None if xl is None else (xl, 700, 450, 400),
        "right": None if xr is None else (xr, 700, 550, 400),
    }


def test_center_is_zero():
    """요약: 차선 정중앙이면 0."""
    out = lane_offset(_lanes(300, 700), W, CAM)
    assert out == {"lane_offset_ratio": 0.0, "lane_status": "ok"}


def test_on_right_line_is_plus_one():
    """요약: 정면이 오른쪽 선 위면 +1, 왼쪽 선 위면 -1."""
    assert lane_offset(_lanes(0, 500), W, CAM)["lane_offset_ratio"] == pytest.approx(1.0)
    assert lane_offset(_lanes(500, 1000), W, CAM)["lane_offset_ratio"] == pytest.approx(-1.0)


def test_half_way_to_right():
    """요약: 중앙과 오른쪽 선 사이 절반이면 +0.5."""
    assert lane_offset(_lanes(200, 600), W, CAM)["lane_offset_ratio"] == pytest.approx(0.5)


def test_missing_lane_is_unavailable():
    """요약: 한쪽 차선이 없으면 None + unavailable."""
    for lanes in (_lanes(300, None), _lanes(None, 700), _lanes(None, None)):
        assert lane_offset(lanes, W, CAM) == {"lane_offset_ratio": None, "lane_status": "unavailable"}


def test_crossed_lanes_is_unavailable():
    """요약: 왼쪽 선이 오른쪽 선보다 오른쪽이면(잘못 찾음) unavailable."""
    assert lane_offset(_lanes(700, 300), W, CAM)["lane_status"] == "unavailable"
