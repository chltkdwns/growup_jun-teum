"""앞차 정하기 테스트: 차선과 박스를 직접 넣어 "내 차선 안에서 가장 가까운 차"를 고르는지 확인한다."""
from pathlib import Path

import yaml

from app.lane.front_car import (
    FrontCarTracker,
    default_lanes,
    select_front_car,
    x_on_line,
)

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))
W, H = 1280, 720
# 아래(y=612)에서 x 300~980, 위(y=432)에서 x 600~680 으로 모이는 내 차선
LANES = {"left": (300, 612, 600, 432), "right": (980, 612, 680, 432), "estimated": (), "lane_change": None}


def _det(x1: float, y1: float, x2: float, y2: float, cls_name: str = "car", track_id: int = 1) -> dict:
    """요약: detect_vehicles() 형식의 가짜 탐지 하나."""
    return {"track_id": track_id, "cls_name": cls_name, "score": 0.9, "box_xyxy_px": (x1, y1, x2, y2), "t_s": 0.0}


def _select(dets: list, lanes: dict = LANES):
    """요약: 기본 config로 select_front_car를 부른다."""
    return select_front_car(dets, lanes, W, H, CFG["camera"], CFG["lane"], CFG["front_car"])


def test_x_on_line_extends():
    """요약: 직선 위 x를 높이로 구하고, 범위 밖 높이도 연장해서 구하는지."""
    assert x_on_line(LANES["left"], 612) == 300
    assert x_on_line(LANES["left"], 522) == 450
    assert x_on_line(LANES["left"], 702) == 150


def test_picks_car_in_my_lane_over_closer_car_in_next_lane():
    """요약: 옆 차로 차가 더 가까워도(아래에 있어도) 내 차선 안의 차를 고르는지."""
    mine = _det(560, 480, 720, 560, track_id=1)
    next_lane = _det(100, 500, 260, 600, track_id=2)  # 박스 아래 가운데 x=180 < 왼쪽 차선(y=600에서 약 320)
    assert _select([next_lane, mine])["track_id"] == 1


def test_picks_nearest_of_two_in_lane():
    """요약: 내 차선에 두 대면 박스 아래가 더 아래인(가까운) 차를 고르는지."""
    far = _det(610, 440, 670, 470, track_id=1)
    near = _det(540, 500, 740, 590, track_id=2)
    assert _select([far, near])["track_id"] == 2


def test_motorcycle_can_be_front():
    """요약: 이륜차도 앞차가 될 수 있는지 (SPEC: 이륜차 포함)."""
    moto = _det(620, 480, 660, 560, cls_name="motorcycle", track_id=7)
    assert _select([moto])["cls_name"] == "motorcycle"


def test_far_car_uses_lane_width_at_its_height():
    """요약: 멀리 있는 차는 그 높이의 좁은 차선 폭으로 판단하는지 (y=450에서 차선은 x 570~710)."""
    inside = _det(620, 420, 660, 450, track_id=1)
    outside = _det(720, 420, 760, 450, track_id=2)  # 가운데 x=740 > 오른쪽 차선 710
    assert _select([outside]) is None
    assert _select([inside, outside])["track_id"] == 1


def test_none_when_no_car_in_lane():
    """요약: 내 차선에 차가 없으면 None."""
    assert _select([_det(0, 500, 150, 600)]) is None
    assert _select([]) is None


def test_default_lane_used_when_lane_missing():
    """요약: 차선을 못 찾으면 기본 차선 영역으로 고르는지 (정면 근처 차는 선택, 화면 끝 차는 제외)."""
    no_lanes = {"left": None, "right": None, "estimated": (), "lane_change": None}
    center = _det(560, 480, 720, 560, track_id=1)
    edge = _det(1100, 480, 1270, 600, track_id=2)
    front = _select([edge, center], no_lanes)
    assert front["track_id"] == 1 and front["lane_source"] == "default"
    assert _select([center])["lane_source"] == "lane"
    dl = default_lanes(W, H, CFG["camera"], CFG["lane"], CFG["front_car"])
    assert dl["left"][0] < W * CFG["camera"]["center_x_ratio"] < dl["right"][0]


TRACK_CFG = {"switch_hold_s": 0.3, "lost_hold_s": 0.3}


def test_tracker_adopts_first_candidate():
    """요약: 앞차가 없을 때는 후보를 바로 쓰고 front_changed=True."""
    a = _det(560, 480, 720, 560, track_id=1)
    out = FrontCarTracker(TRACK_CFG).update(dict(a, lane_source="lane"), [a], 0.0)
    assert out["track_id"] == 1 and out["front_changed"] is True


def test_tracker_ignores_short_flicker():
    """요약: 다른 차가 0.3초 미만으로만 1등이면 앞차를 유지."""
    a, b = _det(560, 480, 720, 560, track_id=1), _det(500, 500, 700, 580, track_id=2)
    tr = FrontCarTracker(TRACK_CFG)
    tr.update(dict(a, lane_source="lane"), [a, b], 0.0)
    assert tr.update(dict(b, lane_source="lane"), [a, b], 0.1)["track_id"] == 1
    assert tr.update(dict(b, lane_source="lane"), [a, b], 0.2)["track_id"] == 1
    out = tr.update(dict(a, lane_source="lane"), [a, b], 0.3)  # 다시 a가 1등 → 대기 기록도 초기화
    assert out["track_id"] == 1 and out["front_changed"] is False


def test_tracker_switches_after_hold():
    """요약: 다른 차가 0.3초 이상 계속 1등이면 교체 (경계 0.3초 포함)."""
    a, b = _det(560, 480, 720, 560, track_id=1), _det(500, 500, 700, 580, track_id=2)
    tr = FrontCarTracker(TRACK_CFG)
    tr.update(dict(a, lane_source="lane"), [a, b], 0.0)
    tr.update(dict(b, lane_source="lane"), [a, b], 0.1)
    out = tr.update(dict(b, lane_source="lane"), [a, b], 0.4)
    assert out["track_id"] == 2 and out["front_changed"] is True


def test_tracker_same_car_box_refreshed():
    """요약: 같은 차가 계속 1등이면 박스를 최신 값으로 바꾼다."""
    a1, a2 = _det(560, 480, 720, 560, track_id=1), _det(550, 490, 730, 570, track_id=1)
    tr = FrontCarTracker(TRACK_CFG)
    tr.update(dict(a1, lane_source="lane"), [a1], 0.0)
    assert tr.update(dict(a2, lane_source="lane"), [a2], 0.1)["box_xyxy_px"] == a2["box_xyxy_px"]


def test_tracker_keeps_box_while_lost_then_drops():
    """요약: 앞차가 탐지에서 사라지면 0.3초까지 마지막 박스로 유지하고, 넘으면 후보(없으면 None)로."""
    a = _det(560, 480, 720, 560, track_id=1)
    tr = FrontCarTracker(TRACK_CFG)
    tr.update(dict(a, lane_source="lane"), [a], 0.0)
    assert tr.update(None, [], 0.2)["track_id"] == 1
    assert tr.update(None, [], 0.31) is None


def test_tracker_keeps_out_of_lane_car_briefly():
    """요약: 앞차가 보이지만 내 차선 밖으로 나가면(후보 None) 0.3초 동안은 유지 후 None."""
    a = _det(560, 480, 720, 560, track_id=1)
    tr = FrontCarTracker(TRACK_CFG)
    tr.update(dict(a, lane_source="lane"), [a], 0.0)
    assert tr.update(None, [a], 0.1)["track_id"] == 1
    assert tr.update(None, [a], 0.4) is None


def test_tracker_switches_immediately_on_lane_change():
    """요약: 차로 변경을 감지한 프레임에는 0.3초를 기다리지 않고 새 차선의 앞차로 바로 바꾼다."""
    a, b = _det(560, 480, 720, 560, track_id=1), _det(600, 450, 680, 500, track_id=2)
    tr = FrontCarTracker(TRACK_CFG)
    tr.update(dict(a, lane_source="lane"), [a, b], 0.0)
    out = tr.update(dict(b, lane_source="lane"), [a, b], 0.1, lane_change="right")
    assert out["track_id"] == 2 and out["front_changed"] is True
