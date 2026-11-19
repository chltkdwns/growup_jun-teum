"""경고 단계 판단 테스트: 경계값(정확히 3.0초·1.5초), 단계 상향, 내림 대기, 차선 이탈을 값만 넣어 확인한다 (모델·영상 없이 실행)."""
from pathlib import Path

import pytest
import yaml

from app.risk.warning import WarningJudge, level_from_ttc, raw_level

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))["risk"]
NO_OFFSET = {"lane_offset_ratio": None, "lane_status": "unavailable"}
SAFE, CAUTION, WARNING, DANGER = 0, 1, 2, 3  # LEVELS의 순서


def _judge_once(ttc_s=None, distance_m=30.0, brake_on=None, cut_in=None, speed_kmh=None, offset=None, lane_change=None, t_s=0.0, judge=None):
    """요약: 값 하나로 판단기를 한 번 돌려 경고 목록을 돌려준다 (judge를 넘기면 이어서 쓴다)."""
    judge = judge or WarningJudge(CFG)
    lead = {"distance_m": distance_m, "ttc_s": ttc_s}
    brake = None if brake_on is None else {"brake_on": brake_on}
    return judge.update(lead, brake, cut_in, offset or NO_OFFSET, lane_change, speed_kmh, t_s, "car")


# ---------- TTC 경계값 ----------
@pytest.mark.parametrize(
    "ttc_s, expected",
    [
        (None, SAFE),
        (3.01, SAFE),
        (3.0, WARNING),  # 정확히 3초는 경고 (이하)
        (2.99, WARNING),
        (1.51, WARNING),
        (1.5, DANGER),  # 정확히 1.5초는 위험 (이하)
        (1.49, DANGER),
        (0.1, DANGER),
    ],
)
def test_level_from_ttc_boundaries(ttc_s, expected):
    """요약: TTC 3.0초 이하 경고, 1.5초 이하 위험 (경계값 포함)."""
    assert level_from_ttc(ttc_s, CFG) == expected


# ---------- 안전거리 ----------
def test_headway_short_is_caution_and_boundary():
    """요약: 속도 72km/h(20m/s)면 안전거리 40m. 39.9m는 주의, 정확히 40m는 안전."""
    assert raw_level(None, 39.9, 72.0, None, False, CFG)[0] == CAUTION
    assert raw_level(None, 40.0, 72.0, None, False, CFG)[0] == SAFE


def test_headway_skipped_without_speed():
    """요약: 내 속도를 모르면 안전거리 판단을 건너뛴다 (SPEC)."""
    assert raw_level(None, 1.0, None, None, False, CFG)[0] == SAFE


# ---------- 단계 상향 ----------
def test_brake_on_while_closing_raises_one_level():
    """요약: 브레이크등 켜짐 + 가까워지는 중(TTC 있음)이면 한 단계 위. 안전→주의, 경고→위험."""
    assert raw_level(8.0, 30.0, None, True, False, CFG) == (CAUTION, "brake")
    assert raw_level(2.0, 30.0, None, True, False, CFG) == (DANGER, "brake")


def test_brake_on_without_closing_does_not_raise():
    """요약: 브레이크등이 켜져도 가까워지지 않으면(TTC 없음) 올리지 않는다."""
    assert raw_level(None, 30.0, None, True, False, CFG) == (SAFE, "lead")


def test_brake_off_or_unknown_does_not_raise():
    """요약: 브레이크등 꺼짐·모름(None)은 올리지 않는다."""
    assert raw_level(2.0, 30.0, None, False, False, CFG) == (WARNING, "lead")
    assert raw_level(2.0, 30.0, None, None, False, CFG) == (WARNING, "lead")


def test_cut_in_raises_one_level_and_caps_at_danger():
    """요약: 끼어들기가 확정되면 한 단계 위, 위험보다 높아지지 않는다."""
    assert raw_level(None, 30.0, None, None, True, CFG) == (CAUTION, "cut_in")
    assert raw_level(1.0, 30.0, None, None, True, CFG) == (DANGER, "cut_in")


def test_cut_in_and_brake_stack():
    """요약: 끼어들기와 브레이크등이 같이 있으면 두 단계 올라간다 (TTC 3초 초과: 안전 → 경고)."""
    assert raw_level(8.0, 30.0, None, True, True, CFG) == (WARNING, "cut_in")


# ---------- 내림 대기 (촬영 시각 기준) ----------
def test_level_goes_up_immediately():
    """요약: 단계를 올릴 때는 기다리지 않는다."""
    j = WarningJudge(CFG)
    assert _judge_once(ttc_s=None, judge=j, t_s=0.0) == []
    out = _judge_once(ttc_s=1.0, judge=j, t_s=0.033)
    assert [w.level for w in out] == ["danger"]


def test_level_down_waits_hold_time_exactly():
    """요약: 위험 → 안전으로 내릴 때 level_down_hold_s(0.5초)가 지나야 내린다. 0.49초까지는 유지, 0.5초에 내림."""
    j = WarningJudge(CFG)
    _judge_once(ttc_s=1.0, judge=j, t_s=0.0)
    assert [w.level for w in _judge_once(ttc_s=None, judge=j, t_s=0.25)] == ["danger"]  # 낮은 상태 시작
    assert [w.level for w in _judge_once(ttc_s=None, judge=j, t_s=0.74)] == ["danger"]  # 0.49초 지남
    assert _judge_once(ttc_s=None, judge=j, t_s=0.75) == []  # 정확히 0.5초 지남 → 안전


def test_down_timer_resets_when_risk_returns():
    """요약: 낮아졌다가 다시 높아지면 내림 대기 시간을 처음부터 센다."""
    j = WarningJudge(CFG)
    _judge_once(ttc_s=1.0, judge=j, t_s=0.0)
    _judge_once(ttc_s=None, judge=j, t_s=0.25)
    _judge_once(ttc_s=1.0, judge=j, t_s=0.5)  # 다시 위험
    assert [w.level for w in _judge_once(ttc_s=None, judge=j, t_s=0.75)] == ["danger"]  # 여기서 다시 세기 시작
    assert [w.level for w in _judge_once(ttc_s=None, judge=j, t_s=1.0)] == ["danger"]  # 0.25초만 지남
    assert _judge_once(ttc_s=None, judge=j, t_s=1.25) == []  # 정확히 0.5초 지남


def test_cut_in_one_frame_is_held_by_hold_time():
    """요약: 끼어들기는 확정된 한 프레임에만 오지만, 내림 대기 덕분에 그 단계가 hold 시간 동안 유지된다."""
    j = WarningJudge(CFG)
    out = _judge_once(cut_in={"track_id": 3, "side": "left", "t_s": 1.0}, judge=j, t_s=1.0)
    assert [(w.type, w.level) for w in out] == [("cut_in", "caution")]
    assert [w.type for w in _judge_once(judge=j, t_s=1.25)] == ["cut_in"]  # 낮은 상태 시작
    assert [w.type for w in _judge_once(judge=j, t_s=1.5)] == ["cut_in"]  # 0.25초 지남
    assert _judge_once(judge=j, t_s=1.75) == []  # 정확히 0.5초 지남


# ---------- 출력 내용 ----------
def test_warning_carries_ttc_distance_and_class():
    """요약: 경고에 촬영 시각·TTC·거리·대상 종류가 들어간다 (기록 쪽이 그대로 저장)."""
    out = _judge_once(ttc_s=2.0, distance_m=12.5, t_s=4.0)
    w = out[0]
    assert (w.t_s, w.type, w.level, w.ttc_s, w.distance_m, w.target_class) == (4.0, "lead", "warning", 2.0, 12.5, "car")


def test_safe_gives_no_warning():
    """요약: 안전이면 경고 목록이 비어 있다."""
    assert _judge_once(ttc_s=None) == []


# ---------- 차선 이탈 ----------
def _offset(ratio):
    return {"lane_offset_ratio": ratio, "lane_status": "ok"}


def test_lane_departure_boundary():
    """요약: |치우침| 0.99는 정상, 정확히 1.0(선 위)은 이탈. 왼쪽(-1.0)도 같다."""
    assert _judge_once(offset=_offset(0.99)) == []
    out = _judge_once(offset=_offset(1.0))
    assert [(w.type, w.level) for w in out] == [("lane_departure", CFG["lane_departure_level"])]
    assert [w.type for w in _judge_once(offset=_offset(-1.0))] == ["lane_departure"]


def test_lane_departure_unknown_offset_no_warning():
    """요약: 차선을 모르면(None) 이탈 경고를 내지 않는다."""
    assert _judge_once(offset=NO_OFFSET) == []


def test_lane_departure_skipped_after_lane_change():
    """요약: 차로 변경을 감지한 뒤 1.0초 동안은 이탈 판단 안 함. 정확히 1.0초 지난 시각부터 다시 판단."""
    j = WarningJudge(CFG)
    assert _judge_once(offset=_offset(1.2), lane_change="left", judge=j, t_s=5.0) == []
    assert _judge_once(offset=_offset(1.2), judge=j, t_s=5.99) == []
    assert [w.type for w in _judge_once(offset=_offset(1.2), judge=j, t_s=6.0)] == ["lane_departure"]


def test_lane_departure_and_lead_warning_together():
    """요약: 앞차 경고와 차선 이탈은 따로 나온다 (type이 다르므로 기록에서 따로 합쳐짐)."""
    out = _judge_once(ttc_s=2.0, offset=_offset(1.1))
    assert sorted(w.type for w in out) == ["lane_departure", "lead"]
