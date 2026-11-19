"""PipelineWarningSource 통합 테스트: 가짜 파이프라인(박스가 커지는 앞차)을 넣어 진짜 경고가 기록 형식으로 이어지는지 확인한다 (모델·DB 없이 실행)."""
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from app.nearmiss import to_event_in
from app.risk.lead import focal_px
from app.risk.warning_source import PipelineWarningSource

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))["risk"]
IMG_W, IMG_H, FPS = 1280, 720, 30
FRAME = np.zeros((IMG_H, IMG_W, 3), np.uint8)
NO_SPEED_KMH = 0.0  # 안전거리 0m라 안전거리 주의가 안 나온다 (TTC·경고만 보려는 테스트용)
NO_OFFSET = {"lane_offset_ratio": None, "lane_status": "unavailable"}


class FakePipeline:
    """요약: 정해 둔 시각의 거리로 앞차 박스를 만들어 PerceptionPipeline.process() 모양의 결과를 돌려준다."""

    def __init__(self, distance_at, brake_on=None, offset=None):
        self.distance_at = distance_at  # t_s → 앞차 거리(m), None이면 앞차 없음
        self.brake_on = brake_on
        self.offset = offset or NO_OFFSET

    def process(self, frame, t_s):
        d = self.distance_at(t_s)
        front = None
        if d is not None:
            w = focal_px(IMG_W, CFG["hfov_deg"]) * CFG["real_width_m"]["car"] / d
            front = {"track_id": 1, "cls_name": "car", "box_xyxy_px": (600.0, 300.0, 600.0 + w, 400.0)}
        brake = None if self.brake_on is None else {"brake_on": self.brake_on}
        return {"front": front, "brake": brake, "cut_in": None, "offset": self.offset, "lane_change": None}


def _run(source, seconds):
    """요약: source를 FPS로 seconds초 돌리고 (t_s, 경고 목록) 목록을 돌려준다."""
    return [(i / FPS, source(FRAME, i / FPS)) for i in range(int(seconds * FPS) + 1)]


def test_closing_lead_car_becomes_warning_then_danger():
    """요약: 15m 앞차가 6m/s로 다가오면 TTC 3초 근처부터 경고, 1.5초 근처부터 위험이 나온다."""
    src = PipelineWarningSource(FakePipeline(lambda t: 20.0 - 6.0 * t), CFG, speed_kmh=NO_SPEED_KMH)
    levels = {t: [w.level for w in ws] for t, ws in _run(src, 2.0)}
    assert levels[0.0] == []  # 기록이 모자란 처음 프레임들
    first_warning = next(t for t, lv in levels.items() if "warning" in lv)
    first_danger = next(t for t, lv in levels.items() if "danger" in lv)
    assert 0.3 <= first_warning <= 0.5  # 진짜 TTC 3.0초는 t=0.333 (거리 18m)
    assert 1.8 <= first_danger <= 1.95  # 진짜 TTC 1.5초는 t=1.833 (거리 9m)


def test_far_steady_car_gives_no_warning():
    """요약: 거리가 일정한 앞차는 경고가 없다."""
    src = PipelineWarningSource(FakePipeline(lambda t: 25.0), CFG, speed_kmh=NO_SPEED_KMH)
    assert all(ws == [] for _, ws in _run(src, 1.0))


def test_no_front_car_gives_no_warning():
    """요약: 앞차가 없으면 경고가 없다."""
    src = PipelineWarningSource(FakePipeline(lambda t: None), CFG)
    assert all(ws == [] for _, ws in _run(src, 0.5))


def test_warning_converts_to_event_for_recording():
    """요약: 경고가 기록 쪽 to_event_in()을 그대로 통과한다 (이름·단위가 맞음)."""
    src = PipelineWarningSource(FakePipeline(lambda t: 20.0 - 6.0 * t), CFG)
    w = next(w for _, ws in _run(src, 2.0) for w in ws if w.level in ("warning", "danger"))
    ev = to_event_in(w, datetime(2026, 11, 19, tzinfo=UTC))
    assert ev.type == "lead" and ev.level in ("warning", "danger")
    assert ev.min_ttc_s is not None and ev.distance_m is not None and ev.target_class == "car"


def test_speed_enables_headway_caution():
    """요약: 내 속도 72km/h(안전거리 40m)면 30m 앞차는 주의, 90m 앞차는 안전."""
    assert [w.level for w in PipelineWarningSource(FakePipeline(lambda t: 30.0), CFG, speed_kmh=72.0)(FRAME, 0.0)] == ["caution"]
    assert PipelineWarningSource(FakePipeline(lambda t: 90.0), CFG, speed_kmh=72.0)(FRAME, 0.0) == []


def test_default_speed_used_when_not_given():
    """요약: 속도를 안 주면 default_speed_kmh(100km/h, 안전거리 약 55.6m)를 쓴다. 50m는 주의, 60m는 안전."""
    assert CFG["default_speed_kmh"] == 100
    assert [w.level for w in PipelineWarningSource(FakePipeline(lambda t: 50.0), CFG)(FRAME, 0.0)] == ["caution"]
    assert PipelineWarningSource(FakePipeline(lambda t: 60.0), CFG)(FRAME, 0.0) == []


def test_speed_change_applies_from_next_frame():
    """요약: set_speed가 오면 서버가 speed_kmh에 대입하고, 다음 프레임부터 새 속도로 판단한다."""
    src = PipelineWarningSource(FakePipeline(lambda t: 30.0), CFG, speed_kmh=36.0)  # 안전거리 20m
    assert src(FRAME, 0.0) == []
    src.speed_kmh = 72.0  # 안전거리 40m
    assert [w.level for w in src(FRAME, 1.0)] == ["caution"]


def test_lane_departure_comes_with_lead_warning():
    """요약: 차선 이탈은 앞차 경고와 함께 별도 경고로 나온다."""
    off = {"lane_offset_ratio": 1.2, "lane_status": "ok"}
    src = PipelineWarningSource(FakePipeline(lambda t: None, offset=off), CFG)
    assert [w.type for w in src(FRAME, 0.0)] == ["lane_departure"]
