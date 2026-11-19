"""거리·TTC 테스트: 일정한 속도로 다가오는 가짜 앞차 박스를 만들어 거리와 TTC를 확인한다 (모델·영상 없이 실행)."""
from pathlib import Path

import pytest
import yaml

from app.risk.lead import LeadRiskTracker, estimate_distance_m, fit_ttc_s, focal_px

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))["risk"]
IMG_W = 1280
FPS = 30
CAR_W_M = 1.8


def _front(distance_m: float, track_id: int = 1, cls_name: str = "car") -> dict:
    """요약: 그 거리에 있는 차가 화면에 찍힐 박스 폭(핀홀 공식 거꾸로)으로 가짜 앞차를 만든다."""
    w_px = focal_px(IMG_W, CFG["hfov_deg"]) * CFG["real_width_m"][cls_name] / distance_m
    x1 = IMG_W / 2 - w_px / 2
    return {"track_id": track_id, "cls_name": cls_name, "box_xyxy_px": (x1, 300.0, x1 + w_px, 400.0)}


def _approach(tr: LeadRiskTracker, start_m: float, speed_mps: float, duration_s: float, t0_s: float = 0.0) -> dict:
    """요약: start_m에서 speed_mps로 다가오는 앞차를 duration_s초 동안 FPS로 넣고 마지막 결과를 돌려준다."""
    out = None
    n = round(duration_s * FPS) + 1
    for i in range(n):
        t = i / FPS
        out = tr.update(_front(start_m - speed_mps * t), IMG_W, t0_s + t)
    return out


def test_focal_px_from_fov():
    """요약: 화각 90도면 초점거리 = 화면 절반 폭 (tan 45° = 1)."""
    assert focal_px(1000, 90) == pytest.approx(500)


def test_distance_pinhole():
    """요약: 초점거리 1000px, 승용차 1.8m, 박스 100px → 18m."""
    assert estimate_distance_m(100, 1000, 1.8) == pytest.approx(18.0)


def test_distance_uses_class_width():
    """요약: 같은 거리의 이륜차(0.8m)도 클래스별 폭을 써서 같은 거리로 나온다."""
    out = LeadRiskTracker(CFG).update(_front(12.0, cls_name="motorcycle"), IMG_W, 0.0)
    assert out["distance_m"] == pytest.approx(12.0)


def test_ttc_exactly_3s_and_1_5s():
    """요약: 5m/s로 다가와 지금 15m면 TTC 3.0초, 7.5m면 1.5초 (경고·위험 경계값)."""
    out = _approach(LeadRiskTracker(CFG), start_m=15.0 + 5.0 * 0.7, speed_mps=5.0, duration_s=0.7)
    assert out["distance_m"] == pytest.approx(15.0)
    assert out["ttc_s"] == pytest.approx(3.0, abs=1e-6)
    out = _approach(LeadRiskTracker(CFG), start_m=7.5 + 5.0 * 0.7, speed_mps=5.0, duration_s=0.7)
    assert out["ttc_s"] == pytest.approx(1.5, abs=1e-6)


def test_fit_ttc_from_line():
    """요약: 1/폭이 1초에 0.1씩 줄고 지금 0.3이면 0이 될 때까지 3초."""
    samples = [(0.0, 0.5), (1.0, 0.4), (2.0, 0.3)]
    assert fit_ttc_s(samples, 2.0) == pytest.approx(3.0)


def test_receding_or_same_distance_has_no_ttc():
    """요약: 멀어지거나 거리가 그대로면 TTC는 None (거리는 있음)."""
    out = _approach(LeadRiskTracker(CFG), start_m=20.0, speed_mps=-3.0, duration_s=0.7)
    assert out["ttc_s"] is None and out["distance_m"] == pytest.approx(22.1)
    out = _approach(LeadRiskTracker(CFG), start_m=20.0, speed_mps=0.0, duration_s=0.7)
    assert out["ttc_s"] is None


def test_too_slow_approach_over_max_is_none():
    """요약: TTC가 ttc_max_s보다 크면 None (아주 천천히 다가옴 = 위험 아님)."""
    out = _approach(LeadRiskTracker(CFG), start_m=30.0, speed_mps=1.0, duration_s=0.7)
    assert out["ttc_s"] is None  # 약 29.3m ÷ 1m/s ≈ 29초 > 10초


def test_too_few_samples_no_ttc_yet():
    """요약: 기록이 ttc_min_samples개보다 적으면 TTC None, 거리는 바로 나온다."""
    tr = LeadRiskTracker(CFG)
    for i in range(CFG["ttc_min_samples"] - 1):
        out = tr.update(_front(20.0 - 5.0 * i / FPS), IMG_W, i / FPS)
    assert out["ttc_s"] is None and out["distance_m"] is not None


def test_front_change_resets_history():
    """요약: 앞차(track_id)가 바뀌면 이전 차 기록을 버린다 (바뀐 직후 TTC None)."""
    tr = LeadRiskTracker(CFG)
    _approach(tr, start_m=20.0, speed_mps=5.0, duration_s=0.7)
    out = tr.update(_front(10.0, track_id=2), IMG_W, 0.75)
    assert out["ttc_s"] is None and out["distance_m"] == pytest.approx(10.0)


def test_no_front_gives_none():
    """요약: 앞차가 없으면 거리·TTC 모두 None, 다시 나오면 기록을 처음부터 쌓는다."""
    tr = LeadRiskTracker(CFG)
    _approach(tr, start_m=20.0, speed_mps=5.0, duration_s=0.7)
    assert tr.update(None, IMG_W, 0.8) == {"distance_m": None, "ttc_s": None}
    assert tr.update(_front(15.0), IMG_W, 0.9)["ttc_s"] is None


def test_old_samples_dropped_by_time():
    """요약: ttc_window_s보다 오래된 기록은 버려서, 최근 움직임(멈춤)만 보고 TTC None이 된다."""
    tr = LeadRiskTracker(CFG)
    _approach(tr, start_m=20.0, speed_mps=5.0, duration_s=0.7)  # 다가오다가
    out = None
    for i in range(1, int(CFG["ttc_window_s"] * FPS) + 3):  # 같은 거리로 window보다 오래 멈춤
        out = tr.update(_front(16.5), IMG_W, 0.7 + i / FPS)
    assert out["ttc_s"] is None


def test_output_keys_match_schema():
    """요약: 출력 키가 schemas.LeadRisk(11.17 합의)와 같다."""
    from app.schemas import LeadRisk

    assert set(LeadRiskTracker(CFG).update(None, IMG_W, 0.0)) == set(LeadRisk.__annotations__)
