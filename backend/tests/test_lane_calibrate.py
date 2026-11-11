"""calibrate 테스트: 소실점 계산, 중앙값 추정, 가짜 도로 이미지로 보정값을 확인한다."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from app.lane.calibrate import (
    estimate_calibration,
    lane_bottom_center_x,
    vanishing_point,
)
from app.lane.detect import detect_lanes

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))
CALIB = {"min_points": 3, "horizon_margin_ratio": 0.03}
W, H = 1280, 720


def test_vanishing_point_is_intersection():
    """요약: (300,700)→(600,430) 과 (980,700)→(680,430) 의 교점은 x=640, y≈394."""
    lanes = {"left": (300, 700, 600, 430), "right": (980, 700, 680, 430)}
    x, y = vanishing_point(lanes)
    assert x == pytest.approx(640)
    assert y == pytest.approx(700 - 270 * 340 / 300)


def test_parallel_is_none():
    """요약: 두 직선이 평행하면 만나지 않으므로 None."""
    lanes = {"left": (300, 700, 400, 400), "right": (800, 700, 900, 400)}
    assert vanishing_point(lanes) is None


def test_missing_side_is_none():
    """요약: 한쪽 차선이 없으면 소실점·차선 가운데 모두 None."""
    lanes = {"left": (300, 700, 600, 430), "right": None}
    assert vanishing_point(lanes) is None
    assert lane_bottom_center_x(lanes) is None


def test_median_resists_outlier():
    """요약: 엉뚱한 소실점 하나가 섞여도 중앙값은 거의 그대로인지."""
    points = [(640, 400), (642, 401), (638, 399), (2000, 50)]
    out = estimate_calibration(points, [640] * 4, W, H, CALIB)
    assert out["vanishing_x_ratio"] == pytest.approx(641 / W)


def test_too_few_points_is_none():
    """요약: 유효 소실점이 min_points보다 적으면 추천하지 않음."""
    assert estimate_calibration([(640, 400)] * 2, [640] * 2, W, H, CALIB) is None


def test_fake_road_gives_center():
    """요약: 가운데로 모이는 가짜 도로에서 vanishing_x_ratio·center_x_ratio가 약 0.5인지."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.line(img, (300, 700), (600, 430), (255, 255, 255), 8)
    cv2.line(img, (980, 700), (680, 430), (255, 255, 255), 8)
    lanes = detect_lanes(img, CFG["camera"], CFG["lane"])
    n = CALIB["min_points"]
    out = estimate_calibration([vanishing_point(lanes)] * n, [lane_bottom_center_x(lanes)] * n, W, H, CALIB)
    assert out["vanishing_x_ratio"] == pytest.approx(0.5, abs=0.01)
    assert out["center_x_ratio"] == pytest.approx(0.5, abs=0.01)
