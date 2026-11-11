"""detect_lanes 테스트: 검은 바탕에 흰 차선 두 줄을 그린 가짜 이미지로 확인한다."""
import cv2
import numpy as np
import yaml
from pathlib import Path

from app.lane.detect import detect_lanes, lane_roi_polygon

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))
W, H = 1280, 720


def _fake_road(left: bool = True, right: bool = True) -> np.ndarray:
    """요약: 소실점(640, 400)으로 모이는 흰 차선을 그린 가짜 도로 이미지를 만든다."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if left:
        cv2.line(img, (300, 700), (600, 430), (255, 255, 255), 8)
    if right:
        cv2.line(img, (980, 700), (680, 430), (255, 255, 255), 8)
    return img


def _x_at(line, y):
    """요약: 찾은 직선이 높이 y에서 지나는 x."""
    xb, yb, xt, yt = line
    return xb + (xt - xb) * (y - yb) / (yt - yb)


def test_finds_both_lanes_near_drawn_lines():
    """요약: 그린 두 차선을 찾고, 높이 600px에서 x 오차가 10px 이내인지."""
    lanes = detect_lanes(_fake_road(), CFG["camera"], CFG["lane"])
    assert lanes["left"] is not None and lanes["right"] is not None
    # 그린 왼쪽 선: (300,700)→(600,430) 이므로 y=600에서 x≈411. 오른쪽은 대칭으로 x≈869
    assert abs(_x_at(lanes["left"], 600) - 411) < 10
    assert abs(_x_at(lanes["right"], 600) - 869) < 10


def test_missing_lane_is_none():
    """요약: 오른쪽 차선이 없으면 right는 None, 왼쪽은 그대로 찾는지."""
    lanes = detect_lanes(_fake_road(right=False), CFG["camera"], CFG["lane"])
    assert lanes["left"] is not None and lanes["right"] is None


def test_horizontal_line_ignored():
    """요약: 수평선(그림자·정지선 같은 것)만 있으면 차선으로 보지 않는지."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.line(img, (100, 550), (1180, 550), (255, 255, 255), 8)
    lanes = detect_lanes(img, CFG["camera"], CFG["lane"])
    assert lanes == {"left": None, "right": None}


def test_roi_follows_center_ratio():
    """요약: 정면 위치(center_x_ratio)를 바꾸면 관심 영역 윗변도 같이 움직이는지."""
    cam = dict(CFG["camera"], center_x_ratio=0.6)
    roi = lane_roi_polygon(W, H, cam, CFG["lane"])
    assert (roi[1][0] + roi[2][0]) / 2 == int(W * 0.6)
