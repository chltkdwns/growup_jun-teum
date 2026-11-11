"""여러 프레임의 차선으로 카메라 보정값(소실점 위치, 내 차 위치, 관심 영역 높이)을 추정한다.

보정 도구(tools/calibrate_camera.py)에서만 쓴다. 매 프레임 분석에는 쓰지 않는다.
"""
import numpy as np

from app.lane.detect import Line


def _x_of_y(line: Line) -> tuple[float, float]:
    """직선을 x = m·y + k 꼴로 바꾼다. 차선은 세로에 가까워 y로 x를 구한다 (detect._fit_line과 같은 이유)."""
    xb, yb, xt, yt = line
    m = (xt - xb) / (yt - yb)
    return m, xb - m * yb


def vanishing_point(lanes: dict) -> tuple[float, float] | None:
    """좌·우 차선 직선의 교점(소실점)을 구한다.

    사용처: 보정 도구. 프레임마다 detect_lanes() 결과로 호출한다.
    입력: lanes {"left": Line|None, "right": Line|None} (Line = x_bottom, y_bottom, x_top, y_top px)
    출력: (x px, y px) 또는 한쪽이 없거나 두 직선이 평행이면 None
    주의: 한 프레임 값은 흔들리므로 estimate_calibration()에서 여러 프레임의 중앙값을 쓴다.
    """
    if lanes["left"] is None or lanes["right"] is None:
        return None
    m1, k1 = _x_of_y(lanes["left"])
    m2, k2 = _x_of_y(lanes["right"])
    if m1 == m2:
        return None  # 평행하면 만나지 않음
    y = (k2 - k1) / (m1 - m2)
    return m1 * y + k1, y


def lane_bottom_center_x(lanes: dict) -> float | None:
    """화면 아래 높이에서 좌·우 차선의 가운데 x를 구한다 (내 차 위치 추정용).

    사용처: 보정 도구. 프레임마다 detect_lanes() 결과로 호출한다.
    입력: lanes {"left": Line|None, "right": Line|None}
    출력: x(px) 또는 한쪽이 없으면 None
    주의: 운전자가 대체로 차선 가운데로 달린다고 가정한다. 카메라가 옆에 붙어 있으면 소실점 x와 달라진다.
          값이 비교 높이(y_bottom = bonnet_ratio)에 따라 달라지므로, bonnet_ratio가 맞는 설정으로 구해야 한다.
    """
    if lanes["left"] is None or lanes["right"] is None:
        return None
    return (lanes["left"][0] + lanes["right"][0]) / 2


def estimate_calibration(
    points: list[tuple[float, float]], centers_x: list[float], img_w: int, img_h: int, cfg_calib: dict
) -> dict | None:
    """여러 프레임의 소실점·차선 가운데 값의 중앙값으로 보정값을 추천한다.

    사용처: 보정 도구. 영상을 다 돈 뒤 한 번 호출한다.
    입력: points 소실점 [(x px, y px), ...], centers_x 화면 아래 차선 가운데 x(px) 목록, img_w·img_h 프레임 크기(px),
          cfg_calib(min_points 최소 개수, horizon_margin_ratio 소실점보다 얼마나 아래부터 관심 영역을 시작할지)
    출력: {"vanishing_x_ratio", "center_x_ratio", "roi_top_ratio", "n_points", "spread_x_px"} 또는 개수 부족이면 None
          spread_x_px는 소실점 x의 사분위 범위(px). 클수록 값이 흔들린 것이라 덜 믿을 만하다.
    주의: 평균이 아니라 중앙값을 쓴다. 끼어드는 차 때문에 엉뚱한 선을 잡은 프레임(이상값)에 덜 흔들린다.
    """
    if len(points) < cfg_calib["min_points"] or len(centers_x) < cfg_calib["min_points"]:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    q1, q3 = np.percentile(xs, [25, 75])
    return {
        "vanishing_x_ratio": float(np.median(xs)) / img_w,
        "center_x_ratio": float(np.median(centers_x)) / img_w,
        "roi_top_ratio": float(np.median(ys)) / img_h + cfg_calib["horizon_margin_ratio"],
        "n_points": len(points),
        "spread_x_px": float(q3 - q1),
    }
