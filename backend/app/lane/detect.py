"""한 프레임에서 내 차선의 왼쪽·오른쪽 차선을 찾는다 (OpenCV 첫 버전).

흐름: 회색 변환 → 흐림 → 윤곽선(Canny) → 관심 영역(사다리꼴) 자르기 → 직선 찾기(Hough)
     → 기울기 부호로 왼쪽/오른쪽 나누기 → 각각 직선 하나로 합치기
"""
import cv2
import numpy as np

Line = tuple[int, int, int, int]  # (x_bottom, y_bottom, x_top, y_top), 픽셀


def lane_roi_polygon(img_w: int, img_h: int, cfg_camera: dict, cfg_lane: dict) -> np.ndarray:
    """차선을 찾을 관심 영역(사다리꼴) 꼭짓점을 구한다.

    사용처: 2단계(차선). detect_lanes()가 윤곽선 이미지를 자를 때, 실행 스크립트가 영역을 그릴 때.
    입력: img_w, img_h 프레임 크기(px), cfg_camera(bonnet_ratio, center_x_ratio), cfg_lane(roi_* 비율)
    출력: 꼭짓점 4개 [[x, y], ...] (int32), 왼쪽 아래 → 왼쪽 위 → 오른쪽 위 → 오른쪽 아래
    주의: 비율로 정해서 화면 크기가 달라도 같은 설정을 쓴다. 카메라 위치가 다르면 center_x_ratio를 조정한다.
    """
    cx = img_w * cfg_camera["center_x_ratio"]
    y_top = img_h * cfg_lane["roi_top_ratio"]
    y_bottom = img_h * cfg_camera["bonnet_ratio"]
    half_top = img_w * cfg_lane["roi_top_half_width_ratio"]
    return np.array(
        [[0, y_bottom], [cx - half_top, y_top], [cx + half_top, y_top], [img_w, y_bottom]],
        dtype=np.int32,
    )


def _fit_line(segments: list, y_bottom: float, y_top: float) -> Line | None:
    """같은 쪽 선분들을 직선 하나(x = a·y + b)로 합친다.

    사용처: detect_lanes() 안에서 왼쪽·오른쪽 각각 한 번씩.
    입력: segments [(x1, y1, x2, y2), ...], y_bottom·y_top 직선을 그릴 위아래 높이(px)
    출력: (x_bottom, y_bottom, x_top, y_top) 또는 선분이 없으면 None
    주의: 차선은 세로에 가까워 y로 x를 구하는 식을 쓴다 (x로 y를 구하면 기울기가 무한대에 가까워짐).
    """
    if not segments:
        return None
    ys = np.array([[s[1], s[3]] for s in segments], dtype=float).ravel()
    xs = np.array([[s[0], s[2]] for s in segments], dtype=float).ravel()
    a, b = np.polyfit(ys, xs, 1)
    return (int(a * y_bottom + b), int(y_bottom), int(a * y_top + b), int(y_top))


def detect_lanes(frame_bgr: np.ndarray, cfg_camera: dict, cfg_lane: dict) -> dict:
    """한 프레임에서 내 차선의 왼쪽·오른쪽 차선을 찾는다.

    사용처: 2단계(차선). 매 프레임 호출하고, 결과로 차선 이탈·앞차 선정을 한다.
    입력: frame_bgr 원본 프레임(BGR), cfg_camera·cfg_lane config 섹션
    출력: {"left": Line 또는 None, "right": Line 또는 None}
    주의: 첫 버전이라 직선만 찾는다 (급커브는 어긋남). 야간·우천·지워진 차선에서는 None이 많을 수 있다.
          왼쪽/오른쪽은 기울기 부호와 정면(center_x) 기준 위치로 나눈다.
    """
    img_h, img_w = frame_bgr.shape[:2]
    # 1) 윤곽선 찾기: 회색 → 흐림(노이즈 제거) → Canny
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, cfg_lane["canny_low"], cfg_lane["canny_high"])
    # 2) 관심 영역(내 차 앞 도로)만 남기기
    roi = lane_roi_polygon(img_w, img_h, cfg_camera, cfg_lane)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [roi], 255)
    edges = cv2.bitwise_and(edges, mask)
    # 3) 직선 선분 찾기
    segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=cfg_lane["hough_threshold"],
        minLineLength=img_w * cfg_lane["hough_min_len_ratio"],
        maxLineGap=img_w * cfg_lane["hough_max_gap_ratio"],
    )
    # 4) 기울기로 왼쪽/오른쪽 나누기 (화면 좌표는 y가 아래로 커지므로 왼쪽 차선은 기울기가 음수)
    cx = img_w * cfg_camera["center_x_ratio"]
    left, right = [], []
    # OpenCV 버전에 따라 결과 모양이 (N, 1, 4) 또는 (N, 4)라서 (N, 4)로 맞춘다
    segments = [] if segments is None else segments.reshape(-1, 4)
    for x1, y1, x2, y2 in segments:
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < cfg_lane["min_abs_slope"]:
            continue  # 거의 수평인 선은 차선이 아님
        if slope < 0 and max(x1, x2) < cx:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and min(x1, x2) > cx:
            right.append((x1, y1, x2, y2))
    # 5) 각각 직선 하나로 합치기
    y_bottom, y_top = roi[0][1], roi[1][1]
    return {"left": _fit_line(left, y_bottom, y_top), "right": _fit_line(right, y_bottom, y_top)}
