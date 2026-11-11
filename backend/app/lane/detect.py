"""한 프레임에서 내 차선의 왼쪽·오른쪽 차선을 찾는다 (OpenCV 첫 버전).

흐름: 회색 변환 → 흐림 → 윤곽선(Canny) → 관심 영역(사다리꼴) 자르기 → 직선 찾기(Hough)
     → 선분을 화면 아래까지 연장해 선 묶음 만들기 → 내 차 기준 왼쪽·오른쪽에서 가장 가까운 묶음만 골라
     직선 하나로 합치기
"""
import cv2
import numpy as np

Line = tuple[int, int, int, int]  # (x_bottom, y_bottom, x_top, y_top), 픽셀


def lane_roi_polygon(img_w: int, img_h: int, cfg_camera: dict, cfg_lane: dict) -> np.ndarray:
    """차선을 찾을 관심 영역(사다리꼴) 꼭짓점을 구한다.

    사용처: 2단계(차선). detect_lanes()가 윤곽선 이미지를 자를 때, 실행 스크립트가 영역을 그릴 때.
    입력: img_w, img_h 프레임 크기(px), cfg_camera(bonnet_ratio, vanishing_x_ratio), cfg_lane(roi_* 비율)
    출력: 꼭짓점 4개 [[x, y], ...] (int32), 왼쪽 아래 → 왼쪽 위 → 오른쪽 위 → 오른쪽 아래
    주의: 비율로 정해서 화면 크기가 달라도 같은 설정을 쓴다.
          윗변은 소실점(카메라가 바라보는 방향, vanishing_x_ratio)에 맞춘다. 내 차 위치(center_x_ratio)와 다를 수 있다.
    """
    vx = img_w * cfg_camera["vanishing_x_ratio"]
    y_top = img_h * cfg_lane["roi_top_ratio"]
    y_bottom = img_h * cfg_camera["bonnet_ratio"]
    half_top = img_w * cfg_lane["roi_top_half_width_ratio"]
    return np.array(
        [[0, y_bottom], [vx - half_top, y_top], [vx + half_top, y_top], [img_w, y_bottom]],
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


def _group_lines(cands: list, cluster_px: float) -> list:
    """선분들을 화면 아래 끝 x가 가까운 것끼리 묶어 "선" 단위로 만든다.

    사용처: detect_lanes() 안에서 매 프레임 한 번.
    입력: cands [(x_bottom px, 선분), ...], cluster_px 같은 선으로 볼 아래 끝 x 간격(px)
    출력: [(묶음 아래 끝 x 중앙값 px, [선분, ...]), ...] 왼쪽부터
    주의: 점선 하나는 Hough 선분 여러 개로 나뉘므로 선분이 아니라 묶음 단위로 왼쪽/오른쪽을 정한다.
          그래야 차 바로 밑을 지나는 선 하나가 좌우로 쪼개져 양쪽 차선이 되는 일이 없다.
    """
    groups = []
    for x_b, seg in sorted(cands, key=lambda c: c[0]):
        if groups and x_b - groups[-1][-1][0] <= cluster_px:
            groups[-1].append((x_b, seg))
        else:
            groups.append([(x_b, seg)])
    return [(float(np.median([x for x, _ in g])), [seg for _, seg in g]) for g in groups]


def detect_lanes(frame_bgr: np.ndarray, cfg_camera: dict, cfg_lane: dict) -> dict:
    """한 프레임에서 내 차선의 왼쪽·오른쪽 차선을 찾는다.

    사용처: 2단계(차선). 매 프레임 호출하고, 결과로 차선 이탈·앞차 선정을 한다.
    입력: frame_bgr 원본 프레임(BGR), cfg_camera·cfg_lane config 섹션
    출력: {"left": Line 또는 None, "right": Line 또는 None}
    주의: 첫 버전이라 직선만 찾는다 (급커브는 어긋남). 야간·우천·지워진 차선에서는 None이 많을 수 있다.
          왼쪽/오른쪽은 선 묶음의 아래 끝 x(y_bottom까지 연장, 중앙값)가 내 차 위치(center_x)보다 왼쪽인지로 나눈다.
          같으면 오른쪽으로 넣는다. 같은 쪽에 묶음이 여러 개면 내 차에 가장 가까운 것만 쓴다 (먼 선 섞임 방지).
          차로 변경 중 선이 차 바로 밑을 지나는 순간에는
          그 선이 이쪽에서 저쪽으로 넘어가며 lane_offset_ratio가 +1 → -1(또는 반대)로 뒤집힌다 (실제로 선을 넘은 것).
          기울기 부호는 쓰지 않는다. 차 바로 앞을 지나는 선은 거의 수직이라 부호가 어느 쪽이든 될 수 있어서
          부호로 거르면 차선 이탈에 가장 중요한 선이 버려진다.
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
    # 4) 선분을 화면 아래(y_bottom)까지 연장한 x 구하기
    y_bottom, y_top = roi[0][1], roi[1][1]
    cands = []
    # OpenCV 버전에 따라 결과 모양이 (N, 1, 4) 또는 (N, 4)라서 (N, 4)로 맞춘다
    segments = [] if segments is None else segments.reshape(-1, 4)
    for x1, y1, x2, y2 in segments:
        if x1 != x2 and abs((y2 - y1) / (x2 - x1)) < cfg_lane["min_abs_slope"]:
            continue  # 거의 수평인 선은 차선이 아님 (수직선 x1 == x2 는 통과)
        # 직선을 x = x1 + (y - y1)·dx/dy 로 연장 (차선은 세로에 가까워 y로 x를 구함)
        cands.append((x1 + (y_bottom - y1) * (x2 - x1) / (y2 - y1), (x1, y1, x2, y2)))
    # 5) 선분을 선 묶음으로 만들고, 내 차 위치 기준 왼쪽·오른쪽에서 가장 가까운 묶음만 고르기
    car_x = img_w * cfg_camera["center_x_ratio"]
    groups = _group_lines(cands, img_w * cfg_lane["cluster_px_ratio"])
    left = [g for g in groups if g[0] < car_x]
    right = [g for g in groups if g[0] >= car_x]
    # 6) 각각 직선 하나로 합치기
    return {
        "left": _fit_line(left[-1][1] if left else [], y_bottom, y_top),
        "right": _fit_line(right[0][1] if right else [], y_bottom, y_top),
    }
