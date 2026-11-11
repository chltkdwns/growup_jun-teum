"""내 차 정면이 차선 중앙에서 얼마나 치우쳤는지 구한다 (차선 유지 보조용 값).

이탈 경고를 낼지 말지(기준 비율)는 경고 판단 담당 코드에서 정한다. 여기서는 값만 만든다.
"""

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"  # 화면에 "차선 인식 불가"로 표시


def lane_offset(lanes: dict, img_w: int, cfg_camera: dict) -> dict:
    """차선 중앙 대비 내 차 정면의 치우침 비율을 구한다.

    사용처: 2단계(차선). 매 프레임 LaneTracker.update() 결과로 호출한다 (run_lane, run_lane_live).
    입력: lanes {"left": Line|None, "right": Line|None} (Line = x_bottom, y_bottom, x_top, y_top px),
          img_w 프레임 폭(px), cfg_camera(center_x_ratio 정면 위치)
    출력: {"lane_offset_ratio": float|None, "lane_status": "ok"|"unavailable"}
          lane_offset_ratio: 0 = 정중앙, -1 = 왼쪽 선 위, +1 = 오른쪽 선 위, ±1 초과 = 선을 넘음 (단위 없음)
    주의: 화면 맨 아래(y_bottom) 높이에서 비교한다. 좌우 차선이 둘 다 있어야 계산한다.
          m 단위가 아니라 화면 속 차선 폭 대비 비율이라 차로 폭 가정이 필요 없다.
    """
    left, right = lanes["left"], lanes["right"]
    if left is None or right is None:
        return {"lane_offset_ratio": None, "lane_status": STATUS_UNAVAILABLE}
    half_w_px = (right[0] - left[0]) / 2
    if half_w_px <= 0:
        # 왼쪽 선이 오른쪽 선보다 오른쪽에 있으면 잘못 찾은 것이라 계산하지 않는다
        return {"lane_offset_ratio": None, "lane_status": STATUS_UNAVAILABLE}
    lane_center_x_px = (left[0] + right[0]) / 2
    car_x_px = img_w * cfg_camera["center_x_ratio"]
    return {"lane_offset_ratio": (car_x_px - lane_center_x_px) / half_w_px, "lane_status": STATUS_OK}
