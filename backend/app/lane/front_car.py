"""2단계(앞차): 탐지된 차 중 내 차선 안에서 가장 가까운 차를 고른다.

흐름: 차선(없으면 기본 차선 영역) → 차마다 박스 아래 가운데가 두 차선 사이인지 → 그중 박스 아래가 가장 아래인 차
"""
from app.lane.detect import Line


def x_on_line(line: Line, y: float) -> float:
    """차선 직선이 높이 y에서 지나는 x를 구한다 (직선 연장 포함).

    사용처: select_front_car()에서 차마다 박스 아래 높이의 차선 위치를 구할 때.
    입력: line (x_bottom, y_bottom, x_top, y_top) px, y 높이(px)
    출력: x(px)
    """
    xb, yb, xt, yt = line
    return xb + (xt - xb) * (y - yb) / (yt - yb)


def default_lanes(img_w: int, img_h: int, cfg_camera: dict, cfg_lane: dict, cfg_front: dict) -> dict:
    """차선을 못 찾았을 때 쓸 기본 차선 영역(왼쪽·오른쪽 직선)을 만든다.

    사용처: select_front_car()에서 좌우 차선 중 하나라도 없을 때.
    입력: img_w·img_h 프레임 크기(px), cfg_camera(bonnet_ratio, center_x_ratio, vanishing_x_ratio),
          cfg_lane(roi_top_ratio, roi_top_half_width_ratio), cfg_front(default_lane_half_width_ratio)
    출력: {"left": Line, "right": Line} 형식은 detect_lanes()와 같음
    주의: 아래쪽은 내 차 위치(center_x) 기준 고정 폭, 위쪽은 관심 영역 윗변(소실점 쪽)으로 모이는 사다리꼴이다.
          실제 차선 폭과 다를 수 있어 차선이 보일 때보다 부정확하다 (SPEC: 차선 인식 불가 + 기본 차선 영역).
    """
    y_bottom = img_h * cfg_camera["bonnet_ratio"]
    y_top = img_h * cfg_lane["roi_top_ratio"]
    car_x = img_w * cfg_camera["center_x_ratio"]
    vx = img_w * cfg_camera["vanishing_x_ratio"]
    half_bottom = img_w * cfg_front["default_lane_half_width_ratio"]
    half_top = img_w * cfg_lane["roi_top_half_width_ratio"]
    return {
        "left": (car_x - half_bottom, y_bottom, vx - half_top, y_top),
        "right": (car_x + half_bottom, y_bottom, vx + half_top, y_top),
    }


def select_front_car(
    dets: list[dict], lanes: dict, img_w: int, img_h: int, cfg_camera: dict, cfg_lane: dict, cfg_front: dict
) -> dict | None:
    """내 차선 안의 차(이륜차 포함) 중 가장 가까운 차 하나를 고른다.

    사용처: 2단계(앞차). 매 프레임 detect_vehicles()와 LaneTracker.update() 다음에 호출하고,
            결과로 거리·TTC·브레이크등을 판단한다.
    입력: dets detect_vehicles() 결과 목록, lanes LaneTracker.update() 결과({"left", "right"} Line 또는 None),
          img_w·img_h 프레임 크기(px), cfg_camera·cfg_lane·cfg_front config 섹션
    출력: 고른 차의 dict (dets 항목에 "lane_source": "lane"|"default"를 붙인 사본) 또는 내 차선에 차가 없으면 None
          lane_source: 실제 차선(트래커 추정 포함)으로 골랐는지, 기본 차선 영역으로 골랐는지
    주의: 차 위치는 박스 아래 가운데(바퀴가 땅에 닿는 곳)로 본다. 차선은 그 높이에서의 x로 비교한다.
          "가장 가까움"은 박스 아래 y가 가장 큰(화면 아래쪽) 차다. 박스 폭은 차종마다 달라서 쓰지 않는다.
          좌우 차선 중 하나라도 없으면 기본 차선 영역(default_lanes)을 쓴다.
          선 위에 걸친 차(끼어드는 중)는 박스 아래 가운데가 어느 쪽인지로만 정한다 (끼어들기 판단은 2차 기능).
          트래커가 한쪽 차선을 폭으로 추정한 프레임(estimated)은 그 추정 오차만큼 경계가 어긋날 수 있다.
          차로 변경(lane_change) 직후 앞차가 바뀌는 것은 정상이다 (내 차선 자체가 바뀜).
          보닛 아래는 탐지 단계에서 이미 잘려 있어, 아주 가까운 차는 박스 아래가 보닛 선에 붙어 있을 수 있다.
    """
    source = "lane"
    if lanes["left"] is None or lanes["right"] is None:
        lanes = default_lanes(img_w, img_h, cfg_camera, cfg_lane, cfg_front)
        source = "default"
    best, best_y = None, -1.0
    for d in dets:
        x1, _, x2, y2 = d["box_xyxy_px"]
        x_c = (x1 + x2) / 2
        if x_on_line(lanes["left"], y2) <= x_c <= x_on_line(lanes["right"], y2) and y2 > best_y:
            best, best_y = d, y2
    return None if best is None else dict(best, lane_source=source)
