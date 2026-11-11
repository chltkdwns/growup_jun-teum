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


class FrontCarTracker:
    """앞차가 프레임마다 바뀌지 않게, 새 차가 switch_hold_s초 이상 계속 1등일 때만 앞차를 바꾼다.

    사용처: 2단계(앞차). 매 프레임 select_front_car() 바로 뒤에서 update()를 호출한다.
    주의: 시간은 촬영 시각(t_s) 기준이다. 차를 같은 차로 알아보려고 track_id를 쓰므로 탐지 추적(track)이 켜져 있어야 한다.
          클래스로 만든 이유: 지금 앞차와 "바꿀 후보가 언제부터 1등이었는지"를 프레임 사이에 기억해야 해서.
          유지 중인 앞차가 내 차선 밖으로 나가도 최대 switch_hold_s초는 앞차로 남는다.
          단, 차로 변경을 감지한 프레임에는 기다리지 않고 바로 바꾼다 (내 차선 자체가 바뀌었으므로).
          차선 없음(default) 구간도 같은 규칙을 쓴다. 기본 영역이 잠깐 엉뚱한 차를 1등으로 만들어도
          switch_hold_s 안에 차선이 돌아오면 교체되지 않는다. 규칙을 하나로 두어 설명을 단순하게 했다.
    """

    def __init__(self, cfg_front: dict) -> None:
        """설정값을 읽고 앞차 기록을 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_front config의 front_car 섹션 (switch_hold_s 교체 전 대기 시간(s), lost_hold_s 사라진 앞차 유지 시간(s))
        """
        self.switch_hold_s = cfg_front["switch_hold_s"]
        self.lost_hold_s = cfg_front["lost_hold_s"]
        self.current = None  # 지금 앞차 dict (select_front_car 결과 형식)
        self.last_seen_s = None  # 지금 앞차가 탐지에 마지막으로 보인 촬영 시각
        self.pending_id = None  # 바꿀 후보의 track_id (후보가 "앞차 없음"이면 None)
        self.pending_since_s = None  # 그 후보가 1등이 된 촬영 시각

    def update(self, candidate: dict | None, dets: list[dict], t_s: float, lane_change: str | None = None) -> dict | None:
        """이번 프레임 1등 후보를 받아, 유지 규칙을 적용한 앞차를 돌려준다.

        사용처: 2단계(앞차). 매 프레임 select_front_car() 결과로 호출한다.
        입력: candidate select_front_car() 결과(없으면 None), dets 이번 프레임 detect_vehicles() 전체 결과, t_s 촬영 시각(s),
              lane_change LaneTracker.update() 결과의 "lane_change" (차로 변경 감지 프레임에 "left"|"right")
        출력: 앞차 dict(+ "front_changed": 이번 프레임에 앞차가 바뀌었으면 True) 또는 앞차 없음 None
        주의: 규칙 순서
              0) 차로 변경을 감지한 프레임이면 후보를 바로 쓴다.
              1) 지금 앞차가 없으면 후보를 바로 쓴다.
              2) 후보가 지금 앞차와 같은 차면 박스를 최신으로 바꿔 유지한다.
              3) 지금 앞차가 탐지에서 사라졌으면 lost_hold_s초까지 마지막 박스로 유지하고, 넘으면 후보로 바꾼다.
              4) 다른 후보(또는 "앞차 없음")가 switch_hold_s초 이상 계속 1등이면 바꾸고, 그 전에는 지금 앞차를 유지한다.
        """
        cand_id = None if candidate is None else candidate["track_id"]
        cur_id = None if self.current is None else self.current["track_id"]
        # 지금 앞차가 이번 프레임 탐지에 있는지 (있으면 최신 박스를 쓰기 위해)
        cur_now = next((d for d in dets if cur_id is not None and d["track_id"] == cur_id), None)
        if cur_now is not None:
            self.last_seen_s = t_s

        if lane_change is not None or self.current is None or cand_id == cur_id:
            take_candidate = True  # 0), 1), 2)
        elif cur_now is None and t_s - self.last_seen_s > self.lost_hold_s:
            take_candidate = True  # 3) 사라진 지 오래됨
        else:
            # 4) 다른 후보가 언제부터 1등이었는지 기록하고, 충분히 오래됐을 때만 바꾼다
            if self.pending_since_s is None or cand_id != self.pending_id:
                self.pending_id, self.pending_since_s = cand_id, t_s
            take_candidate = t_s - self.pending_since_s >= self.switch_hold_s

        if take_candidate:
            new = candidate
            self.pending_id, self.pending_since_s = None, None
            if candidate is not None:
                self.last_seen_s = t_s
        elif cur_now is not None:
            new = dict(cur_now, lane_source=self.current["lane_source"])  # 유지, 박스만 최신으로
        else:
            new = self.current  # 사라졌지만 lost_hold_s 안: 마지막 박스로 유지

        new_id = None if new is None else new["track_id"]
        self.current = new
        return None if new is None else dict(new, front_changed=new_id != cur_id)
