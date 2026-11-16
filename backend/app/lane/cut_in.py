"""4단계(위험 판단): 옆 차로 차가 내 차선으로 들어오는 끼어들기를 찾는다.

흐름: 차마다 박스 아래 가운데(바퀴 위치)가 내 차선 안인지 기록 → 밖에서 안으로 들어온 차를 후보로
     → 앞차보다 가깝거나(앞차가 없으면 화면 아래쪽) → 그 상태가 confirm_s초 이어지면 끼어들기 확정
출력 형식은 임시다 (cut_in = {"track_id", "side", "t_s"}, 이름·단위는 경고 판단 담당과 합의 필요).
"""
from app.lane.front_car import x_on_line


class CutInDetector:
    """추적 중인 옆 차가 내 차선 경계를 안쪽으로 넘어 가까이 들어오면 끼어들기로 알린다.

    사용처: 4단계(위험 판단). 매 프레임 LaneTracker·detect_vehicles·FrontCarTracker 다음에 update()를 호출한다.
    주의: 시간은 촬영 시각(t_s) 기준이고, 같은 차를 알아보려고 track_id를 쓰므로 탐지 추적(track)이 켜져 있어야 한다.
          클래스로 만든 이유: 차마다 "직전에 내 차선 밖이었는지", "언제부터 안에 있었는지"를 프레임 사이에 기억해야 해서.
          내 차가 차로를 바꾸면(lane_change) 모든 차가 상대적으로 경계를 넘어 보이므로 ignore_after_lane_change_s 동안 판단하지 않는다.
          차선이 안 보이는 프레임(좌우 중 하나라도 None)은 판단하지 않는다. 기본 차선 영역은 실제 경계가 아니라서 오탐이 많다.
    """

    def __init__(self, cfg_cut_in: dict) -> None:
        """설정값을 읽고 차별 기록을 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_cut_in config의 cut_in 섹션
              (confirm_s 확정까지 안에 있어야 하는 시간(s), near_y_ratio 앞차가 없을 때 "가까움"으로 볼 박스 아래 높이(화면 높이 대비),
               ignore_after_lane_change_s 내 차로 변경 뒤 판단을 쉬는 시간(s))
        """
        self.confirm_s = cfg_cut_in["confirm_s"]
        self.near_y_ratio = cfg_cut_in["near_y_ratio"]
        self.ignore_s = cfg_cut_in["ignore_after_lane_change_s"]
        self.outside_side = {}  # track_id → 마지막으로 내 차선 밖에 있던 쪽 "left"|"right" (안에 있으면 지움)
        self.entered = {}  # track_id → (안으로 들어온 촬영 시각 s, 들어온 쪽)
        self.reported = set()  # 이미 끼어들기로 알린 track_id (차선 밖으로 나가면 지움)
        self.last_lane_change_s = None
        self.prev_front_id = None  # 직전 프레임의 앞차 track_id (이미 내 앞에 있던 차는 끼어들기가 아님)

    def update(
        self, dets: list[dict], lanes: dict, front: dict | None, t_s: float, img_h: int, lane_change: str | None = None
    ) -> dict | None:
        """이번 프레임에서 끼어들기가 확정된 차가 있으면 알려 준다.

        사용처: 4단계(위험 판단). 매 프레임 호출한다.
        입력: dets detect_vehicles() 결과, lanes LaneTracker.update() 결과, front FrontCarTracker.update() 결과(없으면 None),
              t_s 촬영 시각(s), img_h 프레임 높이(px), lane_change LaneTracker 결과의 "lane_change"
        출력: {"track_id": int, "side": "left"|"right"(들어온 쪽), "t_s": 확정 시각(s)} 또는 없으면 None
        주의: 셋 다 맞아야 끼어들기다.
              1) 박스 아래 가운데가 내 차선 밖에 있다가 안으로 들어왔다.
              2) 지금 앞차(다른 차)보다 가깝거나, 앞차가 없으면 박스 아래가 near_y_ratio보다 아래(가까움)다.
              3) 안에 들어온 상태가 confirm_s초 이상 이어졌다.
              같은 차는 차선 밖으로 다시 나가기 전까지 한 번만 알린다. 한 프레임에 여러 대면 가장 가까운 차 하나만 알린다.
              안으로 들어오는 순간에 "직전 프레임의 앞차"였던 차는 제외한다. 차선 경계가 흔들려 앞차가 잠깐
              밖으로 나갔다 들어온 것처럼 보여도 끼어들기가 아니기 때문이다 (시내 영상에서 앞차 track 1이 두 번 잘못 잡힌 문제).
              "이번 프레임" 앞차가 아니라 "직전 프레임" 앞차로 보는 이유: 앞차가 없을 때 들어온 차는
              앞차 트래커가 그 프레임에 바로 앞차로 삼아서, 이번 프레임 기준이면 진짜 끼어들기를 놓친다.
        """
        prev_front_id = self.prev_front_id  # 직전 프레임 앞차 (아래에서 쓰고, 이번 앞차로 바꿔 둔다)
        self.prev_front_id = None if front is None else front["track_id"]
        if lane_change is not None:
            self.last_lane_change_s = t_s
        if lanes["left"] is None or lanes["right"] is None:
            return None  # 차선을 모르면 경계도 모름
        # 내 차로 변경 직후에는 판단하지 않고 차별 안/밖 기록만 새로 쌓는다
        judging = self.last_lane_change_s is None or t_s - self.last_lane_change_s > self.ignore_s
        if not judging:
            self.entered.clear()

        best, best_y = None, -1.0
        for d in dets:
            tid = d["track_id"]
            if tid is None:
                continue
            x1, _, x2, y2 = d["box_xyxy_px"]
            x_c = (x1 + x2) / 2
            x_left, x_right = x_on_line(lanes["left"], y2), x_on_line(lanes["right"], y2)
            if x_c < x_left or x_c > x_right:
                # 1) 밖에 있음: 어느 쪽 밖인지 기억하고, 진행 중인 기록은 지운다
                self.outside_side[tid] = "left" if x_c < x_left else "right"
                self.entered.pop(tid, None)
                self.reported.discard(tid)
                continue
            if tid in self.outside_side:
                # 밖 → 안으로 들어온 순간. 들어오기 직전에 이미 앞차였다면 경계 흔들림이라 기록하지 않는다
                if judging and tid != prev_front_id:
                    self.entered[tid] = (t_s, self.outside_side[tid])
                del self.outside_side[tid]
            if tid not in self.entered or tid in self.reported:
                continue
            # 2) 가까운가: 앞차(다른 차)보다 아래에 있거나, 앞차가 없으면 화면 아래쪽
            front_y = None if front is None or front["track_id"] == tid else front["box_xyxy_px"][3]
            near = y2 > front_y if front_y is not None else y2 >= img_h * self.near_y_ratio
            # 3) 안에 들어온 지 confirm_s초 이상
            if near and t_s - self.entered[tid][0] >= self.confirm_s and y2 > best_y:
                best, best_y = d, y2
        if best is None:
            return None
        tid = best["track_id"]
        self.reported.add(tid)
        return {"track_id": tid, "side": self.entered.pop(tid)[1], "t_s": t_s}
