"""프레임마다 흔들리는 차선 검출값을 시간 기준으로 부드럽게 만든다 (실시간·파일 공용)."""
from collections import deque
from statistics import median

SIDES = ("left", "right")


class LaneTracker:
    """최근 smooth_window_s초 검출값의 평균을 내고, 잠깐 놓친 차선은 hold_s초 동안 유지한다.
    한쪽만 보이면 최근 차선 폭으로 반대쪽을 추정한다 (차로 변경 중 선을 밟는 순간 대응).
    선이 한쪽에서 반대쪽으로 넘어오면 차로 변경으로 보고 위치 기록을 비운다 (이전 차로 선과 섞이지 않게).

    사용처: 2단계(차선). detect_lanes() 바로 뒤에서 매 프레임 update()를 호출한다.
    주의: 프레임 수가 아니라 촬영 시각(t_s)으로 판단한다. fps가 바뀌어도 같은 시간 범위를 본다.
          클래스로 만든 이유: 이전 프레임들의 검출값과 차선 폭(상태)을 기억해야 해서.
    """

    def __init__(self, cfg_lane: dict) -> None:
        """설정값을 읽고 좌·우 기록을 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_lane config의 lane 섹션 (smooth_window_s 평균 낼 시간(s), hold_s 놓쳐도 유지할 시간(s),
              width_window_s 차선 폭 중앙값을 낼 시간(s), estimate_max_s 두 선을 마지막으로 본 뒤 추정해도 되는 시간(s),
              lane_change_match_ratio 넘어온 선으로 볼 위치 차이(차선 폭 대비), lane_change_cooldown_s 재감지 금지 시간(s))
        """
        self.window_s = cfg_lane["smooth_window_s"]
        self.hold_s = cfg_lane["hold_s"]
        self.width_window_s = cfg_lane["width_window_s"]
        self.estimate_max_s = cfg_lane["estimate_max_s"]
        # 쪽마다 (촬영 시각 s, x_bottom px, x_top px) 기록
        self.history = {side: deque() for side in SIDES}
        self.last_seen_s = {side: None for side in SIDES}
        self.last_out = {side: None for side in SIDES}
        # 두 선을 함께 본 프레임의 (촬영 시각 s, 아래 폭 px, 위 폭 px) 기록
        self.widths = deque()
        self.match_ratio = cfg_lane["lane_change_match_ratio"]
        self.cooldown_s = cfg_lane["lane_change_cooldown_s"]
        self.last_change_s = None

    def update(self, lanes: dict, t_s: float) -> dict:
        """이번 프레임 검출 결과를 넣고, 부드럽게 만든(필요하면 추정한) 차선을 돌려준다.

        사용처: 2단계(차선). 매 프레임 detect_lanes() 결과로 호출한다.
        입력: lanes detect_lanes() 결과 {"left": Line|None, "right": Line|None}, t_s 촬영 시각(s, 늘어나기만 해야 함)
        출력: {"left": Line|None, "right": Line|None, "estimated": 폭으로 추정한 쪽 tuple, 예: ("right",) 또는 (),
               "lane_change": 이번 프레임에 감지한 차로 변경 방향 "left"|"right" 또는 None}
        주의: 규칙 순서
              0) 차로 변경을 먼저 확인한다. 감지되면 좌우 위치 기록과 hold를 비운다 (차선 폭 기록은 유지).
              1) 이번 프레임에 실제로 찾은 쪽은 평균값을 쓴다.
              2) 한쪽만 찾았고 차선 폭을 알면, 못 찾은 쪽은 hold보다 "찾은 쪽 ± 폭" 추정을 우선한다.
                 hold 값은 몇 프레임 전 위치라 차로 변경처럼 차가 옆으로 움직일 때 어긋나지만,
                 추정값은 지금 보이는 선을 따라 함께 움직이기 때문.
              3) 그 밖에 못 찾은 쪽은 마지막 검출 뒤 hold_s초까지 직전 값, 넘으면 None.
              추정값은 차선 폭이 그대로라고 가정해서 오차가 있다. 급한 차로 변경은 평균 때문에 약간 늦게 따라간다.
        """
        lane_change = self._detect_lane_change(lanes, t_s)
        if lane_change is not None:
            self.history = {side: deque() for side in SIDES}
            self.last_seen_s = {side: None for side in SIDES}
            self.last_out = {side: None for side in SIDES}
            self.last_change_s = t_s

        out = {}
        for side in SIDES:
            if lanes[side] is not None:
                out[side] = self._smooth(side, lanes[side], t_s)
            elif self.last_seen_s[side] is not None and t_s - self.last_seen_s[side] <= self.hold_s:
                out[side] = self.last_out[side]  # 점선 빈 구간처럼 잠깐 놓친 경우 직전 값 유지
            else:
                out[side] = None
        out["estimated"] = ()

        found = [side for side in SIDES if lanes[side] is not None]
        if len(found) == 2:
            self._remember_width(out, t_s)
        elif len(found) == 1:
            width = self._recent_width(t_s)
            if width is not None:
                missing = "left" if found[0] == "right" else "right"
                out[missing] = self._shift(out[found[0]], width, to_left=missing == "left")
                out["estimated"] = (missing,)
        out["lane_change"] = lane_change
        return out

    def _detect_lane_change(self, lanes: dict, t_s: float) -> str | None:
        """직전 오른쪽(왼쪽) 선이 이번 프레임에 왼쪽(오른쪽) 선으로 넘어왔는지 본다.

        사용처: update() 맨 앞에서 매 프레임.
        입력: lanes 이번 프레임 detect_lanes() 결과, t_s 촬영 시각(s)
        출력: "right"(내 차가 오른쪽 차로로 감) | "left" | None
        주의: 위치 기준이다. 이번 왼쪽 선의 아래 끝 x가 직전 오른쪽 선과 (차선 폭 × match_ratio) 이내로 가깝고,
              직전 왼쪽 선과는 그보다 멀면 같은 선이 넘어온 것으로 본다. 직전 선은 hold_s 안에 본 것만 쓴다.
              차선 폭을 모르거나(estimate_max_s 초과) 쿨다운(cooldown_s) 중이면 감지하지 않는다.
        """
        if self.last_change_s is not None and t_s - self.last_change_s < self.cooldown_s:
            return None
        width = self._recent_width(t_s)
        if width is None:
            return None
        tol_px = width[0] * self.match_ratio
        for new_side, old_side, direction in (("left", "right", "right"), ("right", "left", "left")):
            line = lanes[new_side]
            old = self._recent_line(old_side, t_s)
            if line is None or old is None or abs(line[0] - old[0]) > tol_px:
                continue
            same = self._recent_line(new_side, t_s)
            if same is None or abs(line[0] - same[0]) > tol_px:
                return direction
        return None

    def _recent_line(self, side: str, t_s: float) -> tuple | None:
        """hold_s 안에 마지막으로 본 한쪽 차선(평균값)을 돌려준다. 오래됐으면 None.

        사용처: _detect_lane_change()에서 직전 선 위치 비교용.
        입력: side "left"|"right", t_s 촬영 시각(s)
        출력: Line(px) 또는 None
        """
        seen = self.last_seen_s[side]
        if seen is None or t_s - seen > self.hold_s:
            return None
        return self.last_out[side]

    def _smooth(self, side: str, line: tuple, t_s: float) -> tuple:
        """한쪽 차선의 최근 smooth_window_s초 x값 평균을 낸다.

        사용처: update()에서 이번 프레임에 찾은 쪽마다.
        입력: side "left"|"right", line 이번 검출 Line(px), t_s 촬영 시각(s)
        출력: 평균 낸 Line (x_bottom, y_bottom, x_top, y_top px)
        """
        xb, yb, xt, yt = line
        hist = self.history[side]
        hist.append((t_s, xb, xt))
        # window보다 오래된 값은 버린다 (경계값 t_s - window_s 는 포함)
        while hist[0][0] < t_s - self.window_s:
            hist.popleft()
        xb_avg = round(sum(h[1] for h in hist) / len(hist))
        xt_avg = round(sum(h[2] for h in hist) / len(hist))
        self.last_seen_s[side] = t_s
        self.last_out[side] = (xb_avg, yb, xt_avg, yt)
        return self.last_out[side]

    def _remember_width(self, out: dict, t_s: float) -> None:
        """두 선을 함께 찾은 프레임의 차선 폭(아래·위 x 차이)을 기록한다.

        사용처: update()에서 좌우를 이번 프레임에 둘 다 실제로 찾았을 때만.
        입력: out 평균 낸 좌우 Line, t_s 촬영 시각(s)
        주의: 추정·유지값으로는 폭을 기록하지 않는다 (추정이 추정을 낳아 오차가 쌓이는 것을 막음).
        """
        left, right = out["left"], out["right"]
        self.widths.append((t_s, right[0] - left[0], right[2] - left[2]))
        while self.widths[0][0] < t_s - self.width_window_s:
            self.widths.popleft()

    def _recent_width(self, t_s: float) -> tuple[float, float] | None:
        """기억한 차선 폭의 중앙값을 돌려준다. 두 선을 마지막으로 함께 본 지 estimate_max_s초가 넘었으면 None.

        사용처: update()에서 한쪽만 찾았을 때.
        입력: t_s 촬영 시각(s)
        출력: (아래 폭 px, 위 폭 px) 또는 None
        주의: 중앙값을 쓰는 이유: 어쩌다 엉뚱한 선을 잡은 프레임의 폭에 덜 흔들린다.
        """
        if not self.widths or t_s - self.widths[-1][0] > self.estimate_max_s:
            return None
        return median(w[1] for w in self.widths), median(w[2] for w in self.widths)

    @staticmethod
    def _shift(line: tuple, width: tuple[float, float], to_left: bool) -> tuple:
        """보이는 차선을 차선 폭만큼 옆으로 옮겨 반대쪽 차선을 만든다.

        사용처: update()에서 한쪽만 찾았을 때.
        입력: line 보이는 쪽 Line(px), width (아래 폭, 위 폭) px, to_left 왼쪽으로 옮길지
        출력: 추정한 Line (x_bottom, y_bottom, x_top, y_top px)
        """
        xb, yb, xt, yt = line
        sign = -1 if to_left else 1
        return (round(xb + sign * width[0]), yb, round(xt + sign * width[1]), yt)
