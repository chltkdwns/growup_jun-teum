"""프레임마다 흔들리는 차선 검출값을 시간 기준으로 부드럽게 만든다 (실시간·파일 공용)."""
from collections import deque

SIDES = ("left", "right")


class LaneTracker:
    """최근 smooth_window_s초 검출값의 평균을 내고, 잠깐 놓친 차선은 hold_s초 동안 유지한다.

    사용처: 2단계(차선). detect_lanes() 바로 뒤에서 매 프레임 update()를 호출한다.
    주의: 프레임 수가 아니라 촬영 시각(t_s)으로 판단한다. fps가 바뀌어도 같은 시간 범위를 본다.
          클래스로 만든 이유: 이전 프레임들의 검출값(상태)을 기억해야 해서.
    """

    def __init__(self, cfg_lane: dict) -> None:
        """설정값을 읽고 좌·우 기록을 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_lane config의 lane 섹션 (smooth_window_s 평균 낼 시간(s), hold_s 놓쳐도 유지할 시간(s))
        """
        self.window_s = cfg_lane["smooth_window_s"]
        self.hold_s = cfg_lane["hold_s"]
        # 쪽마다 (촬영 시각 s, x_bottom px, x_top px) 기록
        self.history = {side: deque() for side in SIDES}
        self.last_seen_s = {side: None for side in SIDES}
        self.last_out = {side: None for side in SIDES}

    def update(self, lanes: dict, t_s: float) -> dict:
        """이번 프레임 검출 결과를 넣고, 부드럽게 만든 차선을 돌려준다.

        사용처: 2단계(차선). 매 프레임 detect_lanes() 결과로 호출한다.
        입력: lanes detect_lanes() 결과 {"left": Line|None, "right": Line|None}, t_s 촬영 시각(s, 늘어나기만 해야 함)
        출력: {"left": Line|None, "right": Line|None} 형식은 detect_lanes()와 같음
        주의: 마지막 검출 뒤 hold_s초가 넘게 못 찾으면 None. 급한 차로 변경에서는 평균 때문에 약간 늦게 따라간다.
        """
        out = {}
        for side in SIDES:
            line = lanes[side]
            if line is not None:
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
                out[side] = self.last_out[side]
            elif self.last_seen_s[side] is not None and t_s - self.last_seen_s[side] <= self.hold_s:
                out[side] = self.last_out[side]  # 점선 빈 구간처럼 잠깐 놓친 경우 직전 값 유지
            else:
                out[side] = None
        return out
