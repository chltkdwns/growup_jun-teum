"""4단계(거리·위험): 앞차까지 거리(m)와 TTC(충돌까지 남은 시간, s)를 구한다.

거리: 핀홀 공식. 거리 = 초점거리(px) × 실제 차 폭(m) ÷ 화면 속 박스 폭(px).
TTC: 최근 ttc_window_s초 동안 1/박스 폭을 촬영 시각에 대해 직선으로 맞추고, 그 직선이 0이 되기까지 남은 시간.
     1/박스 폭은 거리에 비례하므로(거리 = 상수 × 1/박스 폭) 실제 차 폭이나 초점거리를 몰라도 TTC가 나온다.
     두 프레임의 거리 차이로 구하지 않는다 (박스 떨림이 그대로 오차가 되기 때문, SPEC).
출력 이름·단위는 schemas.LeadRisk (distance_m, ttc_s, 2026-11-17 세 명 합의).
"""
import math
from collections import deque

import numpy as np


def focal_px(img_w_px: int, hfov_deg: float) -> float:
    """카메라 수평 화각으로 초점거리(px)를 구한다.

    사용처: 4단계. LeadRiskTracker.update()에서 매 프레임 (프레임 크기가 바뀌어도 맞게).
    입력: img_w_px 프레임 가로(px), hfov_deg 카메라 수평 화각(도, config risk.hfov_deg)
    출력: 초점거리(px)
    주의: 화면 가운데 기준 공식이라 광각 블랙박스의 렌즈 왜곡(가장자리가 휘어 보임)은 반영하지 않는다.
    """
    # 화면 절반 폭 ÷ tan(화각 절반) = 렌즈 중심에서 화면까지의 거리(px)
    return (img_w_px / 2) / math.tan(math.radians(hfov_deg) / 2)


def estimate_distance_m(box_w_px: float, focal_length_px: float, real_w_m: float) -> float:
    """박스 폭으로 차까지의 거리(m)를 구한다.

    사용처: 4단계. LeadRiskTracker.update()에서 매 프레임 앞차에 대해.
    입력: box_w_px 박스 폭(px), focal_length_px 초점거리(px), real_w_m 실제 폭(m, config: 승용차 1.8, 이륜차 0.8)
    출력: 거리(m)
    주의: 트럭·경차처럼 실제 폭이 다르거나, 화면 가장자리에서 잘린 박스는 오차가 크다.
    """
    # 핀홀 공식: 거리 = 초점거리 × 실제 폭 ÷ 화면 속 폭
    return focal_length_px * real_w_m / box_w_px


def fit_ttc_s(samples: list[tuple[float, float]], t_now_s: float) -> float | None:
    """(촬영 시각, 1/박스 폭) 기록에 직선을 맞춰 TTC(s)를 구한다.

    사용처: 4단계. LeadRiskTracker.update()에서 매 프레임.
    입력: samples [(t_s, 1/박스 폭), ...] 최근 기록, t_now_s 지금 촬영 시각(s)
    출력: TTC(s). 가까워지고 있지 않으면(직선 기울기 0 이상) None
    주의: 시각을 "지금 기준(t - t_now)"으로 바꿔 맞추면 직선의 절편 a가 지금 값이 된다.
          그러면 0이 되는 시점까지 남은 시간이 바로 -a/b (b = 기울기, 가까워지면 음수).
    """
    t = np.array([s[0] - t_now_s for s in samples])
    inv_w = np.array([s[1] for s in samples])
    b, a = np.polyfit(t, inv_w, 1)  # inv_w ≈ a + b·t
    if b >= 0:
        return None
    return float(-a / b)


class LeadRiskTracker:
    """앞차의 거리와 TTC를 매 프레임 구한다 (TTC를 위해 같은 앞차의 박스 폭 기록을 시간 기준으로 보관).

    사용처: 4단계. 매 프레임 FrontCarTracker.update() 다음에 update()를 호출한다.
    주의: 클래스로 만든 이유: TTC는 최근 ttc_window_s초의 박스 폭 기록이 필요해서 프레임 사이에 기억해야 한다.
          앞차(track_id)가 바뀌거나 사라지면 기록을 지운다 (다른 차의 폭과 섞이면 TTC가 엉뚱하게 나옴).
          시간은 촬영 시각(t_s) 기준이다 (프레임 수가 아님).
    """

    def __init__(self, cfg_risk: dict) -> None:
        """설정값을 읽고 기록을 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_risk config의 risk 섹션 (hfov_deg, real_width_m, ttc_window_s, ttc_min_samples, ttc_max_s)
        """
        self.cfg = cfg_risk
        self.front_id = None  # 지금 기록 중인 앞차 track_id
        self.samples = deque()  # (t_s, 1/박스 폭 px) 최근 기록

    def update(self, front: dict | None, img_w_px: int, t_s: float) -> dict:
        """이번 프레임 앞차의 거리(m)와 TTC(s)를 돌려준다.

        사용처: 4단계. 매 프레임.
        입력: front 앞차 dict(FrontCarTracker 결과, 없으면 None. "track_id", "cls_name", "box_xyxy_px" 사용),
              img_w_px 프레임 가로(px), t_s 촬영 시각(s)
        출력: {"distance_m": float|None, "ttc_s": float|None} (schemas.LeadRisk)
              distance_m: 앞차가 없으면 None
              ttc_s: 앞차가 없거나, 기록이 ttc_min_samples개보다 적거나, 가까워지지 않거나, ttc_max_s보다 크면 None
        """
        cfg = self.cfg
        front_id = None if front is None else front["track_id"]
        if front_id != self.front_id:
            self.front_id = front_id
            self.samples.clear()
        if front is None:
            return {"distance_m": None, "ttc_s": None}

        x1, _, x2, _ = front["box_xyxy_px"]
        box_w_px = x2 - x1
        real_w_m = cfg["real_width_m"].get(front["cls_name"], cfg["real_width_m"]["car"])
        distance_m = estimate_distance_m(box_w_px, focal_px(img_w_px, cfg["hfov_deg"]), real_w_m)

        # 오래된 기록 버리기 (시간 기준)
        self.samples.append((t_s, 1.0 / box_w_px))
        while t_s - self.samples[0][0] > cfg["ttc_window_s"]:
            self.samples.popleft()

        ttc_s = None
        if len(self.samples) >= cfg["ttc_min_samples"]:
            ttc_s = fit_ttc_s(list(self.samples), t_s)
            if ttc_s is not None and ttc_s > cfg["ttc_max_s"]:
                ttc_s = None  # 너무 먼 미래는 "가까워지지 않음"과 같게 본다 (기울기가 0에 가까우면 TTC가 수백 초로 튐)
        return {"distance_m": distance_m, "ttc_s": ttc_s}
