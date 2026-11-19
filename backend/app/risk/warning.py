"""4단계(거리·위험): 거리·TTC·브레이크등·끼어들기·차선 치우침을 모아 경고 단계(안전/주의/경고/위험)를 정한다.

규칙(docs/SPEC.md "경고 단계"):
  기본 단계: TTC 3초 이하 경고, 1.5초 이하 위험, 안전거리(내 속도 × 2초)보다 가까우면 주의
  상향: 브레이크등 켜짐 + 가까워지는 중이면 한 단계, 끼어들기 확정이면 한 단계 (위험이 최고)
  깜빡임 방지: 올릴 때는 즉시, 내릴 때는 level_down_hold_s초 기다렸다가
  추가 경고: 차선 이탈 (차로 변경 직후 잠깐은 판단 안 함)
출력은 schemas.NearMissWarning. 시간은 모두 촬영 시각(t_s) 기준이다. FastAPI·DB를 쓰지 않는다.
"""
from app.schemas import NearMissType, NearMissWarning

# 단계 순서: 인덱스가 높을수록 위험하다 (schemas.WarningLevel 이름과 같음)
LEVELS = ("safe", "caution", "warning", "danger")
SAFE, CAUTION, WARNING, DANGER = range(len(LEVELS))
KMH_TO_MS = 1 / 3.6


def level_from_ttc(ttc_s: float | None, cfg_risk: dict) -> int:
    """TTC(s)로 기본 단계를 구한다.

    사용처: 4단계. raw_level()에서 매 프레임.
    입력: ttc_s TTC(초, 없으면 None), cfg_risk config의 risk 섹션 (ttc_warning_s, ttc_danger_s)
    출력: 단계 번호 (SAFE·WARNING·DANGER). TTC가 없으면 SAFE
    주의: 기준값과 같으면 높은 쪽이다 (정확히 3.0초는 경고, 정확히 1.5초는 위험).
    """
    if ttc_s is None:
        return SAFE
    if ttc_s <= cfg_risk["ttc_danger_s"]:
        return DANGER
    if ttc_s <= cfg_risk["ttc_warning_s"]:
        return WARNING
    return SAFE


def is_headway_short(distance_m: float | None, speed_kmh: float | None, cfg_risk: dict) -> bool:
    """앞차까지 거리가 안전거리(내 속도 × safe_time_s)보다 가까운지 본다.

    사용처: 4단계. raw_level()에서 매 프레임.
    입력: distance_m 앞차까지 거리(m, 없으면 None), speed_kmh 내 속도(km/h, 모르면 None), cfg_risk (safe_time_s)
    출력: 가까우면 True
    주의: 내 속도나 거리를 모르면 False다 (SPEC: 속도를 모르면 안전거리 기준은 건너뜀).
          안전거리와 정확히 같으면 부족하지 않은 것으로 본다.
    """
    if distance_m is None or speed_kmh is None:
        return False
    return distance_m < speed_kmh * KMH_TO_MS * cfg_risk["safe_time_s"]


def raw_level(
    ttc_s: float | None,
    distance_m: float | None,
    speed_kmh: float | None,
    brake_on: bool | None,
    cut_in_now: bool,
    cfg_risk: dict,
) -> tuple[int, NearMissType]:
    """이번 프레임 값만으로 앞차 관련 단계와 그 이유(경고 종류)를 구한다 (내림 대기 적용 전).

    사용처: 4단계. WarningJudge.update()에서 매 프레임.
    입력: ttc_s TTC(초), distance_m 앞차 거리(m), speed_kmh 내 속도(km/h, 모르면 None),
          brake_on 앞차 브레이크등 켜짐(모르면 None), cut_in_now 이번 프레임에 끼어들기가 확정됐는지, cfg_risk risk 섹션
    출력: (단계 번호, 종류) 종류는 끼어들기로 올랐으면 "cut_in", 브레이크등 때문이면 "brake", 나머지는 "lead"
    주의: "가까워지는 중"은 TTC가 나오는 것으로 본다 (TTC는 1/박스 폭 직선 기울기가 가까워질 때만 나옴).
          상향은 기본 단계에 한 단계씩 더하고 위험에서 멈춘다.
    """
    level = level_from_ttc(ttc_s, cfg_risk)
    if level == SAFE and is_headway_short(distance_m, speed_kmh, cfg_risk):
        level = CAUTION
    kind: NearMissType = "lead"
    if brake_on and ttc_s is not None:
        level = min(level + 1, DANGER)
        kind = "brake"
    if cut_in_now:
        level = min(level + 1, DANGER)
        kind = "cut_in"
    return level, kind


class WarningJudge:
    """프레임마다 경고 목록을 만든다. 단계를 내릴 때 기다리고, 차로 변경 직후 이탈 판단을 건너뛰기 위해 상태를 기억한다.

    사용처: 4단계. 매 프레임 LeadRiskTracker.update() 다음에 update()를 호출한다 (app/risk/warning_source.py).
    주의: 클래스로 만든 이유: 지금 단계, 낮은 상태가 시작된 시각, 마지막 차로 변경 시각을 프레임 사이에 기억해야 해서.
          영상(카메라) 하나에 하나씩 만든다.
          경고가 이어지는 동안 매 프레임 경고를 낸다 (기록 쪽이 같은 종류를 한 건으로 합치고 클립 끝을 늘림).
    """

    def __init__(self, cfg_risk: dict) -> None:
        """설정값을 읽고 상태를 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_risk config의 risk 섹션 (ttc_warning_s, ttc_danger_s, safe_time_s, level_down_hold_s,
              lane_departure_ratio, lane_departure_skip_after_change_s, lane_departure_level)
        """
        self.cfg = cfg_risk
        self.level = SAFE  # 지금 유지 중인 단계
        self.kind: NearMissType = "lead"  # 지금 단계가 생긴 이유
        self.below_since_s: float | None = None  # 이번 프레임 값이 지금 단계보다 낮아지기 시작한 촬영 시각
        self.last_lane_change_s: float | None = None  # 마지막으로 차로 변경을 감지한 촬영 시각

    def update(
        self,
        lead: dict,
        brake: dict | None,
        cut_in: dict | None,
        offset: dict,
        lane_change: str | None,
        speed_kmh: float | None,
        t_s: float,
        target_class: str | None,
    ) -> list[NearMissWarning]:
        """이번 프레임의 경고 목록을 돌려준다.

        사용처: 4단계. 매 프레임.
        입력: lead schemas.LeadRisk (distance_m, ttc_s), brake schemas.BrakeState 또는 None(브레이크등 모델 없음),
              cut_in schemas.CutIn 또는 None (끼어들기가 확정된 프레임에만 값이 있음),
              offset schemas.LaneOffset (lane_offset_ratio), lane_change "left"|"right"|None (차로 변경 감지 프레임),
              speed_kmh 내 속도(km/h, 모르면 None), t_s 촬영 시각(s), target_class 앞차 종류(YOLO 클래스 이름, 없으면 None)
        출력: NearMissWarning 목록. 안전이고 이탈도 아니면 빈 목록
        주의: 앞차 경고(lead·brake·cut_in 중 하나)와 차선 이탈은 따로 나온다 (기록에서 종류별로 합쳐지므로).
              단계가 주의(caution)인 경고도 목록에 들어간다 (화면 표시용. 기록은 경고 이상만 저장함).
        """
        cfg = self.cfg
        raw, kind = raw_level(
            lead["ttc_s"], lead["distance_m"], speed_kmh, None if brake is None else brake["brake_on"], cut_in is not None, cfg
        )
        if raw >= self.level:
            # 올리거나 같으면 즉시 반영하고 이유도 갱신
            self.level, self.kind, self.below_since_s = raw, kind, None
        else:
            # 낮아졌으면 level_down_hold_s초 이어질 때만 내린다 (경계 근처에서 단계가 깜빡이지 않게)
            if self.below_since_s is None:
                self.below_since_s = t_s
            if t_s - self.below_since_s >= cfg["level_down_hold_s"]:
                self.level, self.kind, self.below_since_s = raw, kind, None

        warnings: list[NearMissWarning] = []
        if self.level > SAFE:
            warnings.append(
                NearMissWarning(
                    t_s=t_s,
                    type=self.kind,
                    level=LEVELS[self.level],
                    ttc_s=lead["ttc_s"],
                    distance_m=lead["distance_m"],
                    target_class=target_class,
                )
            )
        departure = self._lane_departure(offset, lane_change, t_s)
        if departure is not None:
            warnings.append(departure)
        return warnings

    def _lane_departure(self, offset: dict, lane_change: str | None, t_s: float) -> NearMissWarning | None:
        """차선 이탈 경고를 구한다 (차로 변경 직후 lane_departure_skip_after_change_s초는 건너뜀).

        사용처: 4단계. update()에서 매 프레임.
        입력: offset schemas.LaneOffset, lane_change 차로 변경 감지("left"|"right"|None), t_s 촬영 시각(s)
        출력: 이탈이면 NearMissWarning(type="lane_departure"), 아니면 None
        주의: 차선을 모르면(lane_offset_ratio가 None) 판단하지 않는다.
              의도한 차로 변경과 졸음 이탈은 구분하지 못한다 (방향지시등 인식은 2차 기능).
        """
        cfg = self.cfg
        if lane_change is not None:
            self.last_lane_change_s = t_s
        ratio = offset["lane_offset_ratio"]
        if ratio is None:
            return None
        if self.last_lane_change_s is not None and t_s - self.last_lane_change_s < cfg["lane_departure_skip_after_change_s"]:
            return None
        if abs(ratio) < cfg["lane_departure_ratio"]:
            return None
        return NearMissWarning(t_s=t_s, type="lane_departure", level=cfg["lane_departure_level"])
