"""단계 사이에 주고받는 데이터 형식. 이름·단위를 바꿀 때는 세 명 모두에게 먼저 알린다 (CLAUDE.md 5절).

2~3단계 출력(차선 치우침·브레이크등·끼어들기)과 4단계 거리·TTC는 2026-11-17 세 명 합의 내용이다.
4단계(경고 판단) → 5단계(기록) 값은 docs/SPEC.md "아찔한 순간 기록" 초안을 따랐다.
각 모듈은 dict를 돌려주므로 TypedDict로 키 이름과 단위만 정한다 (모듈 코드는 그대로 dict를 쓴다).
"""

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

import numpy as np

# 경고 단계 (화면·판단·기록이 같은 이름을 쓴다): 안전 / 주의 / 경고 / 위험
WarningLevel = Literal["safe", "caution", "warning", "danger"]
# 무엇 때문인지: 앞차 급접근 / 끼어들기 / 브레이크등 / 차선 이탈 / 뒷차 / 이륜차
NearMissType = Literal[
    "lead", "cut_in", "brake", "lane_departure", "rear", "two_wheeler"
]


class LaneOffset(TypedDict):
    """2단계(차선, 장혜민 lane/offset.py) → 4단계. 내 차가 차선 중앙에서 벗어난 정도.

    차선 이탈 경고(lane_departure)는 내지 않고 값만 넘긴다. 이탈 판단은 경고 판단(성아영)이 이 값으로 한다.
    """

    # 차로 폭 절반 대비 비율 (단위 없음). 0 = 중앙, ±1 = 차선 위, ±1 초과 = 선을 넘음, 음수 = 왼쪽. 차선을 모르면 None
    lane_offset_ratio: float | None
    lane_status: Literal["ok", "estimated", "unavailable"]


class BrakeState(TypedDict):
    """3단계(브레이크등, 성아영 signal/brake.py BrakeStateTracker) → 4단계. 앞차 브레이크등 상태."""

    brake_on: bool | None  # 최종 판단 (최근 vote_window_s초 다수결). 앞차가 없거나 못 찾으면 None(모름)
    brake_score: float | None  # 모델 확신도 (0~1)
    model_on: bool  # 모델 단독 판단
    third_on: bool  # 보조제동등(가운데 위 등) 켜짐 → 미등 걸러내기 근거
    lamp_rose: bool  # 좌우 등이 평소보다 밝아짐 → 미등 걸러내기 근거


class CutIn(TypedDict):
    """2단계(끼어들기, 장혜민 lane/cut_in.py CutInDetector) → 4단계. 끼어들기가 확정된 순간 1건."""

    track_id: int  # 끼어든 차의 추적 ID
    side: Literal["left", "right"]  # 들어온 쪽
    t_s: float  # 확정 시각 (초, 촬영 시각 기준)


class LeadRisk(TypedDict):
    """4단계(거리·TTC, 성아영). 앞차까지의 거리와 충돌까지 남은 시간."""

    distance_m: float | None  # 앞차까지 거리 (m, 핀홀 공식). 앞차가 없으면 None
    ttc_s: float | None  # TTC(충돌까지 남은 시간, 초). 가까워지지 않으면 None


@dataclass(frozen=True)
class NearMissWarning:
    """경고 판단이 한 프레임에서 낸 경고 1개.

    기록 쪽은 level이 warning·danger인 것만 저장한다 (safe·caution은 넘겨도 무시됨).
    """

    t_s: float  # 촬영 시각 (초, 주행 또는 영상 시작 기준. 프레임 수가 아님)
    type: NearMissType
    level: WarningLevel
    ttc_s: float | None = None  # 그 순간 TTC (초)
    distance_m: float | None = None  # 대상까지 거리 (m, 핀홀)
    lateral_m: float | None = None  # 옆 거리 (m, 끼어들기·이륜차)
    # 대상 종류 (YOLO 클래스 이름 그대로: car, truck, bus, motorcycle. 이륜차도 motorcycle)
    target_class: str | None = None


class WarningSource(Protocol):
    """프레임 1장을 받아 경고 목록을 내는 것 (경고 판단이 이 모양을 따르면 실시간 기록에 바로 꽂힌다).

    사용처: 4단계(경고 판단). 실시간 WebSocket이 연결마다 하나 만들고 매 프레임 부른다.
    입력: frame BGR 이미지(H×W×3), t_s 촬영 시각(초)
    출력: 이 프레임의 경고 목록 (없으면 빈 목록)
    """

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]: ...
