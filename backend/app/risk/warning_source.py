"""분석(PerceptionPipeline) 결과로 진짜 경고를 내는 WarningSource (schemas.WarningSource 모양: 프레임 → 경고 목록).

fake_warning.PipelineFakeWarningSource 자리에 그대로 꽂는다. main.py가 이것을 만들면 가짜 경고는 필요 없어진다.
흐름: pipeline.process() → LeadRiskTracker(거리·TTC) → WarningJudge(경고 단계)
"""
import numpy as np

from app.risk.lead import LeadRiskTracker
from app.risk.warning import WarningJudge
from app.schemas import NearMissWarning


class PipelineWarningSource:
    """프레임 1장을 분석하고 경고 목록을 돌려준다 (영상 하나 또는 실시간 연결 하나에 하나씩 만든다).

    사용처: 4단계(경고 판단) 자리. 영상 업로드 분석 스레드와 /live WebSocket이 매 프레임 호출한다.
    주의: 클래스로 만든 이유: 거리·TTC 기록과 경고 단계 상태를 프레임 사이에 기억해야 해서.
          프레임은 촬영 순서대로 넣어야 한다. speed_kmh를 모르면 None으로 둔다 (안전거리 판단만 건너뜀).
    """

    def __init__(self, pipeline, cfg_risk: dict, speed_kmh: float | None = None) -> None:
        """분석 파이프라인과 거리·경고 판단기를 연결한다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: pipeline app.pipeline.PerceptionPipeline (영상 하나에 하나), cfg_risk config의 risk 섹션,
              speed_kmh 내 속도(km/h, 모르면 None. GPS·수동 입력은 아직 연결 전)
        """
        self._pipeline = pipeline
        self._lead = LeadRiskTracker(cfg_risk)
        self._judge = WarningJudge(cfg_risk)
        self.speed_kmh = speed_kmh

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]:
        """프레임을 분석하고 이번 프레임의 경고 목록을 돌려준다.

        사용처: 영상 분석 루프(upload_analyzer.analyze_video)와 /live 수신 루프에서 매 프레임.
        입력: frame BGR 이미지(H×W×3), t_s 촬영 시각(초)
        출력: NearMissWarning 목록 (없으면 빈 목록). 기록 쪽은 경고 이상만 저장한다
        """
        result = self._pipeline.process(frame, t_s)
        front = result["front"]
        lead = self._lead.update(front, frame.shape[1], t_s)
        return self._judge.update(
            lead,
            result["brake"],
            result["cut_in"],
            result["offset"],
            result["lane_change"],
            self.speed_kmh,
            t_s,
            None if front is None else front["cls_name"],
        )
