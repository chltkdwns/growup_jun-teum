"""경고 판단(성아영)이 아직 없어서 쓰는 가짜 경고. 실시간 클립 저장을 시험하는 용도다.

진짜 경고 판단이 schemas.WarningSource 모양(frame, t_s → 경고 목록)으로 나오면
main.py의 /live와 make_upload_judge()에서 이 클래스 대신 그것을 만들면 된다. 그때 이 파일은 지운다.
"""

import numpy as np

from app.schemas import NearMissWarning

# 가짜 경고의 내용 (tools/try_clip.py와 같은 값: 앞차 급접근, 위험 단계)
FAKE_TYPE = "lead"
FAKE_LEVEL = "danger"
FAKE_TTC_S = 1.3


class FakeWarningSource:
    """정해 둔 시각마다 가짜 경고를 한 번씩 낸다.

    사용처: 4단계(경고 판단) 자리. /live WebSocket이 연결마다 하나 만든다.
    주의: 프레임 내용은 보지 않는다. 시각은 주행 시작 기준 촬영 시각(초)이다.
    """

    def __init__(self, at_s: list[float]) -> None:
        self._pending = sorted(at_s)  # 아직 안 낸 경고 시각

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]:
        """이번 프레임에서 낼 가짜 경고를 돌려준다.

        사용처: /live 수신 루프에서 매 프레임.
        입력: frame BGR 이미지(쓰지 않음), t_s 촬영 시각(초)
        출력: 정해 둔 시각을 막 지났으면 경고 1개, 아니면 빈 목록
        주의: 프레임 간격이 불규칙해서 '같은 시각'이 아니라 '그 시각을 처음 넘은 프레임'에서 낸다.
        """
        warnings = []
        while self._pending and t_s >= self._pending[0]:
            self._pending.pop(0)
            warnings.append(
                NearMissWarning(
                    t_s=t_s, type=FAKE_TYPE, level=FAKE_LEVEL, ttc_s=FAKE_TTC_S
                )
            )
        return warnings


class PipelineFakeWarningSource:
    """분석(PerceptionPipeline)은 진짜로 돌리고, 경고만 가짜로 낸다. 영상 업로드 흐름을 시험하는 용도다.

    사용처: 4단계(경고 판단) 자리. POST /drives/upload의 분석 스레드가 영상마다 하나 만든다.
    주의: 분석 결과는 아직 쓰지 않는다 (TTC·경고 단계는 성아영 담당). 그래도 돌리는 이유는
          올린 영상(예: FINEVU AVI)에서 탐지·차선이 실제로 도는지, 분석 시간이 얼마나 걸리는지 보려는 것.
    """

    def __init__(self, pipeline, at_s: list[float]) -> None:
        self._pipeline = pipeline  # app.pipeline.PerceptionPipeline (영상 하나에 하나)
        self._fake = FakeWarningSource(at_s)

    def __call__(self, frame: np.ndarray, t_s: float) -> list[NearMissWarning]:
        """프레임을 분석하고, 정해 둔 시각이면 가짜 경고를 돌려준다.

        사용처: 영상 분석 루프(upload_analyzer.analyze_video)에서 매 프레임.
        입력: frame BGR 이미지, t_s 영상 속 시각(초)
        출력: FakeWarningSource와 같음
        """
        self._pipeline.process(frame, t_s)
        return self._fake(frame, t_s)
