"""한 프레임을 넣으면 탐지·차선·앞차·끼어들기 결과를 한 번에 돌려준다 (영상 파일·실시간 공용).

흐름: 1 탐지(YOLO) → 클래스 다수결 → 2 차선(검출 + 트래커) → 치우침 → 2 앞차(선택 + 유지) → 4 끼어들기
각 단계의 로직은 원래 모듈 그대로이고, 여기서는 호출 순서와 프레임 사이 상태(트래커들)만 묶는다.
출력 묶음 dict의 이름(키)은 임시다 (세 명 합의 후 schemas.py에 확정).
"""
import numpy as np

from app.detect.vehicles import ClassVoter, detect_vehicles, load_detector
from app.lane.cut_in import CutInDetector
from app.lane.detect import detect_lanes
from app.lane.front_car import FrontCarTracker, select_front_car
from app.lane.offset import lane_offset
from app.lane.tracker import LaneTracker


class PerceptionPipeline:
    """탐지부터 끼어들기까지 혜민 파트 분석을 프레임마다 순서대로 실행한다.

    사용처: 영상 분석 도구(tools/run_pipeline.py), 나중에 서버의 영상 분석·실시간 수신 루프. 영상(카메라) 하나에 하나씩 만든다.
    주의: 클래스로 만든 이유: 차선·앞차·끼어들기·클래스 다수결이 모두 프레임 사이 상태를 기억해서,
          그 상태들을 한곳에 보관해야 같은 영상의 프레임을 순서대로 넣을 수 있다.
          프레임은 촬영 순서대로 넣어야 한다 (YOLO 추적기와 트래커들이 이전 프레임을 기억함).
          새 영상을 시작할 때는 새 인스턴스를 만든다 (이전 영상 상태가 섞이지 않게).
    """

    def __init__(self, cfg: dict, model=None) -> None:
        """설정을 읽고 YOLO 모델과 트래커들을 준비한다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg config.yaml 전체 (camera, lane, front_car, cut_in, detect 섹션 사용),
              model 미리 불러온 YOLO 모델 (None이면 cfg의 weights_path로 불러옴. 테스트에서는 가짜 모델을 넣음)
        주의: 모델 로딩은 몇 초 걸리므로 여기서 한 번만 한다.
        """
        self.cfg = cfg
        self.model = model if model is not None else load_detector(cfg["detect"])
        self.voter = ClassVoter(cfg["detect"])
        self.lane_tracker = LaneTracker(cfg["lane"])
        self.front_tracker = FrontCarTracker(cfg["front_car"])
        self.cut_in_detector = CutInDetector(cfg["cut_in"])

    def process(self, frame_bgr: np.ndarray, t_s: float) -> dict:
        """프레임 한 장을 분석해 단계별 결과를 묶어 돌려준다.

        사용처: 매 프레임 호출한다.
        입력: frame_bgr 원본 프레임(BGR), t_s 촬영 시각(s, 늘어나기만 해야 함)
        출력: {"dets": Detection 목록, "lanes": TrackedLanes, "offset": LaneOffset, "lane_change": "left"|"right"|None,
               "front": FrontCar 또는 None, "cut_in": CutIn 또는 None} (각 형식은 app/schemas.py, 묶음 키 이름은 임시)
        주의: 순서가 중요하다. 앞차는 차선 결과가 있어야 고르고, 끼어들기는 앞차와 차선 결과가 있어야 판단한다.
        """
        cam, lane_cfg, front_cfg = self.cfg["camera"], self.cfg["lane"], self.cfg["front_car"]
        img_h, img_w = frame_bgr.shape[:2]
        # 1단계 탐지 → 같은 차 클래스 고정
        dets = self.voter.update(detect_vehicles(frame_bgr, self.model, cam, self.cfg["detect"], t_s), t_s)
        # 2단계 차선 → 치우침
        lanes = self.lane_tracker.update(detect_lanes(frame_bgr, cam, lane_cfg), t_s)
        offset = lane_offset(lanes, img_w, cam)
        # 2단계 앞차 (1등 후보 → 유지 규칙)
        candidate = select_front_car(dets, lanes, img_w, img_h, cam, lane_cfg, front_cfg)
        front = self.front_tracker.update(candidate, dets, t_s, lanes["lane_change"])
        # 4단계 끼어들기
        cut_in = self.cut_in_detector.update(dets, lanes, front, t_s, img_h, img_w, lanes["lane_change"])
        return {"dets": dets, "lanes": lanes, "offset": offset, "lane_change": lanes["lane_change"],
                "front": front, "cut_in": cut_in}
