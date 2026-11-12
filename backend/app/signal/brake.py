"""3단계(신호 읽기): 앞차의 브레이크등이 켜졌는지 판단한다.

흐름: 브레이크등 YOLO(차 박스 + 켜짐/꺼짐 2클래스)를 전체 프레임에 돌림 → 앞차 박스와 가장 많이 겹치는 결과를 고름
      → 미등 걸러내기(보조 제동등 빨강, 좌우 등이 평소보다 밝아졌는지) → 최근 짧은 시간 다수결
모델: Roboflow "State of vehicle tail lamp detection"(국민대, CC BY 4.0)으로 Colab에서 학습한 YOLOv8n.
      클래스는 car_BrakeOff / car_BrakeOn 이고, 박스는 등이 아니라 차 전체에 쳐져 있다.
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np

# 브레이크등 모델의 "켜짐" 클래스 이름 (학습 데이터에서 정해진 고정 값)
BRAKE_ON_CLASS = "car_BrakeOn"
# OpenCV HSV 색상(H, 0~180)에서 빨강은 0 근처와 180 근처 두 군데에 있다 (색 정의라 바뀌지 않음)
RED_H_LOW = (0, 10)
RED_H_HIGH = (165, 180)


def load_brake_model(cfg_brake: dict):
    """config의 가중치 경로로 브레이크등 YOLO 모델을 불러온다.

    사용처: 3단계(신호 읽기). 영상(또는 카메라) 하나를 시작할 때 한 번.
    입력: cfg_brake config의 brake 섹션 (weights_path 가중치 파일 경로, backend 폴더 기준)
    출력: ultralytics YOLO 모델
    주의: 파일이 없으면 멈춘다 (Colab에서 학습한 best.pt를 weights/ 에 직접 둔다, 커밋 금지).
    """
    # 무거운 라이브러리라 실제로 쓸 때만 불러온다 (가중치 없는 테스트가 빨리 돌게)
    from ultralytics import YOLO

    path = Path(cfg_brake["weights_path"])
    if not path.is_file():
        raise FileNotFoundError(f"브레이크등 가중치가 없음: {path} (Colab에서 받은 best.pt를 이 경로에 두세요)")
    return YOLO(str(path))


def to_brake_boxes(xyxy_px: np.ndarray, cls_ids: np.ndarray, scores: np.ndarray, names: dict) -> list[dict]:
    """브레이크등 YOLO 결과 배열을 dict 목록으로 바꾼다.

    사용처: 3단계. detect_brake_boxes() 안에서 매 프레임. (모델 없이 테스트할 수 있게 분리)
    입력: xyxy_px (N, 4) 차 박스(px), cls_ids (N,) 클래스 번호, scores (N,) 확신도(0~1), names 번호 → 클래스 이름
    출력: [{"box_xyxy_px": (x1, y1, x2, y2) float, "model_on": bool, "score": float}, ...]
    """
    out = []
    for i in range(len(cls_ids)):
        x1, y1, x2, y2 = (float(v) for v in xyxy_px[i])
        out.append(
            {
                "box_xyxy_px": (x1, y1, x2, y2),
                "model_on": names[int(cls_ids[i])] == BRAKE_ON_CLASS,
                "score": float(scores[i]),
            }
        )
    return out


def detect_brake_boxes(frame_bgr: np.ndarray, model, cfg_brake: dict) -> list[dict]:
    """한 프레임 전체에서 차마다 브레이크등 켜짐/꺼짐을 찾는다.

    사용처: 3단계. 매 프레임 앞차를 정한 뒤(select_front_car → FrontCarTracker) 호출한다.
    입력: frame_bgr 원본 프레임(BGR), model load_brake_model() 결과, cfg_brake(score_min, imgsz, device)
    출력: to_brake_boxes() 형식의 dict 목록. 좌표는 원본 프레임 기준(px)
    주의: 앞차만 잘라 넣지 않고 전체 프레임에 돌린다. 모델이 전체 화면 속 차를 보고 학습했기 때문에
          잘라 넣으면 차 크기가 학습 때와 달라져 판단이 흔들린다.
    """
    result = model.predict(
        frame_bgr, conf=cfg_brake["score_min"], imgsz=cfg_brake["imgsz"], device=cfg_brake["device"], verbose=False
    )[0]
    boxes = result.boxes
    return to_brake_boxes(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), result.names)


def box_iou(a: tuple, b: tuple) -> float:
    """두 박스가 얼마나 겹치는지(겹친 넓이 ÷ 합친 넓이, 0~1) 구한다.

    사용처: match_front_brake()에서 앞차 박스와 브레이크등 모델 박스를 짝지을 때.
    입력: a, b (x1, y1, x2, y2) px
    출력: 0~1 (1 = 완전히 같은 박스)
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_front_brake(brake_boxes: list[dict], front_box: tuple, iou_min: float) -> dict | None:
    """브레이크등 모델 박스 중 앞차 박스와 가장 많이 겹치는 것 하나를 고른다.

    사용처: 3단계. detect_brake_boxes() 다음에 매 프레임.
    입력: brake_boxes detect_brake_boxes() 결과, front_box 앞차 박스 (x1, y1, x2, y2) px, iou_min 같은 차로 볼 최소 겹침
    출력: 고른 dict 또는 겹치는 박스가 없으면 None
    주의: 차량 탐지 YOLO는 보닛 아래를 잘라 탐지하므로, 아주 가까운 차는 두 박스의 아래쪽이 다를 수 있다.
    """
    best, best_iou = None, iou_min
    for b in brake_boxes:
        iou = box_iou(b["box_xyxy_px"], front_box)
        if iou >= best_iou:
            best, best_iou = b, iou
    return best


def red_ratio(region_bgr: np.ndarray, cfg_brake: dict) -> float:
    """영역 안에서 밝고 진한 빨강 픽셀의 비율을 구한다 (등이 켜질수록 커짐).

    사용처: lamp_signals()에서 보조 제동등·좌우 등 영역마다.
    입력: region_bgr 잘라낸 영역(BGR), cfg_brake(red_s_min 최소 채도 0~255, red_v_min 최소 밝기 0~255)
    출력: 0~1 비율. 영역이 비어 있으면 0
    주의: 흰 차체·번호판 조명이 섞이지 않게 하양은 세지 않는다. 그래서 너무 밝아 하얗게 날아간 등은 덜 세어진다.
    """
    if region_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    s_min, v_min = cfg_brake["red_s_min"], cfg_brake["red_v_min"]
    mask = cv2.inRange(hsv, (RED_H_LOW[0], s_min, v_min), (RED_H_LOW[1], 255, 255)) | cv2.inRange(
        hsv, (RED_H_HIGH[0], s_min, v_min), (RED_H_HIGH[1], 255, 255)
    )
    return float(np.count_nonzero(mask)) / mask.size


def lamp_signals(frame_bgr: np.ndarray, box: tuple, cfg_brake: dict) -> dict:
    """차 박스에서 보조 제동등 영역과 좌우 등 영역의 빨강 비율을 구한다.

    사용처: BrakeStateTracker.update()에서 앞차에 대해 매 프레임.
    입력: frame_bgr 원본 프레임, box 차 박스 (x1, y1, x2, y2) px,
          cfg_brake(third_lamp_region_ratio 보조 제동등 영역 [x1, y1, x2, y2] 박스 대비,
          side_lamp_y_ratio 좌우 등 세로 범위 [위, 아래] 박스 대비, side_lamp_w_ratio 좌우 등 가로 폭 박스 대비)
    출력: {"third_red": 0~1, "side_red": 0~1 (좌우 평균)}
    주의: 정면 뒷모습 기준의 대략 위치다. 비스듬한 차나 트럭(보조 제동등 없음)은 맞지 않을 수 있다.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[2]), w), min(int(box[3]), h)
    bw, bh = x2 - x1, y2 - y1

    def cut(fx1: float, fy1: float, fx2: float, fy2: float) -> np.ndarray:
        return frame_bgr[y1 + int(bh * fy1) : y1 + int(bh * fy2), x1 + int(bw * fx1) : x1 + int(bw * fx2)]

    tx1, ty1, tx2, ty2 = cfg_brake["third_lamp_region_ratio"]
    sy1, sy2 = cfg_brake["side_lamp_y_ratio"]
    sw = cfg_brake["side_lamp_w_ratio"]
    side = (red_ratio(cut(0.0, sy1, sw, sy2), cfg_brake) + red_ratio(cut(1.0 - sw, sy1, 1.0, sy2), cfg_brake)) / 2
    return {"third_red": red_ratio(cut(tx1, ty1, tx2, ty2), cfg_brake), "side_red": side}


class BrakeStateTracker:
    """앞차 브레이크등 상태를 프레임 사이 기록으로 안정시킨다 (미등 걸러내기 + 시간 기준 다수결).

    사용처: 3단계. 매 프레임 match_front_brake() 다음에 update()를 호출한다.
    주의: 클래스로 만든 이유: "평소(브레이크 안 밟을 때) 좌우 등 밝기"와 "최근 판단들"을 프레임 사이에 기억해야 해서.
          시간은 촬영 시각(t_s) 기준이다. 앞차(track_id)가 바뀌면 기록을 지운다.
          미등 걸러내기: 야간에는 미등만 켜져도 모델이 켜짐으로 잡는 경우가 많다(개발 데모 54장 중 4장).
          그래서 모델이 켜짐이라도 ① 보조 제동등 영역에 빨강이 있거나 ② 좌우 등이 평소보다 확 밝아졌을 때만 켜짐으로 본다.
          미등은 보조 제동등이 안 켜지고, 밝기도 계속 같아서 걸러진다.
    """

    def __init__(self, cfg_brake: dict) -> None:
        """설정값을 읽고 기록을 비운다.

        사용처: 영상(또는 카메라) 하나를 시작할 때 한 번.
        입력: cfg_brake config의 brake 섹션
        """
        self.cfg = cfg_brake
        self.front_id = None  # 지금 기록 중인 앞차 track_id
        self.side_hist = deque()  # (t_s, side_red) 브레이크 근거가 없던 프레임의 좌우 등 빨강 비율
        self.votes = deque()  # (t_s, 켜짐 여부) 최근 판단

    def _reset(self, front_id: int | None) -> None:
        """앞차가 바뀌거나 사라지면 기록을 지운다 (다른 차의 밝기와 비교하면 안 되므로)."""
        self.front_id = front_id
        self.side_hist.clear()
        self.votes.clear()

    def update(self, front: dict | None, matched: dict | None, frame_bgr: np.ndarray, t_s: float) -> dict:
        """이번 프레임 앞차의 브레이크등 상태를 돌려준다.

        사용처: 3단계. 매 프레임 FrontCarTracker.update()와 match_front_brake() 다음에.
        입력: front 앞차 dict(FrontCarTracker 결과, 없으면 None), matched match_front_brake() 결과(없으면 None),
              frame_bgr 원본 프레임, t_s 촬영 시각(s)
        출력: {"brake_on": bool|None, "brake_score": float|None, "model_on": bool, "third_on": bool, "lamp_rose": bool}
              brake_on: 최근 vote_window_s초 다수결 결과. 앞차가 없거나 모델이 앞차를 못 찾으면 None(모름)
              model_on: 모델 단독 판단 (on_score_min 이상일 때만 켜짐), third_on·lamp_rose: 미등 걸러내기 근거
        주의: 영상 파일·실시간 모두 프레임 순서대로 호출해야 한다. 사진을 한 장씩 넣으면 ②(밝아짐)는 동작하지 않는다.
        """
        cfg = self.cfg
        front_id = None if front is None else front["track_id"]
        if front_id != self.front_id:
            self._reset(front_id)
        unknown = {"brake_on": None, "brake_score": None, "model_on": False, "third_on": False, "lamp_rose": False}
        if front is None or matched is None:
            self.votes.clear()
            return unknown

        model_on = matched["model_on"] and matched["score"] >= cfg["on_score_min"]
        sig = lamp_signals(frame_bgr, matched["box_xyxy_px"], cfg)
        third_on = sig["third_red"] >= cfg["third_red_min"]

        # 오래된 기록 버리기 (시간 기준)
        while self.side_hist and t_s - self.side_hist[0][0] > cfg["baseline_window_s"]:
            self.side_hist.popleft()
        lamp_rose = False
        if len(self.side_hist) >= cfg["baseline_min_count"]:
            base = float(np.median([v for _, v in self.side_hist]))
            lamp_rose = sig["side_red"] >= base * cfg["rise_ratio"] and sig["side_red"] - base >= cfg["rise_abs_min"]
        # "평소 밝기"는 밝기 근거가 없는 프레임으로만 쌓는다. 모델 판단은 보지 않는다:
        # 미등을 켜짐으로 잘못 잡는 프레임도 평소 밝기에 들어가야 미등을 걸러낼 수 있기 때문.
        if not third_on and not lamp_rose:
            self.side_hist.append((t_s, sig["side_red"]))

        decided = model_on and (third_on or lamp_rose)

        while self.votes and t_s - self.votes[0][0] > cfg["vote_window_s"]:
            self.votes.popleft()
        self.votes.append((t_s, decided))
        n_on = sum(on for _, on in self.votes)
        return {
            "brake_on": n_on * 2 > len(self.votes),
            "brake_score": matched["score"],
            "model_on": model_on,
            "third_on": third_on,
            "lamp_rose": lamp_rose,
        }
