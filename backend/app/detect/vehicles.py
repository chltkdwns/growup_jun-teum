"""1단계(탐지): 프레임에서 차·이륜차를 YOLO(COCO 사전학습)로 찾는다.

흐름: 보닛 아래를 잘라낸 프레임 → YOLO 추적(또는 탐지) → 차량 클래스만, 확신도 기준 이상만 → dict 목록
출력 형식은 임시다 (단계 사이 형식은 세 명 합의 후 스키마 파일로 옮긴다).
"""
from pathlib import Path

import numpy as np

# COCO 데이터셋의 클래스 번호 (사전학습 YOLO가 쓰는 고정 값). SPEC: 차량과 motorcycle만 쓴다
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_detector(cfg_detect: dict):
    """config의 가중치 경로로 YOLO 모델을 불러온다.

    사용처: 1단계(탐지). 영상(또는 카메라) 하나를 시작할 때 한 번.
    입력: cfg_detect config의 detect 섹션 (weights_path 가중치 파일 경로, backend 폴더 기준)
    출력: ultralytics YOLO 모델
    주의: 파일이 없으면 자동 다운로드하지 않고 멈춘다 (가중치는 직접 받아 weights/ 에 둔다, 커밋 금지).
    """
    # 무거운 라이브러리라 실제로 쓸 때만 불러온다 (가중치 없는 테스트가 빨리 돌게)
    from ultralytics import YOLO

    path = Path(cfg_detect["weights_path"])
    if not path.is_file():
        raise FileNotFoundError(f"YOLO 가중치가 없음: {path} (가중치 파일을 받아 이 경로에 두세요)")
    return YOLO(str(path))


def to_detections(
    xyxy_px: np.ndarray, cls_ids: np.ndarray, scores: np.ndarray, track_ids: list | None, t_s: float,
    score_min: float,
) -> list[dict]:
    """YOLO 결과 배열을 차량 탐지 dict 목록으로 바꾸고, 차량이 아니거나 확신도가 낮은 것은 버린다.

    사용처: 1단계(탐지). detect_vehicles() 안에서 매 프레임. (모델 없이 테스트할 수 있게 분리)
    입력: xyxy_px (N, 4) 박스 좌표(px), cls_ids (N,) COCO 클래스 번호, scores (N,) 확신도(0~1),
          track_ids 추적 번호 목록(N개) 또는 None(추적 안 함), t_s 촬영 시각(s), score_min 확신도 하한
    출력: [{"track_id": int|None, "cls_name": "car"|"motorcycle"|"bus"|"truck", "score": float,
           "box_xyxy_px": (x1, y1, x2, y2) float, "t_s": float}, ...]
    """
    dets = []
    for i in range(len(cls_ids)):
        cls_name = COCO_VEHICLE_CLASSES.get(int(cls_ids[i]))
        if cls_name is None or scores[i] < score_min:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy_px[i])
        dets.append({
            "track_id": None if track_ids is None else int(track_ids[i]),
            "cls_name": cls_name,
            "score": float(scores[i]),
            "box_xyxy_px": (x1, y1, x2, y2),
            "t_s": t_s,
        })
    return dets


def detect_vehicles(frame_bgr: np.ndarray, model, cfg_camera: dict, cfg_detect: dict, t_s: float) -> list[dict]:
    """한 프레임에서 차·이륜차를 찾는다 (추적을 켜면 같은 차에 같은 track_id).

    사용처: 1단계(탐지). 매 프레임 호출하고, 결과로 앞차 선정·거리·TTC를 한다.
    입력: frame_bgr 원본 프레임(BGR), model load_detector() 결과, cfg_camera(bonnet_ratio),
          cfg_detect(score_min, iou, agnostic_nms, imgsz, device, track, tracker), t_s 촬영 시각(s)
    출력: to_detections() 형식의 dict 목록. 좌표는 원본 프레임 기준(px)
    주의: SPEC대로 보닛 아래(bonnet_ratio)를 잘라낸 뒤 탐지한다. 위쪽만 잘라서 좌표는 그대로다.
          보닛에 걸친 가까운 차는 박스 아래가 잘리지만, 거리 계산에 쓰는 박스 폭은 그대로다.
          track=True면 프레임 순서대로 호출해야 한다 (ultralytics 추적기가 이전 프레임을 기억함).
    """
    bonnet_y_px = int(frame_bgr.shape[0] * cfg_camera["bonnet_ratio"])
    kwargs = {
        "classes": list(COCO_VEHICLE_CLASSES), "conf": cfg_detect["score_min"], "iou": cfg_detect["iou"],
        "agnostic_nms": cfg_detect["agnostic_nms"],
        "imgsz": cfg_detect["imgsz"], "device": cfg_detect["device"], "verbose": False,
    }
    crop = frame_bgr[:bonnet_y_px]
    if cfg_detect["track"]:
        result = model.track(crop, persist=True, tracker=cfg_detect["tracker"], **kwargs)[0]
    else:
        result = model.predict(crop, **kwargs)[0]
    boxes = result.boxes
    track_ids = None if boxes.id is None else boxes.id.int().tolist()
    return to_detections(
        boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), track_ids, t_s,
        cfg_detect["score_min"],
    )
