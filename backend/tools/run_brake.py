"""영상에 앞차와 브레이크등 상태를 그려 저장한다 (눈으로 확인용).

실행 (backend 폴더에서):
    python -m tools.run_brake --video data/videos/night.mp4 --out outputs/brake_night.mp4

화면: 앞차 박스 색 = 브레이크등 상태 (빨강 BRAKE ON, 초록 BRAKE OFF, 주황 TAIL ONLY = 모델은 켜짐인데
      미등 걸러내기로 꺼짐 처리, 회색 = 모름), 다른 차는 얇은 회색 박스
"""
import argparse
import time
from pathlib import Path

import cv2
import yaml

from app.detect.vehicles import detect_vehicles, load_detector
from app.lane.detect import detect_lanes
from app.lane.front_car import FrontCarTracker, select_front_car
from app.lane.tracker import LaneTracker
from app.signal.brake import BrakeStateTracker, detect_brake_boxes, load_brake_model, match_front_brake

MS_PER_S = 1000
# 화면 색 (BGR)
COLOR_ON, COLOR_OFF, COLOR_TAIL, COLOR_UNKNOWN = (0, 0, 255), (0, 200, 0), (0, 165, 255), (160, 160, 160)


def draw_brake(frame, dets: list[dict], front: dict | None, brake: dict) -> None:
    """다른 차는 얇게, 앞차는 브레이크등 상태 색의 굵은 박스로 그린다.

    사용처: main()에서 매 프레임 (확인용 화면).
    입력: frame 그림을 그릴 프레임(직접 바뀜), dets detect_vehicles() 결과, front 앞차 dict(없으면 None),
          brake BrakeStateTracker.update() 결과
    출력: 없음
    """
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box_xyxy_px"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_UNKNOWN, 1)
    if front is None:
        return
    if brake["brake_on"] is None:
        color, text = COLOR_UNKNOWN, "BRAKE ?"
    elif brake["brake_on"]:
        color, text = COLOR_ON, f"BRAKE ON {brake['brake_score']:.2f}"
    elif brake["model_on"]:
        color, text = COLOR_TAIL, f"TAIL ONLY (model ON {brake['brake_score']:.2f})"
    else:
        color, text = COLOR_OFF, f"BRAKE OFF {brake['brake_score']:.2f}"
    x1, y1, x2, y2 = (int(v) for v in front["box_xyxy_px"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, text, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    """영상 → 프레임마다 차선·탐지·앞차·브레이크등 → 그린 영상 저장 + 요약 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --config 설정 파일
    출력: 결과 영상, 콘솔에 앞차 있는 프레임 중 켜짐·꺼짐·미등 처리·모름 비율과 처리 ms
    주의: 처리 ms는 브레이크등 모델 + 미등 걸러내기만 잰다 (차선·탐지·앞차는 run_front_car에서 잰다). 첫 프레임은 뺀다.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cam, lane_cfg, front_cfg, brake_cfg = cfg["camera"], cfg["lane"], cfg["front_car"], cfg["brake"]
    detector = load_detector(cfg["detect"])
    brake_model = load_brake_model(brake_cfg)
    lane_tracker = LaneTracker(lane_cfg)
    front_tracker = FrontCarTracker(front_cfg)
    brake_tracker = BrakeStateTracker(brake_cfg)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counts = {"on": 0, "off": 0, "tail": 0, "unknown": 0}
    n = 0
    proc_ms = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S
        lanes = lane_tracker.update(detect_lanes(frame, cam, lane_cfg), t_s)
        dets = detect_vehicles(frame, detector, cam, cfg["detect"], t_s)
        candidate = select_front_car(dets, lanes, w, h, cam, lane_cfg, front_cfg)
        front = front_tracker.update(candidate, dets, t_s, lanes["lane_change"])

        t0_s = time.perf_counter()
        matched = None
        if front is not None:
            matched = match_front_brake(
                detect_brake_boxes(frame, brake_model, brake_cfg), front["box_xyxy_px"], brake_cfg["match_iou_min"]
            )
        brake = brake_tracker.update(front, matched, frame, t_s)
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)

        draw_brake(frame, dets, front, brake)
        writer.write(frame)
        n += 1
        if front is not None:
            if brake["brake_on"] is None:
                counts["unknown"] += 1
            elif brake["brake_on"]:
                counts["on"] += 1
            elif brake["model_on"]:
                counts["tail"] += 1
            else:
                counts["off"] += 1
    cap.release()
    writer.release()

    n_front = max(sum(counts.values()), 1)
    steady = proc_ms[1:] or proc_ms
    print(f"프레임 {n}장 → {args.out} | 앞차 있는 프레임 {sum(counts.values())}장")
    print(
        f"켜짐 {counts['on'] / n_front:.0%} | 꺼짐 {counts['off'] / n_front:.0%} | "
        f"미등 처리(모델 켜짐 → 꺼짐) {counts['tail'] / n_front:.0%} | 모름 {counts['unknown'] / n_front:.0%}"
    )
    print(f"처리 시간(브레이크등, 첫 프레임 제외) 평균 {sum(steady) / len(steady):.1f} ms | 최대 {max(steady):.1f} ms")


if __name__ == "__main__":
    main()
