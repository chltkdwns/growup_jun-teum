"""영상에 차선 + 앞차 + 끼어들기를 그려 저장한다 (눈으로 확인용).

실행 (backend 폴더에서):
    python -m tools.run_cut_in --video data/videos/sample.mp4 --out outputs/cut_in_sample.mp4

화면: 차선·앞차는 run_front_car와 같음, 끼어들기가 확정되면 그 차에 노란 굵은 박스와 "CUT-IN <-/->"를 1초간 표시
"""
import argparse
import time
from pathlib import Path

import cv2
import yaml

from app.detect.vehicles import detect_vehicles, load_detector
from app.lane.cut_in import CutInDetector
from app.lane.detect import detect_lanes
from app.lane.front_car import FrontCarTracker, select_front_car
from app.lane.tracker import LaneTracker
from tools.run_front_car import draw_front
from tools.run_lane import draw_lanes

MS_PER_S = 1000
CUT_IN_SHOW_S = 1.0  # 끼어들기 표시를 화면에 남겨 둘 시간 (초, 확인용 화면 전용)
CUT_IN_TEXT = {"left": "CUT-IN ->", "right": "<- CUT-IN"}  # 들어온 쪽에서 안쪽을 가리키는 화살표


def main() -> None:
    """영상 → 프레임마다 차선·탐지·앞차·끼어들기 → 그린 영상 저장 + 끼어들기 목록과 처리 ms 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --config 설정 파일
    출력: 결과 영상, 콘솔에 끼어들기 시각·track_id·들어온 쪽, 처리 ms(차선+탐지+앞차+끼어들기, 첫 프레임 제외)
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cam, lane_cfg, front_cfg = cfg["camera"], cfg["lane"], cfg["front_car"]
    model = load_detector(cfg["detect"])
    lane_tracker, front_tracker = LaneTracker(lane_cfg), FrontCarTracker(front_cfg)
    cut_in_detector = CutInDetector(cfg["cut_in"])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    events, proc_ms, n = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S
        t0_s = time.perf_counter()
        lanes = lane_tracker.update(detect_lanes(frame, cam, lane_cfg), t_s)
        dets = detect_vehicles(frame, model, cam, cfg["detect"], t_s)
        candidate = select_front_car(dets, lanes, w, h, cam, lane_cfg, front_cfg)
        front = front_tracker.update(candidate, dets, t_s, lanes["lane_change"])
        cut_in = cut_in_detector.update(dets, lanes, front, t_s, h, lanes["lane_change"])
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)

        out = draw_lanes(frame, lanes, cam, lane_cfg)
        draw_front(out, dets, front, None)
        if cut_in is not None:
            events.append((n, cut_in))
        if events and t_s - events[-1][1]["t_s"] <= CUT_IN_SHOW_S:
            last = events[-1][1]
            box = next((d["box_xyxy_px"] for d in dets if d["track_id"] == last["track_id"]), None)
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(out, f"{CUT_IN_TEXT[last['side']]} #{last['track_id']}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(out)
        n += 1
    cap.release()
    writer.release()

    steady = proc_ms[1:] or proc_ms
    print(f"프레임 {n}장 → {args.out}")
    print(f"끼어들기 {len(events)}회:")
    for i, e in events:
        print(f"  {e['t_s']:6.2f}초 #{i}: track {e['track_id']} ({e['side']}에서 들어옴)")
    print(f"처리 시간(차선+탐지+앞차+끼어들기, 첫 프레임 제외) 평균 {sum(steady) / len(steady):.1f} ms"
          f" | 최대 {max(steady):.1f} ms")


if __name__ == "__main__":
    main()
