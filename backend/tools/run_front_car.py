"""영상에 차선 + 차량 탐지 + 앞차를 함께 그려 저장한다 (눈으로 확인용).

실행 (backend 폴더에서):
    python -m tools.run_front_car --video data/videos/highway.mp4 --out outputs/front_highway.mp4 \
        --config configs/lane_shorts_portrait.yaml

화면: 굵은 빨간 박스 + "FRONT #id" = 앞차, 얇은 회색 박스 = 다른 차, 차선 색은 run_lane과 같음,
      기본 차선 영역을 쓰면 위쪽에 "DEFAULT AREA" (그때는 기본 영역 선을 주황으로 그림)
"""
import argparse
import time
from pathlib import Path

import cv2
import yaml

from app.detect.vehicles import detect_vehicles, load_detector
from app.lane.detect import detect_lanes
from app.lane.front_car import default_lanes, select_front_car
from app.lane.tracker import LaneTracker
from tools.run_lane import draw_lanes

MS_PER_S = 1000


def draw_front(frame, dets: list[dict], front: dict | None, lanes_used: dict | None):
    """다른 차는 얇게, 앞차는 굵은 빨간 박스로 그리고, 기본 차선 영역을 썼으면 표시한다.

    사용처: main()에서 draw_lanes() 다음에 매 프레임 (확인용 화면).
    입력: frame 그림을 그릴 프레임(직접 바뀜), dets detect_vehicles() 결과, front select_front_car() 결과,
          lanes_used 기본 차선 영역을 썼을 때 그 직선들(아니면 None)
    출력: 없음
    """
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box_xyxy_px"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
    if lanes_used is not None:
        for xb, yb, xt, yt in (lanes_used["left"], lanes_used["right"]):
            cv2.line(frame, (int(xb), int(yb)), (int(xt), int(yt)), (0, 165, 255), 2)
        cv2.putText(frame, "DEFAULT AREA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if front is not None:
        x1, y1, x2, y2 = (int(v) for v in front["box_xyxy_px"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"FRONT #{front['track_id']}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def main() -> None:
    """영상 → 프레임마다 차선(트래커)·탐지·앞차 → 그린 영상 저장 + 요약 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --config 설정 파일
    출력: 결과 영상, 콘솔에 앞차 있음 비율, 앞차가 바뀐 횟수·시각, 기본 차선 영역 사용 비율, 처리 ms
    주의: 처리 ms는 차선+탐지+앞차 합계이고 첫 프레임(모델 준비)은 뺀다.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cam, lane_cfg, front_cfg = cfg["camera"], cfg["lane"], cfg["front_car"]
    model = load_detector(cfg["detect"])
    tracker = LaneTracker(lane_cfg)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = n_front = n_default = 0
    prev_id, switches, proc_ms = None, [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S
        t0_s = time.perf_counter()
        lanes = tracker.update(detect_lanes(frame, cam, lane_cfg), t_s)
        dets = detect_vehicles(frame, model, cam, cfg["detect"], t_s)
        front = select_front_car(dets, lanes, w, h, cam, lane_cfg, front_cfg)
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)

        used_default = lanes["left"] is None or lanes["right"] is None
        out = draw_lanes(frame, lanes, cam, lane_cfg)
        draw_front(out, dets, front, default_lanes(w, h, cam, lane_cfg, front_cfg) if used_default else None)
        writer.write(out)

        front_id = None if front is None else front["track_id"]
        if front_id != prev_id:
            switches.append((t_s, n, prev_id, front_id))
            prev_id = front_id
        n += 1
        n_front += front is not None
        n_default += used_default
    cap.release()
    writer.release()

    steady = proc_ms[1:] or proc_ms
    print(f"프레임 {n}장 → {args.out}")
    print(f"앞차 있음 {n_front / max(n, 1):.0%} | 기본 차선 영역 사용 {n_default / max(n, 1):.0%}")
    print(f"앞차 바뀜 {len(switches)}회 (없음↔있음 포함):")
    for t_s, i, a, b in switches:
        print(f"  {t_s:6.2f}초 #{i}: {a} → {b}")
    print(f"처리 시간(차선+탐지+앞차, 첫 프레임 제외) 평균 {sum(steady) / len(steady):.1f} ms | 최대 {max(steady):.1f} ms")


if __name__ == "__main__":
    main()
