"""영상에 앞차 거리(m)와 TTC(s)를 그려 저장한다 (눈으로 확인용). --csv를 주면 프레임별 값도 저장한다.

실행 (backend 폴더에서):
    python -m tools.run_lead_risk --video data/videos/highway.mp4 --out outputs/lead_highway.mp4 \
        --csv outputs/lead_highway.csv --config tests/fixtures/config_highway_shorts.yaml

화면: 빨간 박스 = 앞차, 박스 위에 "id 번호 / 거리 m / TTC s" (TTC 없음은 "TTC -")
CSV: t_s, track_id, cls_name, box_w_px, distance_m, ttc_s (TTC가 이상할 때 박스 폭·거리부터 보려고)
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml

from app.detect.vehicles import detect_vehicles, load_detector
from app.lane.detect import detect_lanes
from app.lane.front_car import FrontCarTracker, default_lanes, select_front_car
from app.lane.tracker import LaneTracker
from app.risk.lead import LeadRiskTracker
from tools.run_front_car import draw_front
from tools.run_lane import draw_lanes

MS_PER_S = 1000
CSV_HEADER = ["t_s", "track_id", "cls_name", "box_w_px", "distance_m", "ttc_s"]


def draw_risk(frame, front: dict | None, risk: dict) -> None:
    """앞차 박스 위에 거리와 TTC 글자를 쓴다.

    사용처: main()에서 draw_front() 다음에 매 프레임 (확인용 화면).
    입력: frame 그림을 그릴 프레임(직접 바뀜), front 앞차 dict(없으면 None), risk LeadRiskTracker.update() 결과
    출력: 없음
    """
    if front is None or risk["distance_m"] is None:
        return
    x1, y1, x2, _ = (int(v) for v in front["box_xyxy_px"])
    ttc_text = "TTC -" if risk["ttc_s"] is None else f"TTC {risk['ttc_s']:.2f}s"
    text = f"#{front['track_id']} {risk['distance_m']:.1f}m {ttc_text}"
    cv2.putText(frame, text, (x1, max(y1 - 24, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main() -> None:
    """영상 → 프레임마다 차선·탐지·앞차·거리·TTC → 그린 영상 저장 + 요약 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --csv 프레임별 값 저장 경로(선택), --config 설정 파일
    출력: 결과 영상, (선택) CSV, 콘솔에 앞차 있음 비율, TTC 나온 비율, 가장 낮았던 TTC와 그 시각, 거리 범위, 처리 ms
    주의: 거리는 config의 hfov_deg(추정값)에 크게 좌우된다. 카메라가 다르면 화각부터 확인한다.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cam, lane_cfg, front_cfg = cfg["camera"], cfg["lane"], cfg["front_car"]
    model = load_detector(cfg["detect"])
    lane_tracker = LaneTracker(lane_cfg)
    front_tracker = FrontCarTracker(front_cfg)
    risk_tracker = LeadRiskTracker(cfg["risk"])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv_file = None
    csv_writer = None
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

    n = n_front = n_ttc = 0
    min_ttc = None  # (ttc_s, t_s)
    dist_list, proc_ms = [], []
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
        risk = risk_tracker.update(front, w, t_s)
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)

        used_default = lanes["left"] is None or lanes["right"] is None
        out = draw_lanes(frame, lanes, cam, lane_cfg)
        draw_front(out, dets, front, default_lanes(w, h, cam, lane_cfg, front_cfg) if used_default else None)
        draw_risk(out, front, risk)
        writer.write(out)

        n += 1
        if front is not None:
            n_front += 1
            x1, _, x2, _ = front["box_xyxy_px"]
            dist_list.append(risk["distance_m"])
            if csv_writer:
                ttc_s = "" if risk["ttc_s"] is None else f"{risk['ttc_s']:.3f}"
                csv_writer.writerow(
                    [f"{t_s:.3f}", front["track_id"], front["cls_name"], f"{x2 - x1:.1f}", f"{risk['distance_m']:.2f}", ttc_s]
                )
        if risk["ttc_s"] is not None:
            n_ttc += 1
            if min_ttc is None or risk["ttc_s"] < min_ttc[0]:
                min_ttc = (risk["ttc_s"], t_s)
    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    steady = proc_ms[1:] or proc_ms
    print(f"프레임 {n}장 → {args.out}" + (f" | 값 {args.csv}" if args.csv else ""))
    print(f"앞차 있음 {n_front / max(n, 1):.0%} | TTC 나온 프레임 {n_ttc / max(n, 1):.0%}")
    if min_ttc:
        print(f"가장 낮은 TTC {min_ttc[0]:.2f}초 ({min_ttc[1]:.2f}초 지점)")
    if dist_list:
        print(f"앞차 거리 {min(dist_list):.1f}~{max(dist_list):.1f} m (화각 {cfg['risk']['hfov_deg']}도 기준)")
    print(f"처리 시간(차선+탐지+앞차+거리·TTC, 첫 프레임 제외) 평균 {sum(steady) / len(steady):.1f} ms | 최대 {max(steady):.1f} ms")


if __name__ == "__main__":
    main()
