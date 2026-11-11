"""영상에 차량 탐지 결과를 그려 저장한다 (눈으로 확인용).

실행 (backend 폴더에서):
    python -m tools.run_detect --video data/videos/sample.mp4 --out outputs/detect_sample.mp4 \
        --config configs/lane_shorts_portrait.yaml

화면: 박스 + "클래스 #track_id 확신도", 주황 가로선 = 보닛 선 (이 아래는 탐지에 쓰지 않음)
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import yaml

from app.detect.vehicles import detect_vehicles, load_detector

MS_PER_S = 1000
CLS_COLORS = {"car": (0, 255, 0), "motorcycle": (0, 0, 255), "bus": (255, 0, 255), "truck": (255, 255, 0)}


def draw_detections(frame, dets: list[dict], bonnet_ratio: float):
    """탐지 박스·이름·보닛 선을 프레임 위에 그린다.

    사용처: main()에서 매 프레임 호출 (확인용 화면).
    입력: frame 원본 프레임, dets detect_vehicles() 결과, bonnet_ratio 보닛 선 높이(화면 높이 대비)
    출력: 그림이 그려진 프레임 (원본은 바꾸지 않음)
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (0, int(h * bonnet_ratio)), (w, int(h * bonnet_ratio)), (0, 165, 255), 1)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box_xyxy_px"])
        color = CLS_COLORS[d["cls_name"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tid = "-" if d["track_id"] is None else d["track_id"]
        cv2.putText(out, f"{d['cls_name']} #{tid} {d['score']:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def main() -> None:
    """영상 → 프레임마다 차량 탐지 → 그린 영상 저장 + 요약 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --config 설정 파일
    출력: 결과 영상 파일, 콘솔에 프레임 수·클래스별 탐지 수·track_id 수·처리 ms(평균/최대)·사용 장치
    주의: 첫 프레임은 모델 준비(워밍업) 때문에 느리므로 처리 ms 통계에서 뺀다.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = load_detector(cfg["detect"])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counts, track_ids, proc_ms = Counter(), set(), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0_s = time.perf_counter()
        dets = detect_vehicles(frame, model, cfg["camera"], cfg["detect"], cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S)
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)
        writer.write(draw_detections(frame, dets, cfg["camera"]["bonnet_ratio"]))
        counts.update(d["cls_name"] for d in dets)
        track_ids.update(d["track_id"] for d in dets if d["track_id"] is not None)
    cap.release()
    writer.release()

    steady = proc_ms[1:] or proc_ms
    # model.device는 원본 모델 위치라 GPU로 돌아도 cpu로 보일 수 있다. 실제 추론 장치는 predictor에 있다
    device = model.predictor.device if model.predictor is not None else model.device
    print(f"프레임 {len(proc_ms)}장 → {args.out} | 장치 {device}")
    print("클래스별 탐지 수(박스 합계): " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"서로 다른 track_id {len(track_ids)}개")
    print(f"처리 시간(첫 프레임 제외) 평균 {sum(steady) / len(steady):.1f} ms | 최대 {max(steady):.1f} ms"
          f" | 첫 프레임 {proc_ms[0]:.0f} ms")


if __name__ == "__main__":
    main()
