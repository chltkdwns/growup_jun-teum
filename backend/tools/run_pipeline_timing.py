"""파이프라인 단계별 처리 시간과 브레이크등 판단 비율을 브레이크등 실행 간격(every_n)별로 잰다 (성능표용).

실행 (backend 폴더에서):
    python -m tools.run_pipeline_timing --video data/videos/night.mp4 --every-n 1 2 3
    python -m tools.run_pipeline_timing --video data/videos/night.mp4 --every-n 1 3 --device cpu   # CPU로 고정
    python -m tools.run_pipeline_timing --video data/videos/night.mp4 --every-n 3 --resize 1280x720  # 폰이 보낼 크기로 줄여서

출력: every_n마다 단계별(탐지·차선·앞차·브레이크등·끼어들기·전체) 평균/p95/최대 ms, 15fps(67ms) 안에 드는지,
      앞차 있는 프레임 중 켜짐·꺼짐·미등 처리·모름 비율. 마지막에 발표 자료에 붙일 마크다운 표.
"""
import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.detect.vehicles import detect_vehicles
from app.lane.detect import detect_lanes
from app.lane.front_car import select_front_car
from app.lane.offset import lane_offset
from app.pipeline import PerceptionPipeline

MS_PER_S = 1000
TARGET_FPS = 15  # 목표 처리 속도 (기획서: 15fps = 한 장 약 67ms)
STAGES = ("detect", "lane", "front", "brake", "cut_in", "total")
STAGE_KO = {"detect": "탐지", "lane": "차선", "front": "앞차", "brake": "브레이크등", "cut_in": "끼어들기", "total": "전체"}


def process_timed(pipe: PerceptionPipeline, frame_bgr: np.ndarray, t_s: float) -> tuple[dict, dict]:
    """PerceptionPipeline.process()와 같은 순서로 한 프레임을 처리하면서 단계마다 걸린 시간을 잰다.

    사용처: main()에서 매 프레임 (측정 전용).
    입력: pipe 파이프라인(모델·트래커 보관), frame_bgr 원본 프레임(BGR), t_s 촬영 시각(s)
    출력: (결과 {"front", "brake"}, 단계별 ms {"detect", "lane", "front", "brake", "cut_in", "total"})
    주의: process()를 그대로 부르면 단계별 시간을 알 수 없어서 순서를 여기 다시 적었다.
          pipeline.py의 process() 순서가 바뀌면 여기도 같이 고쳐야 한다.
          YOLO 결과를 .cpu()로 꺼내는 시점에 GPU 작업이 끝나므로 GPU에서도 단계 시간이 맞게 잡힌다 [추정].
    """
    cfg = pipe.cfg
    cam = cfg["camera"]
    img_h, img_w = frame_bgr.shape[:2]
    ms = {}

    t0 = time.perf_counter()
    dets = pipe.voter.update(detect_vehicles(frame_bgr, pipe.model, cam, cfg["detect"], t_s), t_s)
    t1 = time.perf_counter()
    lanes = pipe.lane_tracker.update(detect_lanes(frame_bgr, cam, cfg["lane"]), t_s)
    lane_offset(lanes, img_w, cam)
    t2 = time.perf_counter()
    candidate = select_front_car(dets, lanes, img_w, img_h, cam, cfg["lane"], cfg["front_car"])
    front = pipe.front_tracker.update(candidate, dets, t_s, lanes["lane_change"])
    t3 = time.perf_counter()
    brake = pipe._read_brake(frame_bgr, front, t_s)  # every_n 건너뛰기 규칙까지 파이프라인 것 그대로
    t4 = time.perf_counter()
    pipe.cut_in_detector.update(dets, lanes, front, t_s, img_h, img_w, lanes["lane_change"])
    t5 = time.perf_counter()

    for name, (a, b) in zip(STAGES, ((t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t0, t5))):
        ms[name] = (b - a) * MS_PER_S
    return {"front": front, "brake": brake}, ms


def brake_kind(brake: dict | None) -> str:
    """브레이크등 결과를 켜짐/꺼짐/미등 처리/모름 중 하나로 나눈다 (run_brake.py와 같은 기준).

    사용처: main()에서 앞차 있는 프레임마다.
    입력: brake BrakeState 또는 None(브레이크등 모델 없음)
    출력: "on" | "off" | "tail"(모델은 켜짐인데 미등 걸러내기로 꺼짐) | "unknown"
    """
    if brake is None or brake["brake_on"] is None:
        return "unknown"
    if brake["brake_on"]:
        return "on"
    return "tail" if brake["model_on"] else "off"


def run_once(cfg: dict, video: str, max_frames: int | None, resize: tuple[int, int] | None = None) -> dict:
    """영상 하나를 처음부터 끝까지(또는 max_frames장) 돌려 단계별 ms 목록과 브레이크등 판단 개수를 모은다.

    사용처: main()에서 every_n 값마다 한 번. 매번 새 파이프라인을 만든다 (추적 상태가 섞이지 않게).
    입력: cfg 설정 전체(pipeline.brake_every_n_frames를 바꿔서 넣음), video 영상 경로, max_frames 최대 프레임 수,
          resize (가로, 세로) px로 줄여서 넣기 (실시간에서 폰이 보낼 크기 흉내, None이면 원본 크기)
    출력: {"ms": {단계: [ms, ...]}, "counts": {"on", "off", "tail", "unknown"}, "n": 프레임 수, "has_brake": bool}
    주의: 첫 프레임은 모델 준비(워밍업) 시간이 섞여서 ms 목록에서 뺀다.
    """
    pipe = PerceptionPipeline(cfg)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {video}")
    ms = {s: [] for s in STAGES}
    counts = {"on": 0, "off": 0, "tail": 0, "unknown": 0}
    n = 0
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S
        if resize is not None:
            frame = cv2.resize(frame, resize)  # 줄이는 시간은 폰에서 하는 일이라 측정에서 뺀다
        res, frame_ms = process_timed(pipe, frame, t_s)
        if n > 0:
            for s in STAGES:
                ms[s].append(frame_ms[s])
        if res["front"] is not None:
            counts[brake_kind(res["brake"])] += 1
        n += 1
    cap.release()
    return {"ms": ms, "counts": counts, "n": n, "has_brake": pipe.brake_model is not None}


def device_name(cfg_device) -> str:
    """측정 장치 이름을 돌려준다 (결과표에 같이 적기 위해).

    사용처: main()에서 한 번.
    입력: cfg_device config detect.device 값 (None = 자동, "cpu", 0 = 첫 GPU)
    출력: "GPU 이름" 또는 "CPU 이름"
    """
    import torch  # ultralytics가 이미 쓰는 라이브러리 (새로 추가하는 것 아님)

    if cfg_device != "cpu" and torch.cuda.is_available():
        return f"GPU {torch.cuda.get_device_name(0)}"
    return f"CPU {platform.processor() or platform.machine()}"


def main() -> None:
    """every_n 값마다 영상을 돌려 단계별 처리 시간과 브레이크등 판단 비율을 출력한다.

    입력: --video 영상 경로, --config 설정 파일, --every-n 비교할 간격들, --device cpu/0 (없으면 config 그대로),
          --max-frames 앞에서 몇 장만 잴지
    출력: 콘솔에 every_n별 요약과 마크다운 표
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--every-n", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--device", default=None, help='"cpu" 또는 GPU 번호(0). 없으면 config의 device')
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--resize", default=None, help="넣기 전에 이 크기로 줄임, 예: 1280x720 (실시간 전송 크기 흉내)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device is not None:
        dev = "cpu" if args.device == "cpu" else int(args.device)
        cfg["detect"]["device"] = dev
        cfg["brake"]["device"] = dev
    device = device_name(cfg["detect"]["device"])
    budget_ms = MS_PER_S / TARGET_FPS
    resize = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None
    cap = cv2.VideoCapture(args.video)
    src_size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    cap.release()
    size = f"{src_size} → {args.resize}" if resize else src_size
    print(f"영상 {args.video} ({size}) | 장치 {device} | 탐지 imgsz {cfg['detect']['imgsz']} | 브레이크등 imgsz {cfg['brake']['imgsz']}")

    rows = []
    for every_n in args.every_n:
        cfg["pipeline"]["brake_every_n_frames"] = every_n
        r = run_once(cfg, args.video, args.max_frames, resize)
        if not r["has_brake"]:
            print(f"주의: 브레이크등 가중치가 없음({cfg['brake']['weights_path']}) → 브레이크등 단계 0ms로 잡힘")
        stat = {s: (np.mean(v), np.percentile(v, 95), np.max(v)) if v else (0.0, 0.0, 0.0) for s, v in r["ms"].items()}
        n_front = max(sum(r["counts"].values()), 1)
        ratio = {k: v / n_front for k, v in r["counts"].items()}
        ok = "OK" if stat["total"][1] <= budget_ms else "초과"
        print(f"\n[every_n={every_n}] 프레임 {r['n']}장 | 앞차 있는 프레임 {sum(r['counts'].values())}장")
        for s in STAGES:
            mean, p95, mx = stat[s]
            print(f"  {STAGE_KO[s]:<6} 평균 {mean:6.1f} ms | p95 {p95:6.1f} ms | 최대 {mx:6.1f} ms")
        print(f"  → 전체 p95 {stat['total'][1]:.1f} ms, 목표 {budget_ms:.0f} ms({TARGET_FPS}fps) {ok}")
        print(f"  켜짐 {ratio['on']:.0%} | 꺼짐 {ratio['off']:.0%} | 미등 처리 {ratio['tail']:.0%} | 모름 {ratio['unknown']:.0%}")
        rows.append((every_n, stat, ratio, ok))

    print(f"\n| every_n | 탐지 | 차선 | 앞차 | 브레이크등 | 끼어들기 | 전체 평균 | 전체 p95 | {TARGET_FPS}fps | 켜짐 | 꺼짐 | 미등 처리 | 모름 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for every_n, stat, ratio, ok in rows:
        stage_ms = " | ".join(f"{stat[s][0]:.1f}" for s in STAGES[:-1])
        print(
            f"| {every_n} | {stage_ms} | {stat['total'][0]:.1f} | {stat['total'][1]:.1f} | {ok} | "
            f"{ratio['on']:.0%} | {ratio['off']:.0%} | {ratio['tail']:.0%} | {ratio['unknown']:.0%} |"
        )
    print(f"(단위 ms, 첫 프레임 제외, 장치 {device}, 크기 {size})")


if __name__ == "__main__":
    main()
