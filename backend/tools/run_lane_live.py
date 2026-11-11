"""카메라(또는 영상)에서 프레임을 받는 즉시 차선을 찾아 보여준다 (실시간 확인용).

실행 (backend 폴더에서):
    python -m tools.run_lane_live --source 0
    python -m tools.run_lane_live --source data/videos/sample.mp4 --config configs/lane_shorts_portrait.yaml \
        --realtime --no-show --save outputs/lane_live.mp4

화면 색은 run_lane.py와 같다. 왼쪽 위에 처리 시간(ms)과 실제 fps를 표시한다. q를 누르면 종료.
폰→서버 프레임 수신은 서버 담당 코드에서 한다. 그 루프에서도 아래 process_frame()처럼
detect_lanes() → LaneTracker.update() 순서로 부르면 된다.
"""
import argparse
import time
from pathlib import Path

import cv2
import yaml

from app.lane.detect import detect_lanes
from app.lane.tracker import LaneTracker
from tools.run_lane import draw_lanes

QUIT_KEY = ord("q")
MS_PER_S = 1000
FALLBACK_FPS = 30  # 카메라가 fps를 알려주지 않을 때 저장 영상에 쓸 값


def process_frame(frame, tracker: LaneTracker, t_s: float, cfg: dict) -> tuple[dict, dict]:
    """한 프레임의 차선을 찾고 트래커로 부드럽게 만든다.

    사용처: 2단계(차선). main()의 프레임 반복문에서 매 프레임 호출 (서버 수신 루프에서도 같은 순서로 쓴다).
    입력: frame BGR 프레임, tracker 영상 하나에 하나인 LaneTracker, t_s 촬영 시각(s), cfg 전체 설정
    출력: (raw 트래커 적용 전 결과, smooth 적용 후 결과), 둘 다 {"left": Line|None, "right": Line|None}
    """
    raw = detect_lanes(frame, cfg["camera"], cfg["lane"])
    return raw, tracker.update(raw, t_s)


def main() -> None:
    """프레임 받기 → 차선 인식·트래커 → 표시(·저장)를 반복하고, 끝나면 속도와 검출률을 출력한다.

    입력: --source 카메라 번호 또는 영상 경로·URL, --config 설정 파일, --save 결과 mp4 경로(선택),
          --no-show 창을 띄우지 않음, --realtime 파일을 원본 fps 속도로 재생 (늦으면 프레임 건너뜀)
    출력: 화면 창 또는 저장 영상, 콘솔에 처리 ms(평균/최대)·평균 fps·좌우 검출률(트래커 전/후)
    주의: 촬영 시각은 파일이면 영상 안의 시각(CAP_PROP_POS_MSEC), 카메라·URL이면 받은 순간의 time.perf_counter() (Windows에서 monotonic은 해상도가 약 16ms라 사용하지 않음).
          파일을 빠르게 처리해도 트래커가 실제 촬영 시간 기준으로 동작하게 하려는 것.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--save")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--realtime", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    is_camera = args.source.isdigit()
    is_file = not is_camera and Path(args.source).is_file()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        raise SystemExit(f"입력을 열 수 없음: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS

    writer = None
    if args.save:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    tracker = LaneTracker(cfg["lane"])
    proc_ms = []
    n_skipped = 0
    n_raw = {"left": 0, "right": 0}
    n_smooth = {"left": 0, "right": 0}
    t_start_s = time.perf_counter()
    t_prev_s = t_start_s
    while True:
        if is_file and args.realtime:
            # 원본 속도 재생: 지금 보여야 할 프레임 번호까지 건너뛰고, 앞서 있으면 기다린다
            target_idx = int((time.perf_counter() - t_start_s) * src_fps)
            next_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            while next_idx < target_idx and cap.grab():
                next_idx += 1
                n_skipped += 1
            wait_s = next_idx / src_fps - (time.perf_counter() - t_start_s)
            if wait_s > 0:
                time.sleep(wait_s)
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S if is_file else time.perf_counter()

        t0_s = time.perf_counter()
        raw, smooth = process_frame(frame, tracker, t_s, cfg)
        out = draw_lanes(frame, smooth, cfg["camera"], cfg["lane"])
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)

        now_s = time.perf_counter()
        fps = 1 / max(now_s - t_prev_s, 1e-6)  # 직전 프레임과의 간격으로 본 실제 fps
        t_prev_s = now_s
        cv2.putText(out, f"{proc_ms[-1]:.1f} ms | {fps:.0f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for side in n_raw:
            n_raw[side] += raw[side] is not None
            n_smooth[side] += smooth[side] is not None
        if writer is not None:
            writer.write(out)
        if not args.no_show:
            cv2.imshow("lane (q: quit)", out)
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break
    elapsed_s = time.perf_counter() - t_start_s
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    n = max(len(proc_ms), 1)
    print(f"처리 프레임 {len(proc_ms)}장 (건너뜀 {n_skipped}장) | 평균 {len(proc_ms) / max(elapsed_s, 1e-6):.1f} fps")
    print(f"처리 시간 평균 {sum(proc_ms) / n:.2f} ms | 최대 {max(proc_ms, default=0):.2f} ms")
    print(f"왼쪽 차선 {n_raw['left'] / n:.0%} → 트래커 후 {n_smooth['left'] / n:.0%} | "
          f"오른쪽 차선 {n_raw['right'] / n:.0%} → 트래커 후 {n_smooth['right'] / n:.0%}")


if __name__ == "__main__":
    main()
