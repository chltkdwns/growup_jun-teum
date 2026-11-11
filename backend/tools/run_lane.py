"""영상에 차선 인식 결과를 그려 저장한다 (눈으로 확인용).

실행 (backend 폴더에서):
    python -m tools.run_lane --video data/videos/sample.mp4 --out outputs/lane_sample.mp4

화면: 노란 사다리꼴 = 관심 영역, 초록 = 왼쪽 차선, 파랑 = 오른쪽 차선, 빨간 세로선 = 내 차 정면
"""
import argparse
from pathlib import Path

import cv2
import yaml

from app.lane.detect import detect_lanes, lane_roi_polygon


def draw_lanes(frame, lanes: dict, cfg_camera: dict, cfg_lane: dict):
    """차선·관심 영역·정면 위치를 프레임 위에 그린다.

    사용처: main()에서 매 프레임 호출 (확인용 화면).
    입력: frame 원본 프레임, lanes detect_lanes() 결과, cfg_camera·cfg_lane
    출력: 그림이 그려진 프레임 (원본은 바꾸지 않음)
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.polylines(out, [lane_roi_polygon(w, h, cfg_camera, cfg_lane)], True, (0, 255, 255), 2)
    cx = int(w * cfg_camera["center_x_ratio"])
    cv2.line(out, (cx, int(h * cfg_lane["roi_top_ratio"])), (cx, h), (0, 0, 255), 1)
    for side, color in (("left", (0, 255, 0)), ("right", (255, 0, 0))):
        if lanes[side] is not None:
            xb, yb, xt, yt = lanes[side]
            cv2.line(out, (xb, yb), (xt, yt), color, 4)
    return out


def main() -> None:
    """영상 → 프레임마다 차선 인식 → 그린 영상 저장 + 차선을 찾은 비율 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --config 설정 파일
    출력: 결과 영상 파일, 콘솔에 왼쪽/오른쪽 차선 검출 비율
    주의: 결과 영상은 outputs/ 아래에 둔다 (커밋 금지).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = n_left = n_right = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lanes = detect_lanes(frame, cfg["camera"], cfg["lane"])
        writer.write(draw_lanes(frame, lanes, cfg["camera"], cfg["lane"]))
        n += 1
        n_left += lanes["left"] is not None
        n_right += lanes["right"] is not None
    cap.release()
    writer.release()
    print(f"프레임 {n}장 | 왼쪽 차선 {n_left / max(n, 1):.0%} | 오른쪽 차선 {n_right / max(n, 1):.0%} → {args.out}")


if __name__ == "__main__":
    main()
