"""영상에 차선 인식 결과를 그려 저장한다 (눈으로 확인용).

실행 (backend 폴더에서):
    python -m tools.run_lane --video data/videos/sample.mp4 --out outputs/lane_sample.mp4

화면: 노란 사다리꼴 = 관심 영역, 초록 = 왼쪽 차선, 파랑 = 오른쪽 차선, 빨간 세로선 = 내 차 정면
      왼쪽 아래 글자 = 차선 중앙 대비 치우침 비율(lane_offset_ratio) 또는 "LANE N/A"(차선 인식 불가)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.lane.detect import detect_lanes, lane_roi_polygon
from app.lane.offset import STATUS_ESTIMATED, lane_offset
from app.lane.tracker import LaneTracker

MS_PER_S = 1000


def draw_lanes(frame, lanes: dict, cfg_camera: dict, cfg_lane: dict):
    """차선·관심 영역·정면 위치를 프레임 위에 그린다.

    사용처: main()에서 매 프레임 호출 (확인용 화면).
    입력: frame 원본 프레임, lanes detect_lanes() 또는 LaneTracker.update() 결과(추정한 쪽은 옅게), cfg_camera·cfg_lane
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
            if side in lanes.get("estimated", ()):
                # 폭으로 추정한 선은 옅고 가늘게 (실제로 찾은 선과 구분)
                cv2.line(out, (xb, yb), (xt, yt), tuple(c // 2 for c in color), 2)
            else:
                cv2.line(out, (xb, yb), (xt, yt), color, 4)
    return out


def draw_offset(frame, offset: dict, origin_px: tuple[int, int]):
    """치우침 비율(또는 차선 인식 불가)을 프레임 위에 글자로 쓴다.

    사용처: run_lane.main(), run_lane_live.main()에서 draw_lanes() 다음에 호출 (확인용 화면).
    입력: frame 그림을 그릴 프레임 (직접 바뀜), offset lane_offset() 결과, origin_px 글자 왼쪽 아래 위치(px)
    출력: 없음
    """
    ratio = offset["lane_offset_ratio"]
    text = "LANE N/A" if ratio is None else f"offset {ratio:+.2f}"
    if offset["lane_status"] == STATUS_ESTIMATED:
        text += " (est)"
    cv2.putText(frame, text, origin_px, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main() -> None:
    """영상 → 프레임마다 차선 인식 → 그린 영상 저장 + 차선을 찾은 비율 출력.

    입력: --video 영상 경로, --out 결과 영상 경로, --config 설정 파일
    출력: 결과 영상 파일, 콘솔에 왼쪽/오른쪽 차선 검출 비율, 치우침 비율 범위와 차선 인식 불가 비율
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

    tracker = LaneTracker(cfg["lane"])
    offsets = []
    n = n_left = n_right = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lanes = detect_lanes(frame, cfg["camera"], cfg["lane"])
        # 치우침은 실시간 모드와 같게 트래커 적용 후 차선으로 구한다 (촬영 시각 = 영상 안의 시각)
        smooth = tracker.update(lanes, cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S)
        offset = lane_offset(smooth, w, cfg["camera"])
        out = draw_lanes(frame, smooth, cfg["camera"], cfg["lane"])
        draw_offset(out, offset, (10, h - 20))
        writer.write(out)
        offsets.append(offset["lane_offset_ratio"])
        n += 1
        n_left += lanes["left"] is not None
        n_right += lanes["right"] is not None
    cap.release()
    writer.release()
    print(f"프레임 {n}장 | 왼쪽 차선 {n_left / max(n, 1):.0%} | 오른쪽 차선 {n_right / max(n, 1):.0%} → {args.out}")
    ok_vals = np.array([r for r in offsets if r is not None])
    if ok_vals.size:
        print(f"치우침 비율 최소 {ok_vals.min():+.2f} | 최대 {ok_vals.max():+.2f} | 평균 {ok_vals.mean():+.2f}")
    print(f"차선 인식 불가 {1 - ok_vals.size / max(n, 1):.0%} (트래커 적용 후 기준)")


if __name__ == "__main__":
    main()
