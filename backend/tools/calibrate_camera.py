"""개발용 영상에서 카메라 보정값(vanishing_x_ratio, center_x_ratio, roi_top_ratio) 추천값을 구한다.

실행 (backend 폴더에서):
    python -m tools.calibrate_camera --video data/videos/sample.mp4 --config configs/lane_shorts_portrait.yaml
    python -m tools.calibrate_camera --video data/videos/new.mp4 --wide-roi   # 처음 보는 카메라

config 파일은 고치지 않는다. 추천값을 보고 사람이 고친다. 평가용 영상으로 돌리지 않는다.
"""
import argparse
from pathlib import Path

import cv2
import yaml

from app.lane.calibrate import (
    estimate_calibration,
    lane_bottom_center_x,
    vanishing_point,
)
from app.lane.detect import detect_lanes

DEFAULT_MAX_SECONDS = 30  # 보정에 쓸 앞부분 길이 (초). 너무 길면 오래 걸리고, 짧으면 소실점이 모자람
MS_PER_S = 1000


def main() -> None:
    """앞부분 영상에서 프레임마다 소실점·차선 가운데를 모아 추천값과 현재 설정값을 나란히 출력한다.

    입력: --video 개발용 영상 경로, --config 설정 파일, --max-seconds 앞에서부터 쓸 길이(초),
          --wide-roi 관심 영역을 넓게(config calibration.wide_*) 잡고 찾기 (현재 설정이 틀렸을 때)
    출력: 콘솔에 사용 프레임 수, 소실점 흔들림(px), 항목별 현재값 → 추천값
    주의: bonnet_ratio는 자동으로 추정하지 않는다 (대시보드 모양이 차마다 달라서). 결과 영상으로 직접 확인한다.
          center_x_ratio는 --wide-roi일 때 추천하지 않는다. 차선 가운데는 비교하는 높이(관심 영역 아래변)에 따라
          달라지는데, 넓은 모드는 아래변이 화면 끝이라 실제 분석 높이(bonnet_ratio)와 다르기 때문.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    p.add_argument("--wide-roi", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg_calib = cfg["calibration"]
    cfg_camera, cfg_lane = dict(cfg["camera"]), dict(cfg["lane"])
    if args.wide_roi:
        # 현재 설정이 이 카메라에 안 맞을 수 있으니, 도로가 있을 만한 곳을 넉넉히 덮는다
        cfg_camera["bonnet_ratio"] = cfg_calib["wide_bonnet_ratio"]
        cfg_lane["roi_top_ratio"] = cfg_calib["wide_roi_top_ratio"]
        cfg_lane["roi_top_half_width_ratio"] = cfg_calib["wide_roi_top_half_width_ratio"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    img_w, img_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    points, centers_x = [], []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S > args.max_seconds:
            break
        n += 1
        lanes = detect_lanes(frame, cfg_camera, cfg_lane)
        vp = vanishing_point(lanes)
        if vp is None:
            continue
        points.append(vp)
        centers_x.append(lane_bottom_center_x(lanes))
    cap.release()

    print(f"사용 프레임 {len(points)}/{n}장 (좌우 차선이 모두 보인 프레임){' | 넓은 관심 영역' if args.wide_roi else ''}")
    rec = estimate_calibration(points, centers_x, img_w, img_h, cfg_calib)
    if rec is None:
        raise SystemExit(f"소실점이 {cfg_calib['min_points']}개보다 적어 추천할 수 없음. --wide-roi로 다시 시도해 보세요.")
    print(f"소실점 x 흔들림(사분위 범위) {rec['spread_x_px']:.1f}px")
    print(f"camera.vanishing_x_ratio  {cfg['camera']['vanishing_x_ratio']} → 추천 {rec['vanishing_x_ratio']:.2f}")
    if args.wide_roi:
        print(f"camera.center_x_ratio     {cfg['camera']['center_x_ratio']} → bonnet_ratio를 먼저 맞춘 뒤 --wide-roi 없이 다시 실행하세요.")
    else:
        print(f"camera.center_x_ratio     {cfg['camera']['center_x_ratio']} → 추천 {rec['center_x_ratio']:.2f}")
    print(f"lane.roi_top_ratio        {cfg['lane']['roi_top_ratio']} → 추천 {rec['roi_top_ratio']:.2f}")
    print(f"camera.bonnet_ratio       {cfg['camera']['bonnet_ratio']} → 자동 추정 안 함. 결과 영상에서 대시보드 바로 위인지 확인하세요.")


if __name__ == "__main__":
    main()
