"""영상을 PipelineWarningSource(분석 + 거리·TTC + 경고 단계)로 돌려, 경고 단계를 색으로 그린 영상과 경고 타임라인을 만든다 (확인용).

실행 (backend 폴더에서):
    python -m tools.run_warning --video data/videos/day_60s.mp4 --out outputs/warning_day.mp4 [--speed-kmh 60] [--config ...]

화면: 앞차 박스 색 = 경고 단계 (초록 안전, 노랑 주의, 주황 경고, 빨강 위험), 박스 위에 거리·TTC, 맨 위 줄에 지금 경고 목록.
      브레이크등 가중치(weights/brake_8s.pt)가 있으면 앞차 브레이크등 켜짐(BRAKE ON)도 표시한다.
콘솔: 경고 상태가 바뀐 시각 목록, 단계별 프레임 수, 가장 낮았던 TTC.
"""
import argparse
import time
from pathlib import Path

import cv2
import yaml

from app.pipeline import PerceptionPipeline
from app.risk.warning import LEVELS
from app.risk.warning_source import PipelineWarningSource
from tools.run_lane import draw_lanes

MS_PER_S = 1000
# 단계별 그리기 색 (BGR). 확인용 화면에서만 쓴다 (앱 화면 색은 프론트 상수 파일이 정함)
LEVEL_COLOR_BGR = {"safe": (0, 200, 0), "caution": (0, 255, 255), "warning": (0, 165, 255), "danger": (0, 0, 255)}


def draw_frame(frame, result: dict, lead: dict, warnings: list, cam: dict, lane_cfg: dict):
    """차선·차 박스·앞차의 거리·TTC·경고 단계를 한 프레임에 그린다.

    사용처: main()에서 매 프레임 (확인용 화면).
    입력: frame 원본 프레임, result pipeline.process() 결과, lead 거리·TTC, warnings 이번 프레임 경고 목록, cam·lane_cfg config 섹션
    출력: 그림이 그려진 새 프레임
    """
    out = draw_lanes(frame, result["lanes"], cam, lane_cfg)
    for d in result["dets"]:
        x1, y1, x2, y2 = (int(v) for v in d["box_xyxy_px"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
    front = result["front"]
    # 앞차 박스 색은 앞차 관련 경고(차선 이탈 제외) 중 가장 높은 단계
    lead_levels = [w.level for w in warnings if w.type != "lane_departure"]
    top_level = max(lead_levels, key=LEVELS.index) if lead_levels else "safe"
    if front is not None:
        x1, y1, x2, y2 = (int(v) for v in front["box_xyxy_px"])
        cv2.rectangle(out, (x1, y1), (x2, y2), LEVEL_COLOR_BGR[top_level], 3)
        ttc = "-" if lead["ttc_s"] is None else f"{lead['ttc_s']:.2f}s"
        label = f"{lead['distance_m']:.1f}m TTC {ttc}"
        brake = result["brake"]
        if brake is not None and brake["brake_on"]:
            label += " BRAKE ON"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEVEL_COLOR_BGR[top_level], 2)
    banner = " | ".join(f"{w.type}:{w.level}" for w in warnings) or "safe"
    cv2.putText(out, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LEVEL_COLOR_BGR[top_level], 2)
    return out


def main() -> None:
    """영상 → PipelineWarningSource → 그린 영상 저장 + 경고 요약 출력.

    입력: --video 영상, --out 결과 영상, --speed-kmh 내 속도(생략하면 모름 → 안전거리 판단 건너뜀), --config 설정 파일
    출력: 결과 영상, 콘솔에 경고 상태가 바뀐 시각, 단계별 프레임 비율, 가장 낮은 TTC, 처리 ms
    주의: 실제 가중치(detect·brake)가 필요하다. 브레이크등 가중치가 없으면 브레이크등만 꺼진다 (PerceptionPipeline 동작).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--speed-kmh", type=float, default=None)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cam, lane_cfg = cfg["camera"], cfg["lane"]
    source = PipelineWarningSource(PerceptionPipeline(cfg), cfg["risk"], args.speed_kmh)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    level_frames = dict.fromkeys(LEVELS, 0)  # 프레임마다 가장 높은 단계 (차선 이탈 포함)
    changes, prev_state = [], None
    min_ttc, brake_on_frames, proc_ms = None, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S
        t0_s = time.perf_counter()
        warnings = source(frame, t_s)
        proc_ms.append((time.perf_counter() - t0_s) * MS_PER_S)
        result, lead = source.last_result, source.last_lead
        writer.write(draw_frame(frame, result, lead, warnings, cam, lane_cfg))

        n += 1
        level_frames[max((w.level for w in warnings), key=LEVELS.index, default="safe")] += 1
        state = tuple(sorted((w.type, w.level) for w in warnings))
        if state != prev_state:
            changes.append((t_s, state))
            prev_state = state
        if lead["ttc_s"] is not None and (min_ttc is None or lead["ttc_s"] < min_ttc[0]):
            min_ttc = (lead["ttc_s"], t_s)
        brake_on_frames += bool(result["brake"] and result["brake"]["brake_on"])
    cap.release()
    writer.release()

    print(f"프레임 {n}장 → {args.out} | 내 속도 {args.speed_kmh if args.speed_kmh is not None else '모름(안전거리 판단 건너뜀)'}")
    print("단계별 프레임 비율:", " | ".join(f"{k} {v / max(n, 1):.0%}" for k, v in level_frames.items()))
    if min_ttc:
        print(f"가장 낮은 TTC {min_ttc[0]:.2f}초 ({min_ttc[1]:.2f}초 지점)")
    print(f"브레이크등 켜짐 프레임 {brake_on_frames / max(n, 1):.0%} (브레이크등 모델 {'사용' if source._pipeline.brake_model is not None else '없음'})")
    print(f"경고 상태가 바뀐 시각 {len(changes)}회:")
    for t_s, state in changes[:40]:
        print(f"  {t_s:7.2f}초  {', '.join(f'{t}:{lv}' for t, lv in state) or 'safe'}")
    if len(changes) > 40:
        print(f"  ... 외 {len(changes) - 40}회")
    steady = proc_ms[1:] or proc_ms
    print(f"처리 시간(분석+거리·TTC+경고, 첫 프레임 제외) 평균 {sum(steady) / len(steady):.1f} ms | 최대 {max(steady):.1f} ms")


if __name__ == "__main__":
    main()
