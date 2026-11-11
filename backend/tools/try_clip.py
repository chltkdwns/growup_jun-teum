"""실제 영상으로 기록 흐름 전체를 시험한다.

지정한 시각에 경고가 났다고 치고: 주행 만들기 → 이벤트 저장 → 클립 저장 → clips 표 기록.
나중에 진짜 경고 판단이 붙으면 '--at 대신 판단 결과'로 같은 함수를 부르면 된다.

실행 (backend 폴더에서):
    python -m tools.try_clip ../data/blackbox_4.mp4 --at 6
"""

import argparse
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from app.clip_recorder import ClipRecorder
from app.db import SessionLocal
from app.models import Device, Drive
from app.nearmiss import NEARMISS_CFG, ROOT, EventIn, record_event, save_clip

# 시험용 기기 (/docs에서 쓴 것과 같은 값이라 GET /events로 바로 볼 수 있다)
TEST_DEVICE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def main() -> None:
    """영상을 읽으며 --at 초에 경고 이벤트를 만들고, 클립까지 DB에 기록한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", type=float, required=True, help="경고 시각(초)")
    args = parser.parse_args()

    session = SessionLocal()
    if session.get(Device, TEST_DEVICE_ID) is None:
        session.add(Device(id=TEST_DEVICE_ID))
    drive = Drive(device_id=TEST_DEVICE_ID, source="upload", video_name=args.video.name)
    session.add(drive)
    session.commit()

    recorder = ClipRecorder(
        ROOT / NEARMISS_CFG["clip_dir"], NEARMISS_CFG["pre_s"], NEARMISS_CFG["post_s"]
    )
    started_at = datetime.now().astimezone()
    cap = cv2.VideoCapture(str(args.video))
    saved = []
    event = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 프레임 수가 아니라 영상 속 시각
        if event is None and t_s >= args.at:
            # 진짜로는 경고 판단 결과가 이 자리에 들어온다
            body = EventIn(
                occurred_at=started_at + timedelta(seconds=t_s),
                offset_s=t_s,
                type="lead",
                level="danger",
                min_ttc_s=1.3,
            )
            event, _ = record_event(session, drive.id, body)
            recorder.trigger(event.id, t_s, f"{drive.id}/{event.id}")
        saved += recorder.push(frame, t_s)
    saved += recorder.flush()

    for clip in saved:
        row = save_clip(session, clip)
        print(f"주행 {drive.id}, 이벤트 {clip.event_id} → {row.object_key}")
        print(
            f"  경고 전 {clip.pre_s:.1f}초 / 후 {clip.post_s:.1f}초, "
            f"{clip.fps:.1f}fps, {clip.width}x{clip.height}"
        )
    session.close()


if __name__ == "__main__":
    main()
