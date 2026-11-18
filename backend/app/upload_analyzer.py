"""올린 영상 파일 1개를 처음부터 끝까지 분석하고, 아찔한 순간을 이벤트·클립으로 저장한다 (FastAPI를 import하지 않음).

흐름: 프레임 읽기 → 경고 판단(WarningSource) → 경고(TTC 3초) 이상이면 이벤트 저장 + 클립 예약 → 클립 저장
실시간(live_recorder.py)과 같은 기록 함수(record_event, save_clip, ClipRecorder)를 쓴다.
차이는 프레임이 폰이 아니라 파일에서 오고, 촬영 시각이 받은 순간이 아니라 영상 속 시각이라는 것뿐이다.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
from sqlalchemy.orm import Session, sessionmaker

from app.clip_recorder import ClipRecorder
from app.db import SessionLocal
from app.models import Drive
from app.nearmiss import (
    LEVEL_RANK,
    NEARMISS_CFG,
    ROOT,
    record_event,
    save_clip,
    to_event_in,
)
from app.schemas import WarningSource

logger = logging.getLogger(__name__)

MS_PER_S = 1000  # OpenCV 영상 위치(CAP_PROP_POS_MSEC)는 밀리초


@dataclass
class UploadProgress:
    """영상 분석 1건의 진행 상황. 분석 스레드가 고치고, 상태 API가 읽는다.

    주의: 메모리에만 있어서 서버를 다시 켜면 사라진다 (DB 표를 바꾸지 않으려고. main.py 상태 API 참고).
    """

    # queued 차례 기다림 / running 분석 중 / done 끝 / failed 실패(error에 이유)
    status: Literal["queued", "running", "done", "failed"] = "queued"
    frames_done: int = 0
    frames_total: int | None = (
        None  # 영상 머리 정보의 프레임 수 (코덱에 따라 정확하지 않을 수 있음 [추정])
    )
    error: str | None = None


def analyze_video(
    session: Session,
    drive_id: int,
    video_path: Path,
    judge: WarningSource,
    recorder: ClipRecorder,
    progress: UploadProgress,
) -> None:
    """영상 파일을 프레임마다 판단하고, 경고 이상은 이벤트와 앞뒤 클립으로 저장한다. 끝나면 주행을 닫는다.

    사용처: 5단계(기록). run_upload_job()이 분석 스레드에서 부른다.
    입력: session DB 세션, drive_id 미리 만든 주행 번호(source=upload), video_path 올린 영상 파일,
          judge 경고 판단 (frame, t_s → 경고 목록), recorder 클립 저장기, progress 진행 상황(여기서 고침)
    출력: 없음 (DB의 events·clips·drives 행이 결과)
    주의: 촬영 시각은 프레임 수가 아니라 영상 속 시각(CAP_PROP_POS_MSEC)이다.
          FINEVU AVI는 스트림이 2개(앞·뒤)인데 OpenCV는 첫 스트림(앞)만 읽는다.
          이벤트의 실제 시각(occurred_at)은 영상이 찍힌 시각을 몰라서 '업로드 시각 + 영상 속 시각'으로 둔다.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {video_path.name}")
    progress.frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    drive = session.get(Drive, drive_id)
    started_at: datetime = drive.started_at  # 업로드 시각 (DB가 시간대 포함으로 줌)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / MS_PER_S
            # 경고(TTC 3초) 이상만 기록한다 (docs/SPEC.md). 같은 종류 5초 안은 record_event가 합친다
            for w in judge(frame, t_s):
                if w.level not in LEVEL_RANK:
                    continue
                event, _ = record_event(session, drive_id, to_event_in(w, started_at))
                # 합쳐진 경고면 같은 event_id라 클립 끝 시각만 늘어난다
                recorder.trigger(event.id, w.t_s, f"{drive_id}/{event.id}")
            for clip in recorder.push(frame, t_s):
                save_clip(session, clip)
            progress.frames_done += 1
        # 영상 끝 근처 경고는 뒤 5초가 모자라도 있는 만큼 저장한다
        for clip in recorder.flush():
            save_clip(session, clip)
    finally:
        cap.release()

    drive.ended_at = datetime.now().astimezone()
    drive.frames = progress.frames_done
    session.commit()


def run_upload_job(
    drive_id: int,
    video_path: Path,
    make_judge: Callable[[], WarningSource],
    progress: UploadProgress,
    session_factory: sessionmaker = SessionLocal,
    clip_dir: Path = ROOT / NEARMISS_CFG["clip_dir"],
    pre_s: float = NEARMISS_CFG["pre_s"],
    post_s: float = NEARMISS_CFG["post_s"],
) -> None:
    """영상 분석 1건을 처음부터 끝까지 실행하고, 결과를 progress.status에 남긴다.

    사용처: 5단계(기록). main.py의 POST /drives/upload가 분석 스레드에 맡긴다.
    입력: drive_id 주행 번호, video_path 저장한 영상, make_judge 경고 판단을 만드는 함수,
          progress 진행 상황, 나머지는 테스트에서 바꿔 끼우는 값 (기본은 config.yaml nearmiss)
    출력: 없음
    주의: 경고 판단은 여기(분석 스레드)에서 만든다. YOLO 로딩이 몇 초 걸려서 요청을 붙잡지 않으려고.
          뒤에서 도는 일이라 오류를 그냥 두면 아무도 모르게 사라진다. 그래서 모든 오류를 잡아 failed로 남긴다.
    """
    progress.status = "running"
    try:
        judge = make_judge()
        recorder = ClipRecorder(clip_dir, pre_s, post_s)
        with session_factory() as session:
            analyze_video(session, drive_id, video_path, judge, recorder, progress)
        progress.status = "done"
    except Exception as e:
        logger.exception("영상 분석 실패 (주행 %s)", drive_id)
        progress.status = "failed"
        progress.error = str(e)
