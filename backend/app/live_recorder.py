"""실시간 주행 1회의 아찔한 순간 기록. 프레임과 경고를 받아 DB·클립 저장을 뒤에서(스레드 1개) 한다.

왜 뒤에서 하나: 클립 1개(10초)를 mp4로 쓰는 데 1초 넘게 걸릴 수 있다 [추정]. 이걸 WebSocket 루프에서
하면 그동안 폰에 결과를 못 보내서 경고가 늦어진다. 그래서 루프는 submit()으로 일을 맡기고 바로 다음
프레임으로 간다. 일꾼 스레드는 1개라 프레임 순서가 그대로 지켜진다 (링 버퍼는 순서가 중요함).
"""

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.clip_recorder import ClipRecorder
from app.db import SessionLocal
from app.models import Device, Drive
from app.nearmiss import (
    LEVEL_RANK,
    NEARMISS_CFG,
    ROOT,
    record_event,
    save_clip,
    to_event_in,
)
from app.schemas import NearMissWarning

logger = logging.getLogger(__name__)


class LiveRecorder:
    """실시간 주행 1회 = 객체 1개. 주행(drives 행)을 만들고, 경고가 오면 이벤트와 앞뒤 클립을 저장한다.

    사용처: 5단계(기록). main.py의 /live WebSocket이 연결마다 하나 만든다.
    주의: DB 세션은 일꾼 스레드 안에서만 쓴다 (SQLAlchemy 세션은 여러 스레드가 같이 쓰면 안 됨).
    """

    def __init__(
        self,
        device_id: uuid.UUID,
        session_factory: sessionmaker = SessionLocal,
        clip_dir: Path = ROOT / NEARMISS_CFG["clip_dir"],
        pre_s: float = NEARMISS_CFG["pre_s"],
        post_s: float = NEARMISS_CFG["post_s"],
    ) -> None:
        self.device_id = device_id
        self.drive_id: int | None = None
        self._session_factory = session_factory
        self._session: Session | None = None
        self._recorder = ClipRecorder(clip_dir, pre_s, post_s)
        self._started_at = (
            datetime.now().astimezone()
        )  # t_s=0인 시각 (이벤트 occurred_at 계산용)
        self._frames = 0
        self._pool = ThreadPoolExecutor(max_workers=1)  # 일꾼 1명 → 순서 보장

    def start(self) -> int:
        """주행을 만들고 주행 번호를 돌려준다. 처음 보는 기기면 기기도 등록한다.

        사용처: /live 연결 직후 1번.
        입력: 없음
        출력: drive_id
        주의: 끝날 때까지 기다리므로 async 함수에서는 asyncio.to_thread로 부른다.
        """
        return self._pool.submit(self._create_drive).result()

    def submit(self, jpg: bytes, t_s: float, warnings: list[NearMissWarning]) -> None:
        """프레임 1장과 그 프레임의 경고를 일꾼에게 맡긴다. 기다리지 않고 바로 돌아온다.

        사용처: /live 수신 루프에서 매 프레임.
        입력: jpg 폰이 보낸 JPEG 바이트, t_s 촬영 시각(초, 주행 시작 기준),
              warnings 경고 판단 결과 (safe·caution이 섞여 있어도 됨, 여기서 거름)
        출력: 없음
        """
        future = self._pool.submit(self._handle, jpg, t_s, warnings)
        future.add_done_callback(_log_error)

    def close(self) -> None:
        """주행을 끝낸다. 모으던 클립을 있는 만큼 저장하고 주행 종료 시각·프레임 수를 적는다.

        사용처: /live 연결이 끊겼을 때 1번.
        입력: 없음
        출력: 없음
        주의: 맡겨 둔 프레임을 다 처리할 때까지 기다린다 (async에서는 asyncio.to_thread로).
        """
        self._pool.submit(self._finish).result()
        self._pool.shutdown()

    # ---------- 아래는 일꾼 스레드에서만 실행된다 ----------

    def _create_drive(self) -> int:
        self._session = self._session_factory()
        if self._session.get(Device, self.device_id) is None:
            self._session.add(Device(id=self.device_id))
        drive = Drive(device_id=self.device_id, source="live")
        self._session.add(drive)
        self._session.commit()
        self.drive_id = drive.id
        return drive.id

    def _handle(self, jpg: bytes, t_s: float, warnings: list[NearMissWarning]) -> None:
        # 경고(TTC 3초) 이상만 기록한다 (docs/SPEC.md). 같은 종류 5초 안은 record_event가 합친다
        for w in warnings:
            if w.level not in LEVEL_RANK:
                continue
            event, _ = record_event(
                self._session, self.drive_id, to_event_in(w, self._started_at)
            )
            # 합쳐진 경고면 같은 event_id라 클립 끝 시각만 늘어난다
            self._recorder.trigger(event.id, w.t_s, f"{self.drive_id}/{event.id}")
        for clip in self._recorder.push_jpeg(jpg, t_s):
            save_clip(self._session, clip)
        self._frames += 1

    def _finish(self) -> None:
        for clip in self._recorder.flush():
            save_clip(self._session, clip)
        drive = self._session.get(Drive, self.drive_id)
        drive.ended_at = datetime.now().astimezone()
        drive.frames = self._frames
        self._session.commit()
        self._session.close()


def _log_error(future: Future) -> None:
    """일꾼 스레드에서 난 오류를 로그로 남긴다 (그냥 두면 아무도 모르게 사라짐)."""
    if future.exception() is not None:
        logger.error("실시간 기록 실패", exc_info=future.exception())
