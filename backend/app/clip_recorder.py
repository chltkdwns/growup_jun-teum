"""아찔한 순간 클립 저장. 최근 pre_s초 프레임을 링 버퍼에 들고 있다가, 경고가 오면 앞뒤를 이어 저장한다.

영상 파일 모드와 실시간 모드가 같은 코드를 쓴다. 차이는 프레임이 파일에서 오느냐, 폰에서 오느냐뿐이다.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

# 브라우저에서 재생되도록 H.264로 저장한다. OpenCV는 윈도우/리눅스마다 H.264 지원이 달라서
# ffmpeg가 들어 있는 imageio-ffmpeg로 쓴다
CODEC = "libx264"
# faststart: 파일을 다 받기 전에 재생을 시작할 수 있게 정보를 앞에 둔다
FFMPEG_OUTPUT_PARAMS = ["-movflags", "+faststart"]
# 버퍼에는 JPEG로 압축해 넣는다. 1080p 원본 5초(30fps)는 약 900MB라 메모리가 부족하다
JPEG_QUALITY = 90


@dataclass
class _Recording:
    """저장 중인 클립 1개 (이벤트 1개)."""

    event_id: int
    path: Path
    end_s: float  # 이 촬영 시각까지 모으면 저장
    trigger_s: float
    frames: list[tuple[float, bytes]] = field(default_factory=list)  # (촬영 시각, JPEG)


@dataclass
class SavedClip:
    """저장이 끝난 클립 정보. clips 표에 그대로 넣는다."""

    event_id: int
    path: Path
    pre_s: float  # 실제로 담긴 경고 전 시간 (영상 앞부분이면 설정값보다 짧음)
    post_s: float
    fps: float
    width: int
    height: int


class ClipRecorder:
    """최근 프레임을 버퍼에 두고, 경고 시 앞뒤 클립을 만든다.

    사용처: 5단계(기록). 영상 분석 루프나 실시간 WebSocket이 매 프레임 push(), 경고 이벤트가
            저장될 때 trigger()를 부른다.
    주의: 시간은 프레임 수가 아니라 촬영 시각(초) 기준이다. 프레임은 같은 크기여야 한다.
    """

    def __init__(self, clip_dir: Path, pre_s: float, post_s: float) -> None:
        self.clip_dir = clip_dir
        self.pre_s = pre_s
        self.post_s = post_s
        self._buffer: deque[tuple[float, bytes]] = deque()  # (촬영 시각, JPEG)
        self._active: dict[int, _Recording] = {}

    def trigger(self, event_id: int, t_s: float, name: str) -> None:
        """경고 이벤트가 생기거나 이어질 때 부른다.

        사용처: create_event로 이벤트를 저장(또는 합침)한 직후.
        입력: event_id DB 이벤트 번호, t_s 경고 촬영 시각(초), name 파일 이름(확장자 제외, 예: "6/7")
        출력: 없음
        주의: 같은 event_id로 다시 부르면(합쳐진 경고) 저장 끝 시각만 뒤로 늘린다.
        """
        rec = self._active.get(event_id)
        if rec is not None:
            rec.end_s = t_s + self.post_s
            return
        rec = _Recording(
            event_id=event_id,
            path=self.clip_dir / f"{name}.mp4",
            end_s=t_s + self.post_s,
            trigger_s=t_s,
            # 경고 전 pre_s초를 버퍼에서 복사 (버퍼는 직전 프레임 기준이라 한 번 더 자른다)
            frames=[(ft, f) for ft, f in self._buffer if t_s - ft <= self.pre_s],
        )
        self._active[event_id] = rec

    def push(self, frame: np.ndarray, t_s: float) -> list[SavedClip]:
        """프레임 1장을 넣는다. 다 모인 클립이 있으면 저장하고 돌려준다.

        사용처: 매 프레임.
        입력: frame BGR 이미지(H×W×3), t_s 촬영 시각(초, 영상 시작 또는 주행 시작 기준)
        출력: 이번에 저장이 끝난 클립 목록 (대부분 빈 목록)
        """
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        self._buffer.append((t_s, jpg.tobytes()))
        # 버퍼에는 최근 pre_s초만 남긴다 (링 버퍼)
        while t_s - self._buffer[0][0] > self.pre_s:
            self._buffer.popleft()

        saved = []
        for event_id in list(self._active):
            rec = self._active[event_id]
            rec.frames.append(self._buffer[-1])
            if t_s >= rec.end_s:
                saved.append(self._write(rec))
                del self._active[event_id]
        return saved

    def flush(self) -> list[SavedClip]:
        """영상이 끝났을 때 아직 모으던 클립을 있는 만큼 저장한다.

        사용처: 영상 파일 분석 끝, 실시간 주행 종료.
        입력: 없음
        출력: 저장한 클립 목록
        """
        saved = [self._write(rec) for rec in self._active.values()]
        self._active.clear()
        return saved

    def _write(self, rec: _Recording) -> SavedClip:
        """모은 프레임을 mp4 파일로 쓴다."""
        first_s, last_s = rec.frames[0][0], rec.frames[-1][0]
        span_s = last_s - first_s
        # fps는 실제 촬영 시각으로 계산한다 (실시간은 프레임 간격이 일정하지 않음)
        fps = (len(rec.frames) - 1) / span_s if span_s > 0 else 1.0
        height, width = self._decode(rec.frames[0][1]).shape[:2]

        rec.path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(rec.path),
            (width, height),
            pix_fmt_in="bgr24",  # OpenCV 프레임은 BGR 순서
            fps=fps,
            codec=CODEC,
            macro_block_size=2,  # 크기를 16의 배수로 바꾸지 않게 (짝수만 필요)
            output_params=FFMPEG_OUTPUT_PARAMS,
        )
        writer.send(None)  # 제너레이터 시작
        # 한 장씩 풀어서 쓴다 (전부 풀어 두면 다시 메모리가 부족해진다)
        for _, jpg in rec.frames:
            writer.send(self._decode(jpg))
        writer.close()

        return SavedClip(
            event_id=rec.event_id,
            path=rec.path,
            pre_s=rec.trigger_s - first_s,
            post_s=last_s - rec.trigger_s,
            fps=fps,
            width=width,
            height=height,
        )

    @staticmethod
    def _decode(jpg: bytes) -> np.ndarray:
        """버퍼의 JPEG를 BGR 이미지로 되돌린다."""
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
