"""경고 입력 쪽 테스트 (DB 없이 돈다): 가짜 경고 시각, 경고 → 이벤트 모양 바꾸기."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.fake_warning import FakeWarningSource
from app.nearmiss import to_event_in
from app.schemas import NearMissWarning

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def test_fires_once_at_first_frame_past_time():
    """2초 경고는 2초를 처음 넘은 프레임에서 한 번만 나온다 (프레임 간격이 불규칙해도)."""
    judge = FakeWarningSource([2.0])
    assert judge(FRAME, 1.95) == []
    first = judge(FRAME, 2.03)
    assert len(first) == 1
    assert first[0].t_s == pytest.approx(2.03)
    assert first[0].level == "danger"
    assert judge(FRAME, 2.1) == []


def test_empty_list_means_off():
    judge = FakeWarningSource([])
    assert all(judge(FRAME, t / 10) == [] for t in range(100))


def test_to_event_in_maps_names():
    """판단 쪽 ttc_s가 DB 칸 min_ttc_s로, t_s가 시작 시각 + t_s로 바뀐다."""
    t0 = datetime(2026, 11, 16, 9, 0, 0, tzinfo=timezone.utc)
    w = NearMissWarning(
        t_s=12.5, type="cut_in", level="warning", ttc_s=2.7, distance_m=14.0
    )
    e = to_event_in(w, t0)
    assert e.occurred_at == t0 + timedelta(seconds=12.5)
    assert e.offset_s == 12.5
    assert e.min_ttc_s == 2.7
    assert e.distance_m == 14.0
    assert e.type == "cut_in"
