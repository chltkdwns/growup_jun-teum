"""단계별 처리 시간 측정 도구 테스트: 가짜 모델로 process()와 같은 결과가 나오고 시간이 단계마다 잡히는지 확인한다."""
from app.pipeline import PerceptionPipeline
from tests.test_pipeline import CFG, _cfg_with, _FakeBrakeModel, _FakeTrackModel, _road
from tools.run_pipeline_timing import STAGES, brake_kind, process_timed


def test_process_timed_matches_process():
    """요약: 측정용 순서(process_timed)가 파이프라인 process()와 같은 앞차·브레이크등 결과를 내는지 (순서가 어긋나면 실패)."""
    cfg = _cfg_with("pipeline", brake_every_n_frames=2)
    ref = PerceptionPipeline(cfg, model=_FakeTrackModel(), brake_model=_FakeBrakeModel())
    timed = PerceptionPipeline(cfg, model=_FakeTrackModel(), brake_model=_FakeBrakeModel())
    for t_s in (0.0, 0.1, 0.2, 0.3):
        want = ref.process(_road(), t_s)
        got, _ = process_timed(timed, _road(), t_s)
        assert got["front"] == want["front"] and got["brake"] == want["brake"]
    assert ref.brake_model.calls == timed.brake_model.calls == 2


def test_process_timed_reports_every_stage():
    """요약: 단계별 ms가 모두 있고 0 이상이며, 전체는 단계 합과 같은지."""
    pipe = PerceptionPipeline(CFG, model=_FakeTrackModel(), brake_model=_FakeBrakeModel())
    _, ms = process_timed(pipe, _road(), 0.0)
    assert set(ms) == set(STAGES)
    assert all(v >= 0 for v in ms.values())
    assert abs(ms["total"] - sum(ms[s] for s in STAGES if s != "total")) < 1e-6


def test_brake_kind_labels():
    """요약: 켜짐/꺼짐/미등 처리/모름을 run_brake.py와 같은 기준으로 나누는지."""
    base = {"brake_score": 0.9, "third_on": False, "lamp_rose": False}
    assert brake_kind(None) == "unknown"
    assert brake_kind({**base, "brake_on": None, "model_on": False}) == "unknown"
    assert brake_kind({**base, "brake_on": True, "model_on": True}) == "on"
    assert brake_kind({**base, "brake_on": False, "model_on": True}) == "tail"
    assert brake_kind({**base, "brake_on": False, "model_on": False}) == "off"
