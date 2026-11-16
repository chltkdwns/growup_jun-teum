"""끼어들기 테스트: 차선과 박스, 촬영 시각을 직접 넣어 경계값을 확인한다."""
from app.lane.cut_in import CutInDetector

CFG = {"confirm_s": 0.15, "near_y_ratio": 0.7, "ignore_after_lane_change_s": 1.0}
H = 720
# 아래(y=612)에서 x 300~980, 위(y=432)에서 x 600~680 으로 모이는 내 차선
LANES = {"left": (300, 612, 600, 432), "right": (980, 612, 680, 432), "estimated": (), "lane_change": None}
NO_LANES = {"left": None, "right": None, "estimated": (), "lane_change": None}


def _det(x_c: float, y2: float, track_id: int, w: float = 120, h: float = 80) -> dict:
    """요약: 박스 아래 가운데가 (x_c, y2)인 가짜 탐지."""
    return {"track_id": track_id, "cls_name": "car", "score": 0.9,
            "box_xyxy_px": (x_c - w / 2, y2 - h, x_c + w / 2, y2), "t_s": 0.0}


FAR_FRONT = _det(640, 470, track_id=1, w=60, h=40)  # 멀리 있는 앞차 (y2=470)


def _run(steps: list, cfg: dict = CFG) -> list:
    """요약: (t_s, dets, front, lanes, lane_change) 목록을 차례로 넣고 결과 목록을 돌려준다."""
    det = CutInDetector(cfg)
    return [det.update(dets, lanes, front, t, H, lc) for t, dets, front, lanes, lc in steps]


def test_cut_in_from_left_confirmed_after_hold():
    """요약: 왼쪽 밖 → 안으로 들어와 앞차보다 가깝고 0.15초 유지되면 끼어들기 (경계 0.15초 포함)."""
    out_left = _det(250, 560, track_id=2)  # y=560에서 왼쪽 경계 x≈387 → 밖
    inside = _det(450, 560, track_id=2)
    # 시각은 부동소수점 오차가 경계를 넘지 않게 고름 (0.4 - 0.25 = 0.15000000000000002)
    res = _run([
        (0.0, [FAR_FRONT, out_left], FAR_FRONT, LANES, None),
        (0.25, [FAR_FRONT, inside], FAR_FRONT, LANES, None),  # 들어옴, 아직 0초
        (0.35, [FAR_FRONT, inside], FAR_FRONT, LANES, None),  # 0.1초 → 아직
        (0.4, [FAR_FRONT, inside], FAR_FRONT, LANES, None),   # 0.15초 → 확정
    ])
    assert res[:3] == [None, None, None]
    assert res[3] == {"track_id": 2, "side": "left", "t_s": 0.4}


def test_cut_in_from_right_side():
    """요약: 오른쪽 밖에서 들어오면 side는 right."""
    out_right = _det(1000, 560, track_id=3)
    inside = _det(850, 560, track_id=3)
    res = _run([(0.0, [out_right], None, LANES, None), (0.1, [inside], None, LANES, None),
                (0.3, [inside], None, LANES, None)])
    assert res[2]["side"] == "right"


def test_short_crossing_not_confirmed():
    """요약: 안으로 들어왔다가 0.15초 안에 다시 나가면 끼어들기 아님 (선 위 흔들림)."""
    out_left, inside = _det(250, 560, track_id=2), _det(450, 560, track_id=2)
    res = _run([(0.0, [out_left], None, LANES, None), (0.1, [inside], None, LANES, None),
                (0.2, [out_left], None, LANES, None), (0.4, [out_left], None, LANES, None)])
    assert res == [None, None, None, None]


def test_far_entry_behind_front_is_not_cut_in():
    """요약: 지금 앞차보다 멀리서(더 위에서) 들어온 차는 위험하지 않아 끼어들기 아님."""
    near_front = _det(640, 600, track_id=1)
    # y=470에서 내 차선은 x 약 537~743 → 520은 밖, 620은 안
    out_far, in_far = _det(520, 470, track_id=2, w=40, h=30), _det(620, 470, track_id=2, w=40, h=30)
    res = _run([(0.0, [near_front, out_far], near_front, LANES, None),
                (0.1, [near_front, in_far], near_front, LANES, None),
                (0.4, [near_front, in_far], near_front, LANES, None)])
    assert res == [None, None, None]


def test_no_front_uses_near_y_ratio():
    """요약: 앞차가 없으면 박스 아래가 화면 높이 0.7(504px) 아래여야 가까움."""
    # y=480에서 내 차선은 x 약 520~760 → 500은 밖, 620은 안
    far_out, far_in = _det(500, 480, track_id=4, w=40, h=30), _det(620, 480, track_id=4, w=40, h=30)
    res_far = _run([(0.0, [far_out], None, LANES, None), (0.1, [far_in], None, LANES, None),
                    (0.3, [far_in], None, LANES, None)])
    assert res_far[2] is None
    near_out, near_in = _det(250, 560, track_id=5), _det(450, 560, track_id=5)
    res_near = _run([(0.0, [near_out], None, LANES, None), (0.1, [near_in], None, LANES, None),
                     (0.3, [near_in], None, LANES, None)])
    assert res_near[2]["track_id"] == 5


def test_ignored_right_after_my_lane_change():
    """요약: 내 차로 변경 뒤 1초 동안은 옆 차가 경계를 넘어 보여도 끼어들기로 보지 않는다."""
    out_left, inside = _det(250, 560, track_id=2), _det(450, 560, track_id=2)
    res = _run([(0.0, [out_left], None, LANES, None), (0.05, [out_left], None, LANES, "right"),
                (0.1, [inside], None, LANES, None), (0.5, [inside], None, LANES, None),
                (1.5, [inside], None, LANES, None)])
    assert res == [None, None, None, None, None]


def test_no_lanes_no_judgement():
    """요약: 차선을 못 찾은 프레임은 판단하지 않는다."""
    res = _run([(0.0, [_det(250, 560, track_id=2)], None, NO_LANES, None),
                (0.3, [_det(450, 560, track_id=2)], None, NO_LANES, None)])
    assert res == [None, None]


def test_existing_front_car_wobble_not_cut_in():
    """요약: 이미 앞차였던 차가 차선 경계 흔들림으로 밖→안을 오가도 끼어들기가 아니다."""
    front_in, front_out = _det(450, 560, track_id=1), _det(250, 560, track_id=1)
    res = _run([
        (0.0, [front_in], front_in, LANES, None),
        (0.1, [front_out], front_out, LANES, None),  # 경계가 흔들려 잠깐 밖으로 보임 (앞차 트래커는 유지)
        (0.2, [front_in], front_in, LANES, None),
        (0.5, [front_in], front_in, LANES, None),
    ])
    assert res == [None, None, None, None]


def test_entry_with_no_front_still_detected_when_it_becomes_front():
    """요약: 앞차가 없을 때 들어온 차는 그 프레임에 바로 앞차가 돼도 끼어들기로 잡는다 (직전 프레임 앞차 기준)."""
    out_left, inside = _det(250, 560, track_id=6), _det(450, 560, track_id=6)
    res = _run([(0.0, [out_left], None, LANES, None),
                (0.1, [inside], inside, LANES, None),  # 들어온 프레임에 바로 앞차가 됨
                (0.3, [inside], inside, LANES, None)])
    assert res[2] is not None and res[2]["track_id"] == 6


def test_reported_once_per_entry():
    """요약: 같은 차는 한 번 알린 뒤 차선 밖으로 나갔다 다시 들어오기 전까지 다시 알리지 않는다."""
    out_left, inside = _det(250, 560, track_id=2), _det(450, 560, track_id=2)
    res = _run([(0.0, [out_left], None, LANES, None), (0.1, [inside], None, LANES, None),
                (0.3, [inside], None, LANES, None), (0.5, [inside], None, LANES, None)])
    assert res[2] is not None and res[3] is None
