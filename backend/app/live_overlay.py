"""/live 답에 붙일 화면 표시용 값(차 박스·차선)을 만든다 (FastAPI를 import하지 않아 DB 없이 테스트할 수 있다)."""


def live_overlay(result: dict | None) -> dict:
    """분석 결과에서 앱이 카메라 위에 그릴 차 박스와 차선만 골라 JSON으로 보낼 모양으로 바꾼다.

    사용처: 5단계(표시). main.py /live가 프레임마다 답에 붙인다.
    입력: result PerceptionPipeline.process() 결과 (PipelineWarningSource.last_result, 아직 없으면 None)
    출력: {"boxes": [{"track_id", "cls_name", "box_xyxy_px": [x1, y1, x2, y2], "is_front"}],
           "lanes": {"left": [x_bottom, y_bottom, x_top, y_top] 또는 None, "right": 같음}}
    주의: 좌표는 폰이 보낸 프레임 기준(px)이라, 앱이 화면 크기에 맞게 늘려서 그린다.
          numpy 숫자는 JSON으로 못 보내서 float·int로 바꾼다.
    """
    if result is None:
        return {"boxes": [], "lanes": {"left": None, "right": None}}
    front = result["front"]
    front_id = None if front is None else front["track_id"]
    boxes = [
        {
            "track_id": d["track_id"],
            "cls_name": d["cls_name"],
            "box_xyxy_px": [float(v) for v in d["box_xyxy_px"]],
            "is_front": d["track_id"] is not None and d["track_id"] == front_id,
        }
        for d in result["dets"]
    ]
    lanes = {}
    for side in ("left", "right"):
        line = result["lanes"][side]
        lanes[side] = None if line is None else [int(v) for v in line]
    return {"boxes": boxes, "lanes": lanes}
