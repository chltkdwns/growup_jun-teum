"""테스트용 설정 파일(tests/fixtures)의 키 구조가 기본 config.yaml과 같은지 확인한다 (값은 비교하지 않음).

기본 설정에 키를 추가하고 fixture를 안 고치면, fixture로 도구를 돌릴 때 KeyError가 나서 이 테스트로 미리 잡는다.
"""
from pathlib import Path

import yaml

BACKEND = Path(__file__).parents[1]
MAIN_CFG = BACKEND / "config.yaml"
FIXTURE_CFGS = sorted((BACKEND / "tests" / "fixtures").glob("config_*.yaml"))


def key_paths(node: dict, prefix: str = "") -> set[str]:
    """중첩된 설정 dict의 모든 키 경로를 모은다.

    사용처: 이 파일의 test_fixture_keys_match_main()
    입력: node 설정 dict(yaml.safe_load 결과), prefix 상위 키 경로 (예: "detect.")
    출력: 키 경로 집합 (예: {"detect", "detect.score_min", "detect.score_min_by_class.motorcycle"})
    """
    paths = set()
    for k, v in node.items():
        path = f"{prefix}{k}"
        paths.add(path)
        if isinstance(v, dict):
            paths |= key_paths(v, path + ".")
    return paths


def test_fixture_keys_match_main() -> None:
    """fixture마다 기본 설정과 키 경로 집합이 같아야 한다. 다르면 빠진 키·남는 키를 메시지에 보여 준다."""
    assert FIXTURE_CFGS, "tests/fixtures에 config_*.yaml이 없음"
    main_keys = key_paths(yaml.safe_load(MAIN_CFG.read_text(encoding="utf-8")))
    for path in FIXTURE_CFGS:
        fx_keys = key_paths(yaml.safe_load(path.read_text(encoding="utf-8")))
        missing, extra = sorted(main_keys - fx_keys), sorted(fx_keys - main_keys)
        assert not missing and not extra, f"{path.name}: 빠진 키 {missing}, 남는 키 {extra}"
