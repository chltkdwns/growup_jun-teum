# SafeGap — 브레이크등 · 깜빡이 라벨링

테스트 영상 속 앞차마다 브레이크등과 깜빡이 상태를 라벨링하고, 공개 데이터셋 TLD(약 15만 장)와 합쳐 브레이크등 모델을 학습하는 프로젝트.
TLD에 없는 한국 도로·야간 장면을 우리 라벨로 보완하는 것이 목적이다.

- 이번 모델은 **브레이크등만** 학습한다. 깜빡이 라벨은 다음 단계(여러 프레임 모델)용으로 모아 둔다.
- 모델 구조 가정: 기존 YOLO로 차 검출 → 차 이미지를 잘라 상태 분류.

## 구조

```
backend/   FastAPI + 라벨링 도구(tools/) + 테스트
front/     React + Vite
data/      원본 영상 (커밋 금지)
outputs/   프레임·라벨 결과 (커밋 금지)
```

## 실행

```bash
cp .env.example .env
```

```bash
cd backend && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

```bash
cd backend && uvicorn app.main:app --reload
```

```bash
cd front && npm install && npm run dev
```

테스트: `cd backend && pytest`

## 라벨링 요약

차 한 대에 박스 1개 + 속성 2개. 등 자체에는 박스를 그리지 않는다. 기본값은 `brake = off, turn = off`.
애매하면 가이드 판단 규칙을 따르고, 그래도 모르면 `unknown`. 박스 폭 40px 미만, 등이 가려짐, 옆면만 보임, 마주 오는 차는 `unknown`.
자세한 규칙은 `SafeGap_최종_역할분담.pdf` 참고.

## 데이터 규칙

- CVAT 클라우드에는 우리가 직접 찍은 영상만 올린다. KITTI·AI Hub·TLD 원본은 올리지 않는다.
- 프레임·라벨 파일은 `outputs/` 아래에 두고 커밋하지 않는다.
- 발표·데모에 라벨 화면을 넣을 때는 번호판과 얼굴을 흐림 처리한다.
- TLD는 받기 전에 Hugging Face License 태그를 확인한다.
