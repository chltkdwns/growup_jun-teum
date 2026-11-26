// 경고 단계·이벤트 종류의 이름과 색은 이 파일 한곳에서만 정한다 (CLAUDE.md 규칙).
// 단계 키는 서버(schema.sql events.level)가 보내는 값 그대로다. 화면은 단계를 다시 계산하지 않는다.

export const API_URL = import.meta.env.VITE_API_URL
export const DEVICE_ID = import.meta.env.VITE_DEVICE_ID

// docs/SPEC.md "경고 단계": 주의 노랑 / 경고 주황 / 위험 빨강
export const LEVELS = {
  safe: { label: '안전', color: '#4b5563' }, // SPEC: 안전은 '기본' (따로 강조하지 않는 회색)
  caution: { label: '주의', color: '#eab308' },
  warning: { label: '경고', color: '#f97316' },
  danger: { label: '위험', color: '#dc2626' },
}

// schema.sql events.type 값 → 화면 이름
export const EVENT_TYPES = {
  lead: '앞차 급접근',
  cut_in: '끼어들기',
  brake: '앞차 브레이크',
  lane_departure: '차선 이탈',
  rear: '뒷차 급접근',
  two_wheeler: '이륜차',
}

// 영상 분석 상태 (서버 GET /drives/{id}/analysis의 status 값 그대로) → 화면 이름
export const ANALYSIS_STATUS = {
  queued: '차례를 기다리는 중',
  running: '분석 중',
  done: '분석 완료',
  failed: '분석 실패',
}

// 분석 진행 상황을 서버에 다시 묻는 간격 (ms). 분석은 영상 1분에 1분 넘게 걸려서 2초면 충분하다
export const ANALYSIS_POLL_MS = 2000

// 한 프레임에 경고가 여러 개면 화면에는 가장 높은 단계 하나를 크게 보여 준다 (단계는 서버가 정한 값 그대로)
export const LEVEL_ORDER = ['safe', 'caution', 'warning', 'danger']

// 실시간 주행: 서버로 보내는 프레임 크기와 화질. 원본(1080p)을 그대로 보내면 폰 데이터와 전송 시간이 커진다
export const LIVE_FRAME_WIDTH_PX = 640 // 가로 폭 (px)
// 보내는 프레임 가로:세로. 카메라 사진의 가운데를 이 비율로 잘라 보낸다 (폰 세로·가로 상관없이 블랙박스와 같은 비율).
// index.css .live-view의 aspect-ratio(16 / 9)와 같아야 화면에 보이는 부분 = 분석하는 부분이 된다
export const LIVE_FRAME_ASPECT = 16 / 9
export const LIVE_JPEG_QUALITY = 0.7 // JPEG 화질 (0~1)
// 프레임을 보내는 최소 간격 (ms). 서버 답이 빨라도 초당 약 15장을 넘기지 않게 해서 배터리·데이터를 아낀다
export const LIVE_MIN_FRAME_INTERVAL_MS = 66

// 화면 목록 (키 → 화면 이름). 첫 화면은 주행. 이름은 헤더 아래 제목에도 쓴다
export const PAGES = {
  drive: '주행',
  rear: '후방 확인',
  profile: '내 정보',
  records: '내 위험 보기',
  upload: '블랙박스 영상 분석',
  analysis: '주행 분석',
}
// 아래 고정 바에 둘 화면 (운전 중 바로 누르는 것)
export const BOTTOM_PAGES = ['drive', 'rear']
// ☰ 메뉴에 둘 화면 (주행 전후에 보는 것)
export const MENU_PAGES = ['profile', 'records', 'upload', 'analysis']

// GPS: 정확도(오차 반경)가 이보다 나쁜 위치는 거리·속도 계산에 쓰지 않는다 (m). 멈춰 있어도 위치가 튀어서 거리가 늘어나는 것 방지
export const GPS_MAX_ACCURACY_M = 30

// 주행 종류 (서버 drives.source 값) → 화면 이름
export const DRIVE_SOURCES = { live: '실시간', upload: '영상' }

// 카메라 위에 그리는 박스·차선 색 (서버 /live 답의 boxes·lanes를 그대로 그림)
export const OVERLAY_COLORS = {
  box: '#e5e7eb', // 다른 차 (옅은 회색)
  frontBox: '#22d3ee', // 앞차 (경고 색과 겹치지 않는 하늘색)
  lane: '#22c55e', // 내 차선 좌우 선
}
export const OVERLAY_LINE_PX = { box: 2, frontBox: 4, lane: 4 } // 선 굵기 (보낸 프레임 기준 px)

// docs/SPEC.md "경고 단계": 경고 = 경고음, 위험 = 강한 경고음 (주의는 소리 없음). 단계가 올라간 순간에 한 번 울린다
export const WARNING_SOUNDS = {
  warning: { freqHz: 880, beeps: 1 },
  danger: { freqHz: 1320, beeps: 3 },
}
export const BEEP_MS = 150 // 삐 한 번 길이 (ms)
export const BEEP_GAP_MS = 80 // 삐 사이 간격 (ms)
