// 경고 단계·이벤트 종류의 이름과 색은 이 파일 한곳에서만 정한다 (CLAUDE.md 규칙).
// 단계 키는 서버(schema.sql events.level)가 보내는 값 그대로다. 화면은 단계를 다시 계산하지 않는다.

export const API_URL = import.meta.env.VITE_API_URL
export const DEVICE_ID = import.meta.env.VITE_DEVICE_ID

// docs/SPEC.md "경고 단계": 주의 노랑 / 경고 주황 / 위험 빨강
export const LEVELS = {
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
