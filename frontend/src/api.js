import { API_URL, DEVICE_ID } from './constants.js'

// 로그인 대신 모든 요청에 이 기기의 UUID를 헤더로 붙인다.
// 서버는 이 헤더의 기기 것만 보여주고, 남의 기록은 404로 막는다.
const DEVICE_HEADERS = { 'X-Device-Id': DEVICE_ID }

// 이 기기의 아찔한 순간 목록 (서버가 최근 것부터 정렬해서 준다)
export async function fetchEvents() {
  const res = await fetch(`${API_URL}/events`, { headers: DEVICE_HEADERS })
  if (!res.ok) throw new Error(`목록 요청 실패 (${res.status})`)
  return res.json()
}

// 클립 영상을 받아 Blob으로 돌려준다.
// <video src>는 헤더를 못 붙이므로, 먼저 받아서 URL.createObjectURL로 재생한다 (클립은 10초라 작음).
export async function fetchClip(eventId) {
  const res = await fetch(`${API_URL}/events/${eventId}/clip`, { headers: DEVICE_HEADERS })
  if (!res.ok) throw new Error(`클립 요청 실패 (${res.status})`)
  return res.blob()
}
