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

// 블랙박스 영상을 올린다. 서버는 분석을 뒤에서 시작하고 바로 {drive_id, status: "queued"}를 준다.
// FormData를 쓰면 브라우저가 multipart 헤더(경계값 포함)를 알아서 붙인다 (Content-Type을 직접 쓰면 안 됨).
export async function uploadVideo(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_URL}/drives/upload`, {
    method: 'POST',
    headers: DEVICE_HEADERS,
    body: form,
  })
  if (!res.ok) throw new Error(`업로드 실패 (${res.status})`)
  return res.json()
}

// 올린 영상의 분석 진행 상황 {status, frames_done, frames_total, events, error}
export async function fetchAnalysis(driveId) {
  const res = await fetch(`${API_URL}/drives/${driveId}/analysis`, { headers: DEVICE_HEADERS })
  if (!res.ok) throw new Error(`진행 상황 요청 실패 (${res.status})`)
  return res.json()
}

// 실시간 주행 WebSocket 주소. 브라우저 WebSocket은 헤더를 못 붙여서 기기 UUID를 주소(query)로 보낸다
export function liveSocketUrl() {
  return `${API_URL.replace(/^http/, 'ws')}/live?device_id=${DEVICE_ID}`
}

// 이 기기의 주행 목록과 주행별 요약 {id, source, started_at, duration_s, events, danger_events, worst_ttc_s ...}
export async function fetchDrives() {
  const res = await fetch(`${API_URL}/drives`, { headers: DEVICE_HEADERS })
  if (!res.ok) throw new Error(`주행 목록 요청 실패 (${res.status})`)
  return res.json()
}
