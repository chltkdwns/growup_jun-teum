import { useEffect, useState } from 'react'
import { fetchDrives, fetchEvents } from '../api.js'
import { DRIVE_SOURCES, LEVELS } from '../constants.js'

// 주행 분석 화면: 전체 합계 + 주행별 카드(시간, 아찔한 순간 수, 최저 TTC, 언제 일어났는지 시간 줄).
// 요약 숫자는 서버 GET /drives가 계산하고, 시간 줄의 점은 GET /events의 offset_s(주행 시작부터 몇 초)로 찍는다.
export default function DriveAnalysis() {
  const [drives, setDrives] = useState(null)
  const [eventsByDrive, setEventsByDrive] = useState({})
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([fetchDrives(), fetchEvents()])
      .then(([ds, es]) => {
        const grouped = {}
        for (const e of es) (grouped[e.drive_id] ??= []).push(e)
        setDrives(ds)
        setEventsByDrive(grouped)
      })
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <p className="error">{error}</p>
  if (drives === null) return <p className="muted">불러오는 중…</p>
  if (drives.length === 0) return <p className="muted">아직 주행 기록이 없습니다.</p>

  const total = drives.reduce((n, d) => n + d.events, 0)
  const danger = drives.reduce((n, d) => n + d.danger_events, 0)

  return (
    <>
      <div className="panel stats">
        <div><span>주행</span>{drives.length}회</div>
        <div><span>아찔한 순간</span>{total}건</div>
        <div><span style={{ color: LEVELS.danger.color }}>위험</span>{danger}건</div>
        {/* 이동 거리는 GPS를 서버에 보내기 시작해야 쌓인다 (아직 형식 합의 전) */}
        <div><span>100km당</span>거리 기록 전</div>
      </div>
      {drives.map((d) => (
        <DriveCard key={d.id} drive={d} events={eventsByDrive[d.id] ?? []} />
      ))}
    </>
  )
}

// 주행 1건 카드
function DriveCard({ drive, events }) {
  const minutes = drive.duration_s != null ? Math.max(1, Math.round(drive.duration_s / 60)) : null
  return (
    <div className="panel">
      <h2>
        {new Date(drive.started_at).toLocaleString('ko-KR')} · {DRIVE_SOURCES[drive.source] ?? drive.source}
        {drive.video_name && ` · ${drive.video_name}`}
      </h2>
      <div className="stats">
        <div><span>주행 시간</span>{minutes != null ? `${minutes}분` : '-'}</div>
        <div><span>아찔한 순간</span>{drive.events}건 (위험 {drive.danger_events})</div>
        <div><span>가장 낮은 TTC</span>{drive.worst_ttc_s != null ? `${drive.worst_ttc_s.toFixed(1)}초` : '-'}</div>
      </div>
      {/* 실시간은 주행 길이를 알아서 시간 줄로, 영상은 길이를 몰라서 'N초 지점' 목록으로 보여 준다 */}
      {events.length > 0 && drive.duration_s != null && (
        <div className="timeline">
          {events.map((e) => (
            <span
              key={e.id}
              style={{
                left: `${Math.min(100, (e.offset_s / drive.duration_s) * 100)}%`,
                background: LEVELS[e.level].color,
              }}
            />
          ))}
        </div>
      )}
      {events.length > 0 && drive.duration_s == null && (
        <div className="muted">
          {events
            .map((e) => `${e.offset_s.toFixed(1)}초 ${LEVELS[e.level].label}`)
            .join(' · ')}
        </div>
      )}
    </div>
  )
}
