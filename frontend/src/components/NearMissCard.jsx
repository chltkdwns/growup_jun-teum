import { EVENT_TYPES, LEVELS } from '../constants.js'

// 아찔한 순간 1건을 카드로 그린다. 클립이 없으면 누를 수 없다.
export default function NearMissCard({ event, selected, onSelect }) {
  const level = LEVELS[event.level]
  const hasClip = event.clip_key !== null
  const time = new Date(event.occurred_at).toLocaleString('ko-KR')

  return (
    <button
      className={`card${selected ? ' selected' : ''}`}
      style={{ borderLeftColor: level.color }}
      disabled={!hasClip}
      onClick={() => onSelect(event.id)}
    >
      <div className="card-head">
        <span className="badge" style={{ background: level.color }}>
          {level.label}
        </span>
        <strong>{EVENT_TYPES[event.type] ?? event.type}</strong>
      </div>
      <div className="card-time">{time}</div>
      <div className="card-stats">
        {event.min_ttc_s !== null && <span>최소 TTC {event.min_ttc_s.toFixed(1)}초</span>}
        {event.distance_m !== null && <span>거리 {event.distance_m.toFixed(1)}m</span>}
        <span>영상 {event.offset_s.toFixed(1)}초 지점</span>
        {!hasClip && <span className="muted">클립 없음</span>}
      </div>
    </button>
  )
}
