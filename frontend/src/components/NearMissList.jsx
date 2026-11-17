import NearMissCard from './NearMissCard.jsx'

// 아찔한 순간 카드 목록을 그린다. 데이터는 App이 받아서 넘겨준다.
export default function NearMissList({ events, selectedId, onSelect }) {
  if (events.length === 0) return <p className="muted">기록된 아찔한 순간이 없습니다.</p>

  return (
    <div className="list">
      {events.map((e) => (
        <NearMissCard key={e.id} event={e} selected={e.id === selectedId} onSelect={onSelect} />
      ))}
    </div>
  )
}
