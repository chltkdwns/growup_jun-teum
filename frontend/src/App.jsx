import { useEffect, useState } from 'react'
import { fetchEvents } from './api.js'
import ClipPlayer from './components/ClipPlayer.jsx'
import NearMissList from './components/NearMissList.jsx'

// 아찔한 순간 목록 화면: 처음 열 때 목록을 한 번 받고, 카드를 누르면 위에서 클립을 재생한다.
export default function App() {
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState(null)

  useEffect(() => {
    fetchEvents()
      .then(setEvents)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <main>
      <h1>아찔한 순간</h1>
      {selectedId !== null && (
        <ClipPlayer key={selectedId} eventId={selectedId} onClose={() => setSelectedId(null)} />
      )}
      {loading && <p className="muted">불러오는 중…</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && (
        <NearMissList events={events} selectedId={selectedId} onSelect={setSelectedId} />
      )}
    </main>
  )
}
