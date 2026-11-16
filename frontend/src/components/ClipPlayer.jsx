import { useEffect, useState } from 'react'
import { fetchClip } from '../api.js'

// 고른 이벤트의 앞뒤 5초 클립을 재생한다.
// 기기 헤더를 붙여 클립을 받은 뒤 임시 주소(blob:)로 재생하고, 닫거나 바꾸면 임시 주소를 지운다.
// App이 key={eventId}로 그려서, 다른 카드를 누르면 이 컴포넌트가 새로 만들어진다 (상태 초기화).
export default function ClipPlayer({ eventId, onClose }) {
  const [src, setSrc] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let url = null
    let cancelled = false
    fetchClip(eventId)
      .then((blob) => {
        if (cancelled) return
        url = URL.createObjectURL(blob)
        setSrc(url)
      })
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [eventId])

  return (
    <div className="player">
      {error && <p className="error">{error}</p>}
      {!error && !src && <p className="muted">클립 불러오는 중…</p>}
      {src && <video src={src} controls autoPlay playsInline />}
      <button className="close" onClick={onClose}>닫기</button>
    </div>
  )
}
