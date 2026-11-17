import { useCallback, useEffect, useState } from 'react'
import { fetchEvents } from './api.js'
import AppMenu from './components/AppMenu.jsx'
import ClipPlayer from './components/ClipPlayer.jsx'
import DriveAnalysis from './components/DriveAnalysis.jsx'
import LiveDrive from './components/LiveDrive.jsx'
import NearMissList from './components/NearMissList.jsx'
import VideoUpload from './components/VideoUpload.jsx'
import { PAGES } from './constants.js'

// 위쪽 줄(☰ + 화면 이름)과 지금 고른 화면 하나를 그린다 (라우터 라이브러리 없이 state 하나로).
export default function App() {
  const [page, setPage] = useState('drive')
  const [menuOpen, setMenuOpen] = useState(false)
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState(null)

  // useCallback: 아래 useEffect 의존성에 넣어도 렌더마다 다시 실행되지 않게 같은 함수를 유지한다
  const loadEvents = useCallback(() => {
    fetchEvents()
      .then(setEvents)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  // 내 위험 보기를 열 때마다 목록을 다시 받는다 (주행 중에 새로 저장된 아찔한 순간이 보이게)
  useEffect(() => {
    if (page === 'records') loadEvents()
  }, [page, loadEvents])

  function selectPage(key) {
    setPage(key)
    setMenuOpen(false)
  }

  return (
    <>
      <header className="topbar">
        <button className="menu-button" aria-label="메뉴" onClick={() => setMenuOpen(true)}>
          ☰
        </button>
        <span>{PAGES[page]}</span>
      </header>
      {menuOpen && <AppMenu page={page} onSelect={selectPage} onClose={() => setMenuOpen(false)} />}

      <main>
        {page === 'drive' && <LiveDrive />}
        {page === 'upload' && (
          <>
            <p className="muted">녹화된 블랙박스 영상을 올리면 서버가 분석해서 아찔한 순간을 찾아 줍니다.</p>
            <VideoUpload onShowResults={() => selectPage('records')} />
          </>
        )}
        {page === 'records' && (
          <>
            {selectedId !== null && (
              <ClipPlayer key={selectedId} eventId={selectedId} onClose={() => setSelectedId(null)} />
            )}
            {loading && <p className="muted">불러오는 중…</p>}
            {error && <p className="error">{error}</p>}
            {!loading && !error && (
              <NearMissList events={events} selectedId={selectedId} onSelect={setSelectedId} />
            )}
          </>
        )}
        {page === 'analysis' && <DriveAnalysis />}
        {page === 'rear' && (
          <p className="muted">후방 카메라(뒷차 급접근 확인)는 준비 중입니다.</p>
        )}
      </main>
    </>
  )
}
