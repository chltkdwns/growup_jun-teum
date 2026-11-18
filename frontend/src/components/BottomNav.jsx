import { BOTTOM_PAGES, PAGES } from '../constants.js'

// 화면 아래에 고정된 바. 스크롤해도 사라지지 않아서 운전 중에도 주행·후방 확인을 바로 오간다.
// 넣을 화면은 constants.js BOTTOM_PAGES 한곳에서 정한다.
export default function BottomNav({ page, onSelect }) {
  return (
    <nav className="bottom-nav">
      {BOTTOM_PAGES.map((key) => (
        <button key={key} className={key === page ? 'active' : ''} onClick={() => onSelect(key)}>
          {PAGES[key]}
        </button>
      ))}
    </nav>
  )
}
