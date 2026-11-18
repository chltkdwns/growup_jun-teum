import { MENU_PAGES, PAGES } from '../constants.js'

// 오른쪽 위 ☰을 누르면 오른쪽에서 열리는 메뉴. 넣을 화면은 constants.js MENU_PAGES 한곳에서 정한다.
// 바깥(어두운 부분)을 누르면 닫힌다.
export default function AppMenu({ page, onSelect, onClose }) {
  return (
    <div className="menu-backdrop" onClick={onClose}>
      <nav className="menu" onClick={(e) => e.stopPropagation()}>
        <div className="menu-title">틈</div>
        {MENU_PAGES.map((key) => (
          <button
            key={key}
            className={key === page ? 'active' : ''}
            onClick={() => onSelect(key)}
          >
            {PAGES[key]}
          </button>
        ))}
      </nav>
    </div>
  )
}
