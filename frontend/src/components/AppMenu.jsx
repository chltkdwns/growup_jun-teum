import { PAGES } from '../constants.js'

// 왼쪽 위 ☰을 누르면 열리는 메뉴. 화면 이름은 constants.js PAGES 한곳에서 정한다.
// 바깥(어두운 부분)을 누르면 닫힌다.
export default function AppMenu({ page, onSelect, onClose }) {
  return (
    <div className="menu-backdrop" onClick={onClose}>
      <nav className="menu" onClick={(e) => e.stopPropagation()}>
        <div className="menu-title">틈</div>
        {Object.entries(PAGES).map(([key, label]) => (
          <button
            key={key}
            className={key === page ? 'active' : ''}
            onClick={() => onSelect(key)}
          >
            {label}
          </button>
        ))}
      </nav>
    </div>
  )
}
