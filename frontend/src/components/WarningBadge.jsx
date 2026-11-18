import { EVENT_TYPES, LEVELS } from '../constants.js'
import { topWarning } from '../warnings.js'

// 카메라 오른쪽 위: 이번 프레임의 가장 높은 경고 단계를 색과 함께 그린다. 경고가 없으면 "안전".
export default function WarningBadge({ warnings }) {
  const top = topWarning(warnings)
  const level = LEVELS[top?.level ?? 'safe']

  return (
    <div className="warning-badge" style={{ background: level.color }}>
      <strong>{level.label}</strong>
      {top && <span>{EVENT_TYPES[top.type] ?? top.type}</span>}
      {top?.ttc_s != null && <span>TTC {top.ttc_s.toFixed(1)}초</span>}
    </div>
  )
}
