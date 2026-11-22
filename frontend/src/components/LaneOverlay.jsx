import { OVERLAY_COLORS, OVERLAY_LINE_PX } from '../constants.js'

// 카메라 위: 서버가 찾은 내 차선의 좌·우 선을 그린다. 못 찾은 쪽(null)은 그리지 않는다.
// 선 하나 = [x_bottom, y_bottom, x_top, y_top] (보낸 프레임 기준 px). 늘리는 방식은 BoxOverlay와 같다.
export default function LaneOverlay({ lanes, frameW, frameH }) {
  const lines = [lanes.left, lanes.right].filter((l) => l != null)
  return (
    <svg className="overlay" viewBox={`0 0 ${frameW} ${frameH}`} preserveAspectRatio="xMidYMid slice">
      {lines.map(([xb, yb, xt, yt], i) => (
        <line
          key={i}
          x1={xb}
          y1={yb}
          x2={xt}
          y2={yt}
          stroke={OVERLAY_COLORS.lane}
          strokeWidth={OVERLAY_LINE_PX.lane}
          strokeLinecap="round"
        />
      ))}
    </svg>
  )
}
