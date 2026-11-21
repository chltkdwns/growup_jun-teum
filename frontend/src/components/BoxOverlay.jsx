import { OVERLAY_COLORS, OVERLAY_LINE_PX } from '../constants.js'

// 카메라 위: 서버가 찾은 차 박스를 그린다. 앞차(is_front)는 굵은 하늘색.
// 좌표는 폰이 보낸 프레임 기준(px)이라 viewBox를 그 크기로 두고, 화면에는 비디오와 같은 방식(slice = object-fit: cover)으로 늘린다.
export default function BoxOverlay({ boxes, frameW, frameH }) {
  return (
    <svg className="overlay" viewBox={`0 0 ${frameW} ${frameH}`} preserveAspectRatio="xMidYMid slice">
      {boxes.map((b, i) => {
        const [x1, y1, x2, y2] = b.box_xyxy_px
        const kind = b.is_front ? 'frontBox' : 'box'
        return (
          <rect
            key={b.track_id ?? `i${i}`}
            x={x1}
            y={y1}
            width={x2 - x1}
            height={y2 - y1}
            fill="none"
            stroke={OVERLAY_COLORS[kind]}
            strokeWidth={OVERLAY_LINE_PX[kind]}
          />
        )
      })}
    </svg>
  )
}
