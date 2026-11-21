import { useEffect, useRef, useState } from 'react'
import { liveSocketUrl } from '../api.js'
import {
  LEVEL_ORDER,
  LIVE_FRAME_ASPECT,
  LIVE_FRAME_WIDTH_PX,
  LIVE_JPEG_QUALITY,
  LIVE_MIN_FRAME_INTERVAL_MS,
} from '../constants.js'
import useGps from '../useGps.js'
import { playWarningSound, unlockSound } from '../warningSound.js'
import { topWarning } from '../warnings.js'
import BoxOverlay from './BoxOverlay.jsx'
import DriveStats from './DriveStats.jsx'
import GpsPanel from './GpsPanel.jsx'
import LaneOverlay from './LaneOverlay.jsx'
import WarningBadge from './WarningBadge.jsx'

// 이번 주행 분석의 처음 값
function emptyStats() {
  return {
    startedAt: null,
    endedAt: null,
    lastAt: null, // 마지막 서버 답을 받은 시각 (주행 시간 계산용)
    frames: 0,
    rttSumMs: 0,
    counts: { caution: 0, warning: 0, danger: 0 }, // 단계별로 '올라간' 횟수 (같은 경고가 이어지는 프레임은 1번)
    minTtcS: null,
    lastLevel: 'safe', // 직전 프레임의 가장 높은 단계
  }
}

// 서버 답 1개로 이번 주행 분석 값을 갱신한다 (새 객체를 돌려줌)
function updateStats(stats, warnings, rttMs) {
  const level = topWarning(warnings)?.level ?? 'safe'
  const counts = { ...stats.counts }
  // 직전보다 단계가 올라간 순간만 센다. 위험이 10프레임 이어져도 1회
  if (level !== 'safe' && LEVEL_ORDER.indexOf(level) > LEVEL_ORDER.indexOf(stats.lastLevel)) {
    counts[level] += 1
  }
  const ttcs = warnings.map((w) => w.ttc_s).filter((t) => t != null)
  const minTtcS = ttcs.length ? Math.min(stats.minTtcS ?? Infinity, ...ttcs) : stats.minTtcS
  return {
    ...stats,
    frames: stats.frames + 1,
    rttSumMs: stats.rttSumMs + rttMs,
    counts,
    minTtcS,
    lastLevel: level,
    lastAt: Date.now(),
  }
}

// 실시간 주행 화면: 폰 뒤 카메라 영상을 서버(/live)로 한 장씩 보내고, 돌아온 경고를 크게 보여 준다.
// 답을 받은 뒤에 다음 프레임을 보낸다. 그래서 서버는 늘 가장 최근 프레임만 처리하고, 밀린 프레임이 쌓이지 않는다.
// 아찔한 순간 기록(앞뒤 5초 클립)은 서버가 알아서 한다. 화면은 결과를 그리기만 한다.
export default function LiveDrive() {
  const videoRef = useRef(null)
  // 프레임 사이에 계속 쓰는 값 (바뀌어도 다시 그릴 필요가 없어서 state가 아니라 ref)
  const sessionRef = useRef(null) // { ws, stream, canvas, sentAt }
  const [running, setRunning] = useState(false)
  const [driveId, setDriveId] = useState(null)
  const [warnings, setWarnings] = useState([])
  const [overlay, setOverlay] = useState(null) // 서버가 보낸 박스·차선 { boxes, lanes, frameW, frameH } (보낸 프레임 기준 px)
  const lastLevelRef = useRef('safe') // 직전 프레임의 가장 높은 단계 (단계가 올라간 순간에만 소리 내려고)
  const [rttMs, setRttMs] = useState(null) // 프레임을 보내고 답을 받기까지 걸린 시간
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(emptyStats)
  const gps = useGps(running)

  function stop() {
    const s = sessionRef.current
    if (!s) return
    sessionRef.current = null
    s.ws.close()
    s.stream.getTracks().forEach((t) => t.stop()) // 카메라 끄기
    setRunning(false)
    setStats((st) => ({ ...st, endedAt: Date.now() })) // 끝난 뒤에도 분석 결과는 그대로 보여 준다
  }

  // 화면을 떠나면(탭 이동) 카메라와 연결을 정리한다
  useEffect(() => stop, [])

  // GPS 속도가 바뀌면 서버에 알린다. 서버는 이 값으로 안전거리(주의)를 판단한다 (모르면 null → 그 판단만 건너뜀)
  const speedKmh = gps.fix?.speedKmh ?? null
  useEffect(() => {
    const s = sessionRef.current
    if (!s || s.ws.readyState !== WebSocket.OPEN) return
    s.ws.send(JSON.stringify({ speed_kmh: speedKmh }))
  }, [speedKmh, driveId]) // driveId: 연결 직후(첫 메시지 받은 뒤)에도 한 번 보낸다

  // 지금 카메라 화면에서 화면에 보이는 가운데 16:9 부분만 잘라, 줄여서 JPEG로 보낸다.
  // 폰을 세로로 들면 카메라 사진이 세로로 길다. 그대로 보내면 차선 코드(가로 블랙박스 화면 가정)가
  // 화면에 안 보이는 아래쪽을 차선 영역으로 보고 엉뚱한 선을 잡는다. 보이는 것 = 분석하는 것으로 맞춘다.
  function sendFrame() {
    const s = sessionRef.current
    const video = videoRef.current
    if (!s || s.ws.readyState !== WebSocket.OPEN) return
    const vw = video.videoWidth
    const vh = video.videoHeight
    // 가운데 자르기 (CSS object-fit: cover와 같은 영역)
    const cropW = Math.min(vw, vh * LIVE_FRAME_ASPECT)
    const cropH = cropW / LIVE_FRAME_ASPECT
    s.canvas.width = LIVE_FRAME_WIDTH_PX
    s.canvas.height = Math.round(LIVE_FRAME_WIDTH_PX / LIVE_FRAME_ASPECT)
    s.canvas
      .getContext('2d')
      .drawImage(video, (vw - cropW) / 2, (vh - cropH) / 2, cropW, cropH, 0, 0, s.canvas.width, s.canvas.height)
    s.canvas.toBlob(
      (blob) => {
        if (sessionRef.current !== s || s.ws.readyState !== WebSocket.OPEN) return
        s.sentAt = performance.now()
        s.ws.send(blob)
      },
      'image/jpeg',
      LIVE_JPEG_QUALITY,
    )
  }

  async function start() {
    unlockSound() // 버튼을 누른 지금 열어 둬야 폰에서 나중에 경고음이 난다
    lastLevelRef.current = 'safe'
    setError(null)
    setWarnings([])
    setOverlay(null)
    setRttMs(null)
    setDriveId(null)
    setStats(emptyStats())
    let stream
    try {
      // facingMode environment: 폰 뒤 카메라 (PC는 있는 카메라)
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
    } catch (e) {
      setError(`카메라를 켤 수 없습니다 (${e.name})`)
      return
    }
    videoRef.current.srcObject = stream
    await videoRef.current.play()

    const ws = new WebSocket(liveSocketUrl())
    const s = { ws, stream, canvas: document.createElement('canvas'), sentAt: 0 }
    sessionRef.current = s
    setRunning(true)

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if ('drive_id' in msg) {
        // 연결 직후 첫 메시지: 주행 번호. 이제 첫 프레임을 보낸다
        setDriveId(msg.drive_id)
        setStats((st) => ({ ...st, startedAt: Date.now(), lastAt: Date.now() }))
        sendFrame()
        return
      }
      const rtt = performance.now() - s.sentAt
      setRttMs(rtt)
      setWarnings(msg.warnings)
      // 박스·차선 좌표는 방금 보낸 프레임(canvas) 크기 기준이다
      setOverlay({ boxes: msg.boxes, lanes: msg.lanes, frameW: s.canvas.width, frameH: s.canvas.height })
      const level = topWarning(msg.warnings)?.level ?? 'safe'
      if (LEVEL_ORDER.indexOf(level) > LEVEL_ORDER.indexOf(lastLevelRef.current)) playWarningSound(level)
      lastLevelRef.current = level
      setStats((st) => updateStats(st, msg.warnings, rtt))
      // 너무 자주 보내지 않게 최소 간격을 채운 뒤 다음 프레임
      setTimeout(sendFrame, Math.max(0, LIVE_MIN_FRAME_INTERVAL_MS - rtt))
    }
    ws.onerror = () => setError('서버에 연결할 수 없습니다')
    ws.onclose = () => {
      if (sessionRef.current === s) stop() // 서버 쪽에서 끊긴 경우
    }
  }

  return (
    <section className="live">
      <div className="live-view">
        <video ref={videoRef} muted playsInline />
        {running && overlay && <LaneOverlay {...overlay} />}
        {running && overlay && <BoxOverlay {...overlay} />}
        {running && <WarningBadge warnings={warnings} />}
        {!running && <p className="live-idle muted">주행을 시작하면 카메라가 켜집니다</p>}
      </div>
      {error && <p className="error">{error}</p>}
      <button className={`live-button${running ? ' stop' : ''}`} onClick={running ? stop : start}>
        {running ? '주행 끝내기' : '주행 시작'}
      </button>
      <div className="live-info muted">
        {driveId !== null && <span>주행 {driveId}</span>}
        {rttMs !== null && <span>응답 {Math.round(rttMs)}ms</span>}
      </div>
      <GpsPanel gps={gps} running={running} />
      <DriveStats stats={stats} gps={gps} />
    </section>
  )
}
