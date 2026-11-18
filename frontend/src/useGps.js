import { useEffect, useState } from 'react'
import { GPS_MAX_ACCURACY_M } from './constants.js'

const EARTH_RADIUS_M = 6371000
const MS_TO_KMH = 3.6
const MS_PER_H = 3600000

// 두 위치 사이 거리(m). 지구를 공으로 보는 하버사인 공식 (주행 거리 정도는 오차가 무시할 만함)
function distanceM(a, b) {
  const rad = (deg) => (deg * Math.PI) / 180
  const dLat = rad(b.lat - a.lat)
  const dLon = rad(b.lon - a.lon)
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(h))
}

// 주행 중(active) 폰 GPS를 계속 받아 지금 위치·속도와 누적 거리·최고 속도를 돌려준다.
// 속도: 폰이 주는 속도(coords.speed, m/s)를 쓰고, 없으면(일부 기기·PC) 직전 위치와의 거리 ÷ 시간으로 구한다.
// 정확도가 GPS_MAX_ACCURACY_M보다 나쁜 위치는 거리·속도 계산에서 뺀다 (실내·출발 직후 위치 튐).
// 주의: 브라우저는 https나 localhost에서만 위치를 허용한다.
const EMPTY = { fix: null, distanceKm: 0, maxSpeedKmh: 0, error: null }
const SUPPORTED = 'geolocation' in navigator

export default function useGps(active) {
  const [gps, setGps] = useState(EMPTY)

  useEffect(() => {
    if (!active || !SUPPORTED) return
    let last = null // 직전의 정확한 위치 (거리 계산 기준)
    let first = true // 이번 주행의 첫 응답이면 지난 주행 값을 버리고 처음부터 센다
    const watchId = navigator.geolocation.watchPosition(
      (pos) => {
        const { latitude, longitude, accuracy, speed } = pos.coords
        const here = { lat: latitude, lon: longitude, t: pos.timestamp }
        const accurate = accuracy <= GPS_MAX_ACCURACY_M
        let stepM = 0
        let speedKmh = speed != null ? speed * MS_TO_KMH : null
        if (accurate && last) {
          stepM = distanceM(last, here)
          const dtH = (here.t - last.t) / MS_PER_H
          if (speedKmh == null && dtH > 0) speedKmh = stepM / 1000 / dtH
        }
        if (accurate) last = here
        const reset = first
        first = false
        setGps((g0) => {
          const g = reset ? EMPTY : g0
          return {
            fix: { lat: latitude, lon: longitude, accuracyM: accuracy, speedKmh },
            distanceKm: g.distanceKm + stepM / 1000,
            maxSpeedKmh:
              accurate && speedKmh != null ? Math.max(g.maxSpeedKmh, speedKmh) : g.maxSpeedKmh,
            error: null,
          }
        })
      },
      (e) => {
        const reset = first
        first = false
        setGps((g) => ({ ...(reset ? EMPTY : g), error: `위치를 받을 수 없습니다 (${e.message})` }))
      },
      { enableHighAccuracy: true }, // 휴대폰 GPS 칩 사용 (와이파이 위치보다 정확)
    )
    return () => navigator.geolocation.clearWatch(watchId)
  }, [active])

  return SUPPORTED ? gps : { ...EMPTY, error: '이 기기는 위치를 지원하지 않습니다' }
}
