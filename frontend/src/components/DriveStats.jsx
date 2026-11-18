import { LEVELS } from '../constants.js'

const MS_PER_H = 3600000

// 이번 주행 분석: 시간·거리·속도(GPS), 경고 횟수·최저 TTC(서버 결과), 응답 시간·처리 속도(성능).
// 숫자는 LiveDrive·useGps가 모아서 넘겨주고, 여기서는 보기 좋게 그리기만 한다.
export default function DriveStats({ stats, gps }) {
  // 주행 중이면 마지막 서버 답까지, 끝났으면 끝난 시각까지
  const elapsedMs = stats.startedAt ? (stats.endedAt ?? stats.lastAt) - stats.startedAt : 0
  const min = Math.floor(elapsedMs / 60000)
  const sec = Math.floor((elapsedMs % 60000) / 1000)
  const avgSpeed = elapsedMs > 0 ? gps.distanceKm / (elapsedMs / MS_PER_H) : 0
  const avgRtt = stats.frames > 0 ? stats.rttSumMs / stats.frames : null
  const fps = elapsedMs > 0 ? stats.frames / (elapsedMs / 1000) : 0

  return (
    <div className="panel">
      <h2>이번 주행 분석</h2>
      <div className="stats">
        <div><span>주행 시간</span>{min}분 {sec}초</div>
        <div><span>이동 거리</span>{gps.distanceKm.toFixed(2)} km</div>
        <div><span>평균 속도</span>{Math.round(avgSpeed)} km/h</div>
        <div><span>최고 속도</span>{Math.round(gps.maxSpeedKmh)} km/h</div>
        {['caution', 'warning', 'danger'].map((lv) => (
          <div key={lv}>
            <span style={{ color: LEVELS[lv].color }}>{LEVELS[lv].label}</span>
            {stats.counts[lv]}회
          </div>
        ))}
        <div><span>가장 낮은 TTC</span>{stats.minTtcS != null ? `${stats.minTtcS.toFixed(1)}초` : '-'}</div>
        <div><span>평균 응답</span>{avgRtt != null ? `${Math.round(avgRtt)} ms` : '-'}</div>
        <div><span>처리 속도</span>{fps.toFixed(1)} 장/초</div>
      </div>
    </div>
  )
}
