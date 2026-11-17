// 카메라 아래: 지금 속도(GPS)와 위치를 그린다. 값은 useGps가 계산해서 넘겨준다.
export default function GpsPanel({ gps, running }) {
  const fix = gps.fix
  return (
    <div className="panel">
      <h2>GPS</h2>
      {!running && <p className="muted">주행을 시작하면 위치를 받습니다</p>}
      {running && gps.error && <p className="error">{gps.error}</p>}
      {running && !gps.error && !fix && <p className="muted">위치 찾는 중…</p>}
      {running && fix && (
        <>
          <div className="speed">
            {fix.speedKmh != null ? Math.round(fix.speedKmh) : '-'} <small>km/h</small>
          </div>
          <div className="muted">
            {fix.lat.toFixed(5)}, {fix.lon.toFixed(5)} · 오차 {Math.round(fix.accuracyM)}m
          </div>
        </>
      )}
    </div>
  )
}
