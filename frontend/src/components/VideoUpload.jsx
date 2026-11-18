import { useEffect, useState } from 'react'
import { fetchAnalysis, uploadVideo } from '../api.js'
import { ANALYSIS_POLL_MS, ANALYSIS_STATUS } from '../constants.js'

// 블랙박스 영상을 골라 올리고, 서버 분석이 끝날 때까지 진행 상황을 보여 준다.
// 분석은 서버가 뒤에서 하므로, 올린 뒤에는 ANALYSIS_POLL_MS마다 상태를 물어본다.
// 끝나면 '아찔한 순간 보기' 버튼으로 내 위험 보기 화면에 간다(onShowResults).
export default function VideoUpload({ onShowResults }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [driveId, setDriveId] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)

  const finished = analysis?.status === 'done' || analysis?.status === 'failed'

  // 주행 번호가 생기면 끝날 때까지 상태를 묻는다
  useEffect(() => {
    if (driveId === null || finished) return
    const timer = setInterval(() => {
      fetchAnalysis(driveId)
        .then(setAnalysis)
        .catch((e) => setError(e.message))
    }, ANALYSIS_POLL_MS)
    return () => clearInterval(timer)
  }, [driveId, finished])

  async function handleUpload() {
    setUploading(true)
    setError(null)
    setAnalysis(null)
    setDriveId(null)
    try {
      const res = await uploadVideo(file)
      setAnalysis({ status: res.status, frames_done: 0, frames_total: null, events: 0 })
      setDriveId(res.drive_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const busy = uploading || (driveId !== null && !finished)
  // 전체 프레임 수를 모르면(코덱에 따라) 막대 없이 처리한 프레임 수만 보여 준다
  const percent =
    analysis?.frames_total ? Math.min(100, (analysis.frames_done / analysis.frames_total) * 100) : null

  return (
    <section className="upload">
      <input
        type="file"
        accept="video/*,.avi"
        disabled={busy}
        onChange={(e) => setFile(e.target.files[0] ?? null)}
      />
      <button disabled={!file || busy} onClick={handleUpload}>
        {uploading ? '올리는 중…' : '영상 분석'}
      </button>

      {analysis && (
        <div className="upload-status">
          <div>
            {ANALYSIS_STATUS[analysis.status] ?? analysis.status}
            {analysis.status !== 'queued' && ` · ${analysis.frames_done}프레임`}
            {` · 아찔한 순간 ${analysis.events}건`}
          </div>
          {percent !== null && (
            <div className="progress">
              <div style={{ width: `${percent}%` }} />
            </div>
          )}
          {analysis.status === 'failed' && <p className="error">{analysis.error}</p>}
          {analysis.status === 'done' && (
            <button className="results-button" onClick={onShowResults}>
              아찔한 순간 보기
            </button>
          )}
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  )
}
