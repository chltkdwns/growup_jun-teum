import { BEEP_GAP_MS, BEEP_MS, WARNING_SOUNDS } from './constants.js'

// 경고음: 브라우저 기본 AudioContext로 삐 소리를 만든다 (소리 파일·라이브러리 없음).
// 폰 브라우저는 사용자가 누른 순간에 만든 AudioContext만 소리를 내서, 주행 시작 버튼에서 unlockSound()를 먼저 부른다.
let ctx = null

export function unlockSound() {
  ctx ??= new AudioContext()
  ctx.resume()
}

// 단계에 맞는 소리를 한 번 낸다. 소리가 없는 단계(안전·주의)거나 아직 unlock 전이면 아무것도 안 한다
export function playWarningSound(level) {
  const sound = WARNING_SOUNDS[level]
  if (!sound || !ctx) return
  for (let i = 0; i < sound.beeps; i++) {
    const start = ctx.currentTime + (i * (BEEP_MS + BEEP_GAP_MS)) / 1000
    const osc = ctx.createOscillator()
    osc.frequency.value = sound.freqHz
    osc.connect(ctx.destination)
    osc.start(start)
    osc.stop(start + BEEP_MS / 1000)
  }
}
