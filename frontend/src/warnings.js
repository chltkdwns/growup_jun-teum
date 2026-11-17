import { LEVEL_ORDER } from './constants.js'

// 한 프레임의 경고 목록에서 가장 높은 단계의 경고 1개 (없으면 null).
// 단계는 서버가 정한 값 그대로 비교만 한다 (화면에서 TTC로 단계를 다시 계산하지 않음).
export function topWarning(warnings) {
  return warnings.reduce(
    (best, w) => (LEVEL_ORDER.indexOf(w.level) > LEVEL_ORDER.indexOf(best?.level) ? w : best),
    null,
  )
}
