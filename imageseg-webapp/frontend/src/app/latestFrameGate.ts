export type LatestFrameGate = {
  finishRun: () => boolean
  hasInFlightRun: () => boolean
  requestRun: () => boolean
  reset: () => void
}

export function createLatestFrameGate(): LatestFrameGate {
  let hasPendingRun = false
  let hasInFlightRun = false

  return {
    finishRun() {
      if (hasPendingRun) {
        hasPendingRun = false
        hasInFlightRun = true
        return true
      }

      hasInFlightRun = false
      return false
    },
    hasInFlightRun() {
      return hasInFlightRun
    },
    requestRun() {
      if (hasInFlightRun) {
        hasPendingRun = true
        return false
      }

      hasInFlightRun = true
      return true
    },
    reset() {
      hasPendingRun = false
      hasInFlightRun = false
    },
  }
}
