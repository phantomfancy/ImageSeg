import assert from 'node:assert/strict'

import { createLatestFrameGate } from '../src/app/latestFrameGate.ts'

{
  const gate = createLatestFrameGate()

  assert.equal(gate.requestRun(), true)
  assert.equal(gate.hasInFlightRun(), true)

  assert.equal(gate.requestRun(), false)
  assert.equal(gate.requestRun(), false)
  assert.equal(gate.hasInFlightRun(), true)

  assert.equal(gate.finishRun(), true)
  assert.equal(gate.hasInFlightRun(), true)

  assert.equal(gate.finishRun(), false)
  assert.equal(gate.hasInFlightRun(), false)
}

{
  const gate = createLatestFrameGate()

  assert.equal(gate.requestRun(), true)
  assert.equal(gate.requestRun(), false)

  gate.reset()
  assert.equal(gate.hasInFlightRun(), false)

  assert.equal(gate.requestRun(), true)
  assert.equal(gate.finishRun(), false)
}

console.log('frontend latest frame gate tests passed')
