import assert from 'node:assert/strict'

import {
  decodeDetections,
  resolveLabelsFromMetadata,
  resolveModelContract,
  toLabelRecord,
  toTensorDescriptors,
  type TensorLike,
} from '../src/index.ts'

const hfContract = resolveModelContract({
  inputs: toTensorDescriptors([
    { name: 'pixel_values', shape: [1, 3, 640, 640] },
  ]),
  outputs: toTensorDescriptors([
    { name: 'logits', shape: [1, 2, 2] },
    { name: 'pred_boxes', shape: [1, 2, 4] },
  ]),
  displayName: 'hf-rtdetr',
})

assert.equal(hfContract.family, 'hf-detr-like')

const ultralyticsYoloContract = resolveModelContract({
  inputs: toTensorDescriptors([
    { name: 'images', shape: [1, 3, 640, 640] },
  ]),
  outputs: toTensorDescriptors([
    { name: 'output0', shape: [1, 6, 20] },
  ]),
  displayName: 'yolo-detect',
})

assert.equal(ultralyticsYoloContract.family, 'ultralytics-yolo-detect')

const ultralyticsRtDetrContract = resolveModelContract({
  inputs: toTensorDescriptors([
    { name: 'images', shape: [1, 3, 640, 640] },
  ]),
  outputs: toTensorDescriptors([
    { name: 'output0', shape: [1, 300, 84] },
  ]),
  displayName: 'rtdetr-v2',
})

assert.equal(ultralyticsRtDetrContract.family, 'ultralytics-rtdetr')

assert.deepEqual(resolveLabelsFromMetadata({
  names: '{"0":"himars","1":"radar"}',
}), {
  0: 'himars',
  1: 'radar',
})

assert.deepEqual(toLabelRecord("{0: 'Aircraft', 1: 'uav'}"), {
  0: 'Aircraft',
  1: 'uav',
})

assert.deepEqual(resolveLabelsFromMetadata({
  id2label: {
    0: 'himars',
    1: 'radar',
  },
}), {
  0: 'himars',
  1: 'radar',
})

const output0: TensorLike = {
  dims: [1, 6, 2],
  data: new Float32Array([
    0.5, 0.25,
    0.5, 0.25,
    0.2, 0.1,
    0.2, 0.1,
    0.9, 0.1,
    0.2, 0.8,
  ]),
}

const detections = decodeDetections(
  resolveModelContract({
    inputs: toTensorDescriptors([
      { name: 'images', shape: [1, 3, 640, 640] },
    ]),
    outputs: toTensorDescriptors([
      { name: 'output0', shape: [1, 6, 2] },
    ]),
    manifest: {
      ...ultralyticsYoloContract,
      family: 'ultralytics-yolo-detect',
      schemaVersion: '2026-04-19',
    },
  }),
  { output0 },
  640,
  640,
)

assert.equal(detections.length, 2)
assert.ok(detections[0].confidence >= detections[1].confidence)

console.log('contracts tests passed')
