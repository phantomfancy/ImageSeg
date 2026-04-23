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
assert.deepEqual(hfContract.runtimeHints.preferredExecutionProviders, ['webgpu'])
assert.equal(hfContract.decoder.nmsIouThreshold, null)

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
assert.equal(ultralyticsYoloContract.decoder.scoreThreshold, 0.35)
assert.equal(ultralyticsYoloContract.decoder.nmsIouThreshold, 0.45)

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
assert.equal(ultralyticsRtDetrContract.decoder.nmsIouThreshold, null)

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
  {
    sourceWidth: 640,
    sourceHeight: 640,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'stretch',
    scaleX: 1,
    scaleY: 1,
    padLeft: 0,
    padTop: 0,
    contentWidth: 640,
    contentHeight: 640,
  },
)

assert.equal(detections.length, 2)
assert.ok(detections[0].confidence >= detections[1].confidence)

const unlimitedDetections = decodeDetections(
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
      decoder: {
        ...ultralyticsYoloContract.decoder,
        maxDetections: 0,
      },
    },
  }),
  { output0 },
  {
    sourceWidth: 640,
    sourceHeight: 640,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'stretch',
    scaleX: 1,
    scaleY: 1,
    padLeft: 0,
    padTop: 0,
    contentWidth: 640,
    contentHeight: 640,
  },
)

assert.equal(unlimitedDetections.length, 2)

const limitedDetections = decodeDetections(
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
      decoder: {
        ...ultralyticsYoloContract.decoder,
        maxDetections: 1,
      },
    },
  }),
  { output0 },
  {
    sourceWidth: 640,
    sourceHeight: 640,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'stretch',
    scaleX: 1,
    scaleY: 1,
    padLeft: 0,
    padTop: 0,
    contentWidth: 640,
    contentHeight: 640,
  },
)

assert.equal(limitedDetections.length, 1)
assert.equal(limitedDetections[0].confidence, detections[0].confidence)

const yoloWithOverlap: TensorLike = {
  dims: [1, 6, 3],
  data: new Float32Array([
    0.50, 0.51, 0.20,
    0.50, 0.50, 0.20,
    0.20, 0.20, 0.08,
    0.20, 0.20, 0.08,
    0.95, 0.94, 0.91,
    0.10, 0.10, 0.10,
  ]),
}

const nmsDetections = decodeDetections(
  ultralyticsYoloContract,
  { output0: yoloWithOverlap },
  {
    sourceWidth: 640,
    sourceHeight: 640,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'stretch',
    scaleX: 1,
    scaleY: 1,
    padLeft: 0,
    padTop: 0,
    contentWidth: 640,
    contentHeight: 640,
  },
)

assert.equal(nmsDetections.length, 2)
assert.ok(nmsDetections[0].confidence > nmsDetections[1].confidence)

const nmsLimitedDetections = decodeDetections(
  resolveModelContract({
    inputs: toTensorDescriptors([
      { name: 'images', shape: [1, 3, 640, 640] },
    ]),
    outputs: toTensorDescriptors([
      { name: 'output0', shape: [1, 6, 3] },
    ]),
    manifest: {
      ...ultralyticsYoloContract,
      family: 'ultralytics-yolo-detect',
      schemaVersion: '2026-04-19',
      decoder: {
        ...ultralyticsYoloContract.decoder,
        maxDetections: 1,
      },
    },
  }),
  { output0: yoloWithOverlap },
  {
    sourceWidth: 640,
    sourceHeight: 640,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'stretch',
    scaleX: 1,
    scaleY: 1,
    padLeft: 0,
    padTop: 0,
    contentWidth: 640,
    contentHeight: 640,
  },
)

assert.equal(nmsLimitedDetections.length, 1)
assert.equal(nmsLimitedDetections[0].confidence, nmsDetections[0].confidence)

const stretchDetections = decodeDetections(
  hfContract,
  {
    logits: {
      dims: [1, 1, 1],
      data: new Float32Array([4]),
    },
    pred_boxes: {
      dims: [1, 1, 4],
      data: new Float32Array([0.5, 0.5, 0.25, 0.25]),
    },
  },
  {
    sourceWidth: 320,
    sourceHeight: 160,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'stretch',
    scaleX: 2,
    scaleY: 4,
    padLeft: 0,
    padTop: 0,
    contentWidth: 640,
    contentHeight: 640,
  },
)

assert.equal(stretchDetections.length, 1)
assert.deepEqual(stretchDetections[0].box, {
  x: 120,
  y: 60,
  width: 80,
  height: 40,
})

const padDetections = decodeDetections(
  hfContract,
  {
    logits: {
      dims: [1, 1, 1],
      data: new Float32Array([4]),
    },
    pred_boxes: {
      dims: [1, 1, 4],
      data: new Float32Array([0.5, 0.5, 0.25, 0.25]),
    },
  },
  {
    sourceWidth: 320,
    sourceHeight: 160,
    targetWidth: 640,
    targetHeight: 640,
    resizeMode: 'pad',
    scaleX: 2,
    scaleY: 2,
    padLeft: 0,
    padTop: 160,
    contentWidth: 640,
    contentHeight: 320,
  },
)

assert.equal(padDetections.length, 1)
assert.deepEqual(padDetections[0].box, {
  x: 120,
  y: 40,
  width: 80,
  height: 80,
})

console.log('contracts tests passed')
