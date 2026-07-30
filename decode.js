// Pure decoding for the packed pointcloud format — no three.js, no DOM.
// Kept separate from pointcloud.js so it can be unit-tested under Node
// (see tools/test_decode.mjs).
//
// One block, after gunzip:
//   offset 0      float32[3]  lo    quantization lower corner (world, metres)
//   offset 12     float32[3]  hi    quantization upper corner
//   offset 24     int16[n*3]  xyz   mapped lo..hi -> -32768..32767
//   offset 24+6n  uint8[n*3]  rgb

export async function inflate(bytes) {
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function decodeBlock(raw) {
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const lo = [dv.getFloat32(0, true), dv.getFloat32(4, true), dv.getFloat32(8, true)];
  const hi = [dv.getFloat32(12, true), dv.getFloat32(16, true), dv.getFloat32(20, true)];
  const n = Math.floor((raw.byteLength - 24) / 9);   // 6 bytes xyz + 3 bytes rgb
  if (n <= 0) {
    return { pos: new Float32Array(0), col: new Float32Array(0), n: 0, lo, hi };
  }

  // The gunzipped buffer is not guaranteed to be 2-byte aligned for the Int16
  // view, so copy the positions out rather than aliasing in place.
  const qBytes = raw.slice(24, 24 + n * 6);
  const q = new Int16Array(qBytes.buffer, qBytes.byteOffset, n * 3);
  const rgb = raw.subarray(24 + n * 6, 24 + n * 9);

  const pos = new Float32Array(n * 3);
  const sx = (hi[0] - lo[0]) / 65535;
  const sy = (hi[1] - lo[1]) / 65535;
  const sz = (hi[2] - lo[2]) / 65535;
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    pos[j]     = lo[0] + (q[j]     + 32768) * sx;
    pos[j + 1] = lo[1] + (q[j + 1] + 32768) * sy;
    pos[j + 2] = lo[2] + (q[j + 2] + 32768) * sz;
  }

  const col = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) col[i] = rgb[i] / 255;

  return { pos, col, n, lo, hi };
}

/** Split a multi-block byte range into its constituent decoded blocks. */
export async function decodeRange(buf, blocks, start, end, byteStart) {
  const out = [];
  for (let i = start; i < end; i++) {
    const rel = blocks[i].o - byteStart;
    try {
      out.push(decodeBlock(await inflate(buf.subarray(rel, rel + blocks[i].c))));
    } catch {
      out.push(null);
    }
  }
  return out;
}
