// 4D pointcloud view. Owns a Three.js scene but not a clock — the page's
// Conductor tells it which frame to show, so it stays in lockstep with the
// videos.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { decodeRange } from './decode.js';

// Consecutive frame blocks are contiguous in cloud_{preset}.bin, so one HTTP
// Range request covers many frames. This matters: a request per frame would
// hit the Hub's unauthenticated resolve limit (3000 per 300 s) during playback.
const CHUNK_FRAMES = 16;
const MAX_CACHED_CHUNKS = 24;

// Corner ordering matches step3a-2's bbox_world:
//   0:(+x+y+z) 1:(+x+y-z) 2:(+x-y-z) 3:(+x-y+z)
//   4:(-x+y+z) 5:(-x+y-z) 6:(-x-y-z) 7:(-x-y+z)
const BOX_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

export class PointCloudView {
  constructor(host, { hfBase, eid, onStatus = () => {} }) {
    this.host = host;
    this.hfBase = hfBase;
    this.eid = eid;
    this.onStatus = onStatus;

    this.index = null;
    this.chunks = new Map();
    this.chunkOrder = [];
    this.busy = false;
    this.wanted = null;
    this.current = null;
    this.framedOnce = false;
    this.showBox = true;
    this.showPoints = true;
    this.showCamBoxes = false;
    this.showFrustums = false;
    this.cameraGeometry = null;
    this.perCam = null;      // { cams, consensus[], register, boxes:{cam:[8x3|null]} }
    this.pointSize = 2.0;

    this._initScene();
  }

  // -- scene ---------------------------------------------------------------
  _initScene() {
    const host = this.host;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1114);

    const w = host.clientWidth || 640;
    const h = host.clientHeight || 360;
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.01, 100);
    this.camera.up.set(0, 0, 1);          // world is the Franka base frame: Z up
    this.camera.position.set(1.2, -1.2, 0.9);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    host.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    const grid = new THREE.GridHelper(2, 20, 0x3a4050, 0x23272f);
    grid.rotation.x = Math.PI / 2;        // GridHelper is XZ by default; we want XY
    this.scene.add(grid);
    this.scene.add(new THREE.AxesHelper(0.15));

    this.geom = new THREE.BufferGeometry();
    this.geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    this.geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
    this.mat = new THREE.PointsMaterial({ size: this.pointSize, sizeAttenuation: false, vertexColors: true });
    this.points = new THREE.Points(this.geom, this.mat);
    this.scene.add(this.points);

    const mkLines = (color, width) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(BOX_EDGES.length * 6), 3));
      const o = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, linewidth: width }));
      o.visible = false;
      this.scene.add(o);
      return o;
    };

    // Fused consensus box (orange, the project's box colour).
    this.boxLines = mkLines(0xffa500);
    // One box per camera hypothesis; colour is set per frame from consensus.
    this.camBoxLines = { wrist: mkLines(0x22cc55), ext1: mkLines(0x22cc55), ext2: mkLines(0x22cc55) };

    // Camera frustums, in the project's per-cam colours (red/green/blue).
    // 8 segments each: 4 from the apex to the image corners, 4 around the rim.
    const mkFrustum = color => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(8 * 6), 3));
      const o = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
      o.visible = false;
      this.scene.add(o);
      return o;
    };
    this.frustums = { wrist: mkFrustum(0xff4444), ext1: mkFrustum(0x44cc44), ext2: mkFrustum(0x4488ff) };

    new ResizeObserver(() => {
      const cw = host.clientWidth, ch = host.clientHeight;
      if (!cw || !ch) return;
      this.camera.aspect = cw / ch;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(cw, ch);
    }).observe(host);

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  // -- data ----------------------------------------------------------------
  async load(preset) {
    const url = `${this.hfBase}/clouds/${this.eid}/index_${preset}.json`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`index_${preset}.json — HTTP ${r.status}`);
    this.index = await r.json();
    this.preset = preset;
    this.chunks.clear();
    this.chunkOrder.length = 0;
    this.current = null;
    return this.index;
  }

  get avgBlockBytes() {
    if (!this.index) return 0;
    const b = this.index.blocks;
    return b.reduce((s, x) => s + x.c, 0) / Math.max(b.length, 1);
  }

  _chunkId(i) { return Math.floor(i / CHUNK_FRAMES); }

  _fetchChunk(chunkId) {
    if (this.chunks.has(chunkId)) return this.chunks.get(chunkId);
    const blocks = this.index.blocks;
    const start = chunkId * CHUNK_FRAMES;
    if (start >= blocks.length || start < 0) return Promise.resolve([]);
    const end = Math.min(start + CHUNK_FRAMES, blocks.length);
    const byteStart = blocks[start].o;
    const byteEnd = blocks[end - 1].o + blocks[end - 1].c - 1;
    const url = `${this.hfBase}/clouds/${this.eid}/${this.index.bin}`;

    const p = fetch(url, { headers: { Range: `bytes=${byteStart}-${byteEnd}` } })
      .then(async res => {
        if (res.status !== 206 && res.status !== 200) throw new Error(`cloud bin — HTTP ${res.status}`);
        const buf = new Uint8Array(await res.arrayBuffer());
        return decodeRange(buf, blocks, start, end, byteStart);
      })
      .catch(err => { this.chunks.delete(chunkId); throw err; });

    this.chunks.set(chunkId, p);
    this.chunkOrder.push(chunkId);
    while (this.chunkOrder.length > MAX_CACHED_CHUNKS) {
      this.chunks.delete(this.chunkOrder.shift());
    }
    return p;
  }

  /** Frame index for a given source frame `t`, or -1. */
  blockIndexForFrame(t) {
    if (!this.index) return -1;
    return this.index.blocks.findIndex(b => b.t === t);
  }

  /**
   * Show a frame. Called at animation rate, so if a fetch/decode is still in
   * flight we record the latest request and pick it up when free — dropping
   * intermediate frames rather than queueing up behind the clock.
   */
  async show(blockIdx) {
    if (!this.index) return;
    const n = this.index.blocks.length;
    const idx = Math.min(Math.max(blockIdx, 0), n - 1);
    if (idx === this.current) return;
    this.wanted = idx;
    if (this.busy) return;

    this.busy = true;
    try {
      while (this.wanted !== null && this.wanted !== this.current) {
        const target = this.wanted;
        const cid = this._chunkId(target);
        const cached = this.chunks.has(cid);
        if (!cached) this.onStatus('buffering');
        let chunk;
        try {
          chunk = await this._fetchChunk(cid);
        } catch (e) {
          this.onStatus('error');
          break;
        }
        if (!cached) this.onStatus('');
        const payload = chunk[target - cid * CHUNK_FRAMES];
        if (payload) {
          this._render(payload, target);
          this.current = target;
        } else {
          this.current = target;   // nothing to draw for this frame; don't spin
        }
        // Warm the next chunk so playback doesn't stall at the boundary.
        this._fetchChunk(this._chunkId(Math.min(target + CHUNK_FRAMES, n - 1)));
        if (this.wanted === target) this.wanted = null;
      }
    } finally {
      this.busy = false;
    }
  }

  _frameMeta(blockIdx) {
    const t = this.index.blocks[blockIdx].t;
    return (this.index.frames || []).find(f => f.t === t) || {};
  }

  _render(payload, blockIdx) {
    this.geom.setAttribute('position', new THREE.BufferAttribute(payload.pos, 3));
    this.geom.setAttribute('color', new THREE.BufferAttribute(payload.col, 3));
    this.geom.computeBoundingSphere();
    this.lastPayload = payload;

    const meta = this._frameMeta(blockIdx);
    const box = meta.bbox_world;
    if (this.showBox && Array.isArray(box) && box.length === 8) {
      const arr = this.boxLines.geometry.attributes.position.array;
      BOX_EDGES.forEach(([a, b], k) => { arr.set(box[a], k * 6); arr.set(box[b], k * 6 + 3); });
      this.boxLines.geometry.attributes.position.needsUpdate = true;
      this.boxLines.geometry.computeBoundingSphere();
      this.boxLines.visible = true;
    } else {
      this.boxLines.visible = false;
    }

    this._renderCamBoxes(this.index.blocks[blockIdx].t);
    this._renderFrustums(this.index.blocks[blockIdx].t);

    if (!this.framedOnce) {
      if (Array.isArray(box) && box.length === 8) this._frameToBox(box);
      else this._frameTo(payload.lo, payload.hi);
      this.framedOnce = true;
    }
    this.onStatus('', { points: payload.n, meta, camBoxes: this._camBoxState(this.index.blocks[blockIdx].t) });
  }

  _camBoxState(t) {
    if (!this.perCam) return null;
    const consensus = this.perCam.consensus?.[t] ?? [];
    const edgeMax = this.perCam.thresholds?.edge ?? 0.15;
    const out = {};
    for (const cam of this.perCam.cams) {
      const b = this.perCam.boxes?.[cam]?.[t];
      if (!b) { out[cam] = 'none'; continue; }
      if (consensus.includes(cam)) { out[cam] = 'agree'; continue; }
      // Both draw red, but "didn't meet the edge threshold" is a different
      // failure from "tracked confidently and still disagreed".
      const e = this.perCam.edges?.[cam]?.[t];
      out[cam] = (e == null || e > edgeMax) ? 'lowconf' : 'disagree';
    }
    const regCam = this.perCam.register_frames?.[String(t)];
    return { states: out, register: regCam || null,
             status: this.perCam.frame_status?.[t] ?? null };
  }

  /** Draw one box per camera: green where that view agreed with the fused
   *  pose, red where it dissented. Cameras with no pose for this frame are
   *  hidden rather than drawn somewhere arbitrary. */
  /** Draw one box per camera, in three states — not two.
   *
   *  The pipeline gates on confidence *before* it compares poses: a camera with
   *  edge > threshold is dropped from the vote entirely, and the frame status
   *  `agree{X}of{Y}` counts Y = confident cameras, not cameras that produced a
   *  pose. So a low-confidence camera was never "deemed disagreement" — it
   *  abstained. Painting it red claimed the pipeline rejected a box it never
   *  actually considered.
   *
   *    green   in the agreeing clique
   *    red     confident but outside the clique -> genuine disagreement
   *    grey    produced a pose, below the confidence bar -> abstained
   *    hidden  no pose at all
   */
  /** Camera frustums. Intrinsics give the field of view, poses place them; the
   *  two externals are static, the wrist moves so its pose is per frame. */
  _renderFrustums(t) {
    const g = this.cameraGeometry;
    const on = this.showFrustums && g;
    const DEPTH = 0.15;   // metres out from the apex
    for (const cam of Object.keys(this.frustums)) {
      const obj = this.frustums[cam];
      if (!on) { obj.visible = false; continue; }
      const K = g.K?.[cam];
      const hw = g.image_hw?.[cam];
      const T = cam === 'wrist' ? g.wrist_T_c2w?.[t] : g.T_c2w?.[cam];
      if (!K || !hw || !T) { obj.visible = false; continue; }

      const [H, W] = hw;
      const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
      const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) => {
        const x = (u - cx) / fx * DEPTH;
        const y = (v - cy) / fy * DEPTH;
        // camera -> world
        return [
          T[0][0] * x + T[0][1] * y + T[0][2] * DEPTH + T[0][3],
          T[1][0] * x + T[1][1] * y + T[1][2] * DEPTH + T[1][3],
          T[2][0] * x + T[2][1] * y + T[2][2] * DEPTH + T[2][3],
        ];
      });
      const apex = [T[0][3], T[1][3], T[2][3]];

      const arr = obj.geometry.attributes.position.array;
      let k = 0;
      for (const c of corners) { arr.set(apex, k); arr.set(c, k + 3); k += 6; }
      for (let i = 0; i < 4; i++) {
        arr.set(corners[i], k); arr.set(corners[(i + 1) % 4], k + 3); k += 6;
      }
      obj.geometry.attributes.position.needsUpdate = true;
      obj.geometry.computeBoundingSphere();
      obj.visible = true;
    }
  }

  _renderCamBoxes(t) {
    const on = this.showCamBoxes && this.perCam;
    const edgeMax = this.perCam?.thresholds?.edge_metric ?? 0.15;
    for (const cam of Object.keys(this.camBoxLines)) {
      const obj = this.camBoxLines[cam];
      if (!on) { obj.visible = false; continue; }
      const corners = this.perCam.boxes?.[cam]?.[t];
      if (!corners) { obj.visible = false; continue; }
      const arr = obj.geometry.attributes.position.array;
      BOX_EDGES.forEach(([a, b], k) => { arr.set(corners[a], k * 6); arr.set(corners[b], k * 6 + 3); });
      obj.geometry.attributes.position.needsUpdate = true;
      obj.geometry.computeBoundingSphere();

      const inClique = (this.perCam.consensus?.[t] ?? []).includes(cam);
      const e = this.perCam.edges?.[cam]?.[t];
      const confident = e != null && e <= edgeMax;
      if (inClique) {
        obj.material.color.setHex(0x22cc55);
        obj.material.opacity = 1.0;
      } else if (confident) {
        obj.material.color.setHex(0xff3b30);
        obj.material.opacity = 1.0;
      } else {
        obj.material.color.setHex(0x8b93a1);      // abstained, not rejected
        obj.material.opacity = 0.5;
      }
      obj.material.transparent = obj.material.opacity < 1;
      obj.visible = true;
    }
  }

  _frameTo(lo, hi) {
    const c = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.5);
    this.controls.target.set(c[0], c[1], c[2]);
    this.camera.position.set(c[0] + span * 0.9, c[1] - span * 0.9, c[2] + span * 0.7);
    this.camera.near = Math.max(span / 500, 1e-3);
    this.camera.far = span * 30;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /** Centre on the tracked object — the full cloud spans metres, which would
   *  leave the object a few pixels across. */
  _frameToBox(corners) {
    const c = [0, 1, 2].map(k => corners.reduce((s, p) => s + p[k], 0) / 8);
    const pad = 0.55;
    this._frameTo(c.map(v => v - pad), c.map(v => v + pad));
  }

  setPointSize(s) { this.pointSize = s; this.mat.size = s; }

  setShowPoints(v) { this.showPoints = v; this.points.visible = v; }

  setShowBox(v) {
    this.showBox = v;
    if (this.current !== null && this.lastPayload) this._render(this.lastPayload, this.current);
  }

  setShowCamBoxes(v) {
    this.showCamBoxes = v;
    if (this.current !== null && this.index) this._renderCamBoxes(this.index.blocks[this.current].t);
  }

  setPerCam(data) {
    this.perCam = data;
    if (this.current !== null && this.index) this._renderCamBoxes(this.index.blocks[this.current].t);
  }

  setCameraGeometry(g) {
    this.cameraGeometry = g;
    if (this.current !== null && this.index) this._renderFrustums(this.index.blocks[this.current].t);
  }

  setShowFrustums(v) {
    this.showFrustums = v;
    if (this.current !== null && this.index) this._renderFrustums(this.index.blocks[this.current].t);
    else if (!v) for (const o of Object.values(this.frustums)) o.visible = false;
  }

  resetView() {
    this.framedOnce = false;
    if (this.lastPayload && this.current !== null) this._render(this.lastPayload, this.current);
  }
}
