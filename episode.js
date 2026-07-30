// Episode page controller: builds each pipeline-stage section and wires every
// time-varying element to a single Conductor.

import { Conductor } from './sync.js';
import { PointCloudView } from './pointcloud.js';

const HF_REPO = 'Silicon23/droid_pipeline_visualization';
const HF_BASE = `https://huggingface.co/datasets/${HF_REPO}/resolve/main`;

const CAMS = ['wrist', 'ext1', 'ext2'];
const CAM_COLOR = { wrist: 'var(--wrist)', ext1: 'var(--ext1)', ext2: 'var(--ext2)' };

const $ = id => document.getElementById(id);
const EID = new URLSearchParams(location.search).get('eid');

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const fmt = n => (n === null || n === undefined) ? '—' : n.toLocaleString();
const num = (v, d = 3) => (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(d) : '—';

// ---------------------------------------------------------------------------
// section builders
// ---------------------------------------------------------------------------
function camTile({ cam, mediaEl, metaText }) {
  const box = el('div', 'cam');
  box.appendChild(mediaEl);
  const cap = el('div', 'camCap');
  const dot = el('span', 'dot');
  dot.style.background = CAM_COLOR[cam];
  cap.append(dot, el('span', 'name', cam));
  if (metaText) cap.appendChild(el('span', 'meta', metaText));
  box.appendChild(cap);
  return box;
}

function missingTile(cam, msg) {
  const box = el('div', 'cam missing');
  box.appendChild(el('div', null, `${cam} — ${msg}`));
  return box;
}

function buildVideoRow(host, data, kind, conductor) {
  host.textContent = '';
  CAMS.forEach((cam, i) => {
    const src = data.media[kind]?.[cam];
    if (!src) { host.appendChild(missingTile(cam, 'no video')); return; }
    const v = document.createElement('video');
    v.src = `${HF_BASE}/${src}`;
    // No crossOrigin: we never read pixels back, and requesting CORS mode would
    // add a failure path across the Hub's resolve -> CDN redirect for no gain.
    // The wrist input stream is the clock for the entire page.
    conductor.addVideo(v, { master: kind === 'input' && cam === 'wrist' });

    let meta = '';
    if (kind === 'track') {
      const m = data.masktrack?.[cam];
      if (m?.seedless) meta = 'seedless — not tracked';
      else if (m) meta = `seed f${m.seed_frame} · coverage ${(m.coverage * 100).toFixed(0)}%`;
    } else if (kind === 'box') {
      const drawn = data.boxes_drawn?.[cam];
      if (drawn != null) meta = `box on ${fmt(drawn)} / ${fmt(data.n_frames)} frames`;
    } else {
      meta = data.serials?.[cam] ?? '';
    }
    host.appendChild(camTile({ cam, mediaEl: v, metaText: meta }));
  });
}

function buildDescription(host, step1) {
  host.textContent = '';
  const list = step1.language_instructions || [];
  if (!list.length) {
    host.appendChild(el('p', 'none', 'No language instruction was recorded for this episode.'));
    return;
  }
  if (list.length === 1) {
    host.appendChild(el('p', 'quote', list[0]));
    return;
  }
  const ul = el('ul', 'instrList');
  for (const s of list) ul.appendChild(el('li', null, s));
  host.appendChild(ul);
}

function buildStep1(host, s) {
  host.textContent = '';
  const row = el('div', 'labelRow');
  row.appendChild(el('span', 'bigLabel', s.object_name ?? '(unnamed)'));
  const yes = s.suitable_for_tracking === 'YES';
  row.appendChild(el('span', `badge ${yes ? 'ok' : 'bad'}`,
    yes ? 'suitable for tracking' : 'not suitable for tracking'));
  host.appendChild(row);

  host.appendChild(el('span', 'fieldName', 'VLM reasoning'));
  host.appendChild(el('p', 'quote', s.reason || '(no reasoning returned)'));

  const g = el('div', 'kvGrid');
  for (const [k, v] of [['episode length', `${fmt(s.num_steps)} steps`],
                        ['vlm call', s.vlm_success ? 'succeeded' : 'failed']]) {
    const d = el('div', 'kv');
    d.append(el('div', 'k', k), el('div', 'v', v));
    g.appendChild(d);
  }
  host.appendChild(g);
}

function buildStep1b(host, b) {
  host.textContent = '';
  if (!b) {
    host.appendChild(el('p', 'none', 'This episode was not evaluated by the rigidity gate.'));
    return;
  }
  const rigid = b.rigid === 'YES';
  const row = el('div', 'labelRow');
  row.appendChild(el('span', 'bigLabel', rigid ? 'Rigid' : 'Non-rigid'));
  row.appendChild(el('span', `badge ${rigid ? 'ok' : 'bad'}`,
    rigid ? 'passes rigidity gate' : 'dropped — deformable'));
  host.appendChild(row);

  host.appendChild(el('span', 'fieldName', 'VLM reasoning'));
  host.appendChild(el('p', 'quote', b.reason || '(no reasoning returned)'));

  if (b.best_frame != null) {
    const g = el('div', 'kvGrid');
    const d = el('div', 'kv');
    d.append(el('div', 'k', 'frame judged'), el('div', 'v', String(b.best_frame)));
    g.appendChild(d);
    host.appendChild(g);
  }
}

function buildSeeds(host, data) {
  host.textContent = '';
  for (const cam of CAMS) {
    const src = data.media.seed?.[cam];
    const info = data.seeds?.[cam];
    if (!src || !info) {
      const why = 'no seed — SAM3 never detected the object in this view';
      host.appendChild(missingTile(cam, why));
      continue;
    }
    const img = document.createElement('img');
    img.src = `${HF_BASE}/${src}`;
    img.loading = 'lazy';
    img.alt = `${cam} seed mask at frame ${info.frame}`;
    host.appendChild(camTile({
      cam, mediaEl: img,
      metaText: `frame ${info.frame} · score ${num(info.sam3_score, 3)}`,
    }));
  }
}

/** Capturing per-camera poses means re-running the tracker, and FoundationPose
 *  is not bit-deterministic. Where that re-run diverged from the shipped track,
 *  say so rather than quietly showing a different episode's decisions. */
function repro(pc) {
  const r = pc.reproduces_prod;
  if (!r || !r.consensus_total) return '';
  const pct = 100 * r.consensus_match / r.consensus_total;
  if (pct >= 99.5 && r.segments_identical) return '';
  return `<br><br><b>Note:</b> capturing per-camera poses requires re-running the ` +
    `tracker, which is not bit-deterministic. This run matched the shipped track's ` +
    `consensus on ${pct.toFixed(1)}% of frames` +
    (r.segments_identical ? '' : ' and chose a different re-registration point') +
    `. The green/red here is this run's own decision, so it is self-consistent with ` +
    `the boxes drawn — but on the differing frames it is not the decision behind the ` +
    `orange fused box.`;
}

function renderCamBoxStates(state) {
  const host = $('camBoxStates');
  if (!host) return;
  if (!state) { host.textContent = ''; return; }
  const label = { agree: 'in clique', disagree: 'disagrees',
                  lowconf: 'abstained (low confidence)', none: 'no pose' };
  const color = { agree: '#22cc55', disagree: '#ff3b30',
                  lowconf: '#8b93a1', none: 'var(--fg-faint)' };
  host.innerHTML = CAMS.map(c => {
    const s = state.states[c] ?? 'none';
    return `<div style="color:${color[s]}">${c}: ${label[s]}</div>`;
  }).join('')
    + (state.register
        ? `<div style="color:var(--fg-faint)">register frame (${state.register})</div>` : '')
    + (state.status ? `<div style="color:var(--fg-faint)">${state.status}</div>` : '');
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  if (!EID) throw new Error('no ?eid= in the URL');

  const [data, gallery] = await Promise.all([
    fetch(`data/episodes/${EID}.json`).then(r => {
      if (!r.ok) throw new Error(`episode data — HTTP ${r.status}`);
      return r.json();
    }),
    fetch('data/episodes.json').then(r => r.ok ? r.json() : null)
      .then(m => m?.episodes.find(e => e.episode_id === EID) ?? null)
      .catch(() => null),
  ]);

  const objectName = data.step1?.object_name ?? EID;
  $('title').textContent = objectName;
  $('eid').textContent = EID;
  document.title = `${EID} — ${objectName}`;
  const bits = [data.lab && `lab ${data.lab}`, data.date, `${fmt(data.n_frames)} frames`,
                gallery?.instruction && `“${gallery.instruction}”`].filter(Boolean);
  $('subline').textContent = bits.join('  ·  ');

  const conductor = new Conductor({ fps: data.fps, nFrames: data.n_frames, rate: 0.5 });

  buildVideoRow($('inputVideos'), data, 'input', conductor);
  buildDescription($('episodeDesc'), data.step1);
  buildStep1($('step1Card'), data.step1);
  buildStep1b($('step1bCard'), data.step1b);
  buildSeeds($('seedGrid'), data);
  buildVideoRow($('trackVideos'), data, 'track', conductor);
  buildVideoRow($('boxVideos'), data, 'box', conductor);

  // ---- pointcloud ----
  const cloud = new PointCloudView($('cloudHost'), {
    hfBase: HF_BASE, eid: EID,
    onStatus: (msg, extra) => {
      const s = $('cloudStatus');
      s.hidden = !msg;
      if (msg) s.textContent = msg;
      if (extra?.points != null) $('pointCount').textContent = fmt(extra.points);
      if (extra?.meta) {
        $('frameStatus').textContent = extra.meta.status ?? '—';
        $('consensus').textContent = extra.meta.consensus_size ?? '—';
      }
      if (extra && 'camBoxes' in extra) renderCamBoxStates(extra.camBoxes);
    },
  });

  const savedPreset = localStorage.getItem('preset') || 'medium';
  $('presetSel').value = savedPreset;
  await cloud.load(savedPreset);
  const showPresetInfo = () => {
    const ix = cloud.index;
    $('presetInfo').textContent =
      `${(ix.voxel_m * 1000).toFixed(0)} mm voxel · stride ${ix.pixel_stride} · ` +
      `${(cloud.avgBlockBytes / 1024).toFixed(0)} KB/frame, streamed in blocks of 16`;
  };
  showPresetInfo();

  // ---- transport ----
  const timeline = $('timeline');
  timeline.max = String(data.n_frames - 1);

  conductor.addSubscriber((time, frame) => {
    const bi = cloud.blockIndexForFrame(frame);
    if (bi >= 0) cloud.show(bi);
  });

  let scrubbing = false;
  conductor.on('change', c => {
    $('playBtn').textContent = c.playing ? 'Pause' : 'Play';
    const f = c.frame;
    // Don't fight the drag: writing .value mid-gesture snaps the thumb back.
    if (!scrubbing) timeline.value = String(f);
    $('frameLabel').textContent = `${f} / ${data.n_frames - 1}`;
  });

  $('playBtn').addEventListener('click', () => conductor.toggle());
  $('speedSel').addEventListener('change', e => conductor.setRate(Number(e.target.value)));
  const beginScrub = () => { scrubbing = true; conductor.pause(); };
  const endScrub = () => {
    if (!scrubbing) return;
    scrubbing = false;
    conductor.seekFrame(Number(timeline.value));
  };
  for (const ev of ['pointerdown', 'mousedown', 'touchstart', 'keydown']) {
    timeline.addEventListener(ev, beginScrub);
  }
  for (const ev of ['pointerup', 'mouseup', 'touchend', 'keyup', 'blur']) {
    timeline.addEventListener(ev, endScrub);
  }
  addEventListener('pointerup', endScrub);
  timeline.addEventListener('input', () => {
    // Live scrub: seek on every input so the frame follows the thumb.
    if (!scrubbing) conductor.pause();
    conductor.seekFrame(Number(timeline.value));
  });
  timeline.addEventListener('change', endScrub);

  addEventListener('keydown', e => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); conductor.toggle(); }
    if (e.code === 'ArrowRight') { conductor.pause(); conductor.seekFrame(conductor.frame + 1); }
    if (e.code === 'ArrowLeft') { conductor.pause(); conductor.seekFrame(conductor.frame - 1); }
  });

  // ---- pointcloud-only controls ----
  $('presetSel').addEventListener('change', async e => {
    const p = e.target.value;
    localStorage.setItem('preset', p);
    await cloud.load(p);
    showPresetInfo();
    const bi = cloud.blockIndexForFrame(conductor.frame);
    if (bi >= 0) cloud.show(bi);
  });
  $('sizeRange').addEventListener('input', e => {
    cloud.setPointSize(Number(e.target.value));
    $('sizeLabel').textContent = Number(e.target.value).toFixed(1);
  });
  $('boxToggle').addEventListener('change', e => cloud.setShowBox(e.target.checked));
  $('pointsToggle').addEventListener('change', e => cloud.setShowPoints(e.target.checked));
  cloud.setCameraGeometry(data.camera_geometry || null);
  $('frustumToggle').addEventListener('change', e => cloud.setShowFrustums(e.target.checked));
  $('resetView').addEventListener('click', () => cloud.resetView());

  // Per-camera boxes come from a separate per-episode file (each camera's own
  // FoundationPose hypothesis), fetched the first time the toggle is used.
  let perCamPromise = null;
  const loadPerCam = () => {
    perCamPromise ||= fetch(`data/percam/${EID}.json`)
      .then(r => r.ok ? r.json() : null)
      .catch(() => null);
    return perCamPromise;
  };
  $('camBoxToggle').addEventListener('change', async e => {
    const on = e.target.checked;
    $('camBoxPanel').hidden = !on;
    if (!on) { cloud.setShowCamBoxes(false); return; }
    $('camBoxNote').textContent = 'loading per-camera poses…';
    const pc = await loadPerCam();
    if (!pc) {
      $('camBoxNote').textContent =
        'Per-camera poses have not been generated for this episode. ' +
        'Production step3a-2 stores only the fused box.';
      e.target.checked = false;
      return;
    }
    const th = pc.thresholds || {};
    const nseg = (pc.segments || []).length;
    $('camBoxNote').innerHTML =
      `Each camera's own pose hypothesis, as computed by the tracker itself. ` +
      `Green is exactly the pipeline's agreeing clique for that frame; red is exactly ` +
      `what it deemed disagreement. Cameras with no pose that frame are hidden.` +
      `<br><br>The pipeline's rule: a camera is confident at edge ≤ ` +
      `${th.edge_metric ?? 0.15}, two confident cameras agree within ` +
      `${th.consensus_trans_cm ?? 5} cm and ${th.consensus_rot_deg ?? 45}°, and the ` +
      `consensus is the largest mutually-agreeing clique.` +
      `<br><br>Confidence is checked <i>before</i> poses are compared, so a frame status ` +
      `of <code>agree X of Y</code> has Y = cameras that were <b>confident</b>, not cameras ` +
      `that produced a pose. A grey box is one that abstained on confidence — the pipeline ` +
      `never weighed it, so it is not disagreement.` +
      `<br><br>${nseg} registration${nseg === 1 ? '' : 's'} in this episode` +
      (nseg > 1 ? ' — each one restarts every camera from a fresh pose.' : '.') +
      repro(pc);
    cloud.setPerCam(pc);
    cloud.setShowCamBoxes(true);
  });

  conductor.setRate(0.5);
  conductor.start();
  conductor.seekFrame(0);

  $('loading').hidden = true;
  $('app').hidden = false;
}

boot().catch(err => {
  const l = $('loading');
  l.className = 'err';
  l.innerHTML = `Could not load this episode — ${err.message}.<br>` +
    `<span style="font-size:13px">Media and pointclouds are hosted at ` +
    `<code>${HF_REPO}</code> on Hugging Face.</span>`;
});
