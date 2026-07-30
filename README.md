# Visualization site

A **static** site (no server process) for browsing episodes from the DROID 3D
bounding-box pipeline. Built to be published on GitHub Pages, so everything is
plain HTML/CSS/JS with relative paths and no external requests.

```
visualization/
  index.html                  landing page — episode menu with 3-camera previews
  episode.html                per-episode page (placeholder; awaiting display spec)
  data/episodes.json          manifest driving both pages
  assets/previews/{eid}.jpg   wrist | ext1 | ext2 preview strip
  tools/                      build scripts (not published — see note below)
```

## Episode selection

Deliberately unbiased. `tools/select_episodes.py` enumerates **every** episode
with step3a-2 v2 `status == "success"` (26,836 of 71,082 processed), sorts them
for determinism, then applies a single seeded uniform shuffle.
`tools/build_previews.py` walks that permutation in order and publishes the
first N that render, so the sample stays unbiased conditional on renderability.
Anything skipped is recorded in `episodes.json` under `skipped` — currently
none. No filtering by object class, lab, score, or track quality is applied;
the sample includes weak successes as well as clean ones.

Re-running with the same `--seed` reproduces the same episodes exactly. A
different `--seed` draws an independent sample.

## Rebuilding

```bash
PY=/weka/oe-training-default/jasonr/3d_box/miniconda3/envs/sam3d-objects/bin/python

# 1. Enumerate successes + build the seeded permutation.
#    Caches to output/_vizserver_cache/success_episodes.json (~10 s, 71k files).
$PY visualization/tools/select_episodes.py --seed 20260729

# 2. Render previews + manifest.
$PY visualization/tools/build_previews.py --count 12
```

Useful flags: `--count` (episodes published), `--panel-width` (px per camera
panel, default 640), `--quality` (JPEG, default 82).

## Local preview

Browsers block `fetch` on `file://`, so open it over HTTP:

```bash
cd visualization && python -m http.server 8000
# then http://127.0.0.1:8000/
```

## Deployment

`.github/workflows/pages.yml` publishes this directory on every push to `main`
that touches it. **One-time setup:** repo Settings → Pages → Source →
*GitHub Actions*.

`tools/` is uploaded along with the rest of the directory but is inert — it is
never fetched by the pages. Move it out if that matters.

## Episode page

`episode.html` walks the pipeline in order, one section per stage:

1. **Input** — the three synchronized camera streams (left eye only) plus the
   DROID language instructions, which are what the step-1 VLM is given.
2. **Step 1** — the VLM's object label, its suitable-for-tracking verdict, and
   the full reasoning.
3. **Step 1b** — the rigidity verdict and reasoning, rendered as prose.
4. **Step 3a-1 seeds** — each camera's highest-scoring SAM3 image-mode frame
   with its mask. Still images; they do not move with playback.
5. **Step 3d** — the seed-consistency gate.
6. **Step 3a-1 v2** — the SAM3 video-propagated mask tracklet per camera.
7. **Step 3a-2 reprojection** — the final tracked box drawn back onto each raw
   view. The wrist reprojects through its per-frame pose; the externals use the
   static optimized extrinsics.
8. **4D pointcloud** — the fused world-frame cloud with the tracked box.

### Per-camera boxes

The pointcloud panel can overlay each camera's *own* pose hypothesis, green
where the pipeline put it in the agreeing clique and red where the pipeline
deemed it disagreement.

`step3a2_video_track_v2` computes a pose per camera per frame and decides which
agree, but writes out only the fused `bbox_world` plus `consensus_cams`. So
`scripts/_vizserver_step3a2_percam.py` is a copy of the production script whose
**only** change is that it also emits `per_cam_T_obj_in_world` — four marked
additions, no change to tracking, consensus, thresholds or registration.
`tools/build_percam.py` then turns those poses into boxes and takes
`consensus_cams` from that same run **verbatim**; no agreement is recomputed.

That is the point: green/red is the pipeline's own decision. Two earlier
attempts got this wrong and are worth not repeating —

- Using `scripts/step3a2_multiview_sample.py` instead. It registers once where
  production re-registers per segment (one episode here has 14), so it is a
  different trajectory; production's `consensus_cams` did not describe its poses
  and dissenting cameras came out *closer* to the fused box than agreeing ones.
- Substituting a symmetry-invariant box-overlap test for the pipeline's pose
  comparison. It reads better on symmetric objects but it is not what the
  pipeline decided, which is what this overlay is supposed to show. (The
  pipeline's 45° threshold is already deliberately loose for exactly this
  reason — see its docstring on rotation-symmetric objects.)

`build_percam.py` prints, per episode, how closely the re-run reproduces the
shipped output: `consensus_cams` agreement and Hausdorff distance between the
re-run's fused box and production's. Expect ~100% and sub-millimetre; a few
frames flip on some episodes because FoundationPose is not bit-deterministic.

Regenerate with:

```bash
# 1. re-run the tracker with per-camera capture
#    (GPU + foundationpose env; ~1 h for 12 episodes)
$FP_PY scripts/_vizserver_step3a2_percam.py \
    --output_dir output/_vizserver_cache/percam_prod \
    --episode_ids "$(paste -sd, output/_vizserver_cache/viz_eids.txt)" ...
# 2. join into the site payload
$PY visualization/tools/build_percam.py
```

### One clock for the whole page

`sync.js` holds a single `Conductor`. Every time-varying element registers with
it: the three input videos, the three tracklet videos, and the pointcloud. There
is deliberately **no per-element transport** — the `<video>` elements are created
without `controls`, and the Conductor re-asserts play state and rate on every
animation frame, so a video that gets paused out from under it is pulled back
into line. Pausing or changing speed affects everything at once. Default speed
is 0.5x.

The clock is a designated *master* video (the wrist input stream) rather than a
wall-clock accumulator: if the master stalls to buffer, everything waits with it
instead of racing ahead and snapping back. Others resync when they drift more
than ~1 frame.

This only works because every video is encoded at exactly one frame per pipeline
frame at a fixed 15 fps, so `time * 15` is the frame index that indexes the
pointcloud too. `tools/build_media.py` enforces that, and it is worth
re-checking after any encoder change.

### Step 3d presentation

Per the step3d owner, the page shows `verdict`, `reason`, `per_camera_object`
and `disagreeing_cameras`, and deliberately omits `confidence` (uncalibrated —
the model pins it near 0.95), `verdict_vlm`, `src` and `vlm_raw` (internal).
`n/a` means fewer than two cameras were seeded and is presented as
*not applicable*, not a failure. `uncertain` is folded in with `different` as
"flagged" — it occurs once in 95,658 episodes. The copy says a `same` verdict
means the seed detections agree, **not** that every camera stayed on the object
for the whole episode; step3d only inspects the seed frames.

## 4D pointcloud viewer

`episode.html` + `viewer.js` render the per-frame fused 3-camera pointcloud in
the world (Franka base) frame, with a timeline, playback, an orange 3D box
overlay from step3a-2, and a **quality selector**.

The point data does **not** live in this repo — it is far too large. Measured
across the 12 episodes (4,327 frames):

| preset | pixel stride | voxel | KB/frame | all 12 episodes |
|---|---|---|---:|---:|
| `low` | 4 | 15 mm | ~230 | ~1.2 GB |
| `medium` (default) | 4 | 10 mm | ~380 | ~1.7 GB |
| `high` | 2 | 6 mm | ~1250 | ~5.4 GB |

GitHub Pages caps a site at 1 GB and any file at 100 MB, and the soft
100 GB/month bandwidth limit would allow only a few hundred episode views. So
the clouds live in the Hugging Face dataset
[`Silicon23/droid_pipeline_visualization`](https://huggingface.co/datasets/Silicon23/droid_pipeline_visualization)
and the page streams them cross-origin. Verified: the Hub reflects the request
origin in `Access-Control-Allow-Origin`, answers CORS preflight, and honours
`Range` with `206`.

Frames are fetched **16 at a time** by byte range rather than one request per
frame — the Hub rate-limits unauthenticated resolves to 3,000 per 300 s, which
per-frame requests would hit during ordinary playback.

### Rebuilding and publishing the clouds

```bash
# Pack (writes to visualization/hf_dataset/, which is gitignored).
$PY visualization/tools/pack_clouds.py --workers 12

# Push to the Hub (resumable; needs a write token).
HF_HOME=/root/.cache/huggingface $PY visualization/tools/upload_hf.py
```

`--presets low,medium` limits which presets are built; `--frame-stride 2` halves
the data by dropping every other frame. Presets sharing a pixel stride reuse one
unprojection pass, so building all three costs little more than building two.

## Size budget

The published site is ~2 MB (previews + vendored Three.js). All heavy data is on
Hugging Face, so the GitHub side stays small indefinitely.

## Conventions worth keeping

- **Stereo detection is `w > 2 * h`, never `w > 1920`.** DROID MP4s are
  side-by-side left|right; everything downstream uses the **left** eye only. An
  absolute-width test misses ZED WVGA stereo (1344×376) and silently reads the
  wrong eye. `scripts/visualize_pointcloud.py` still has this bug — do not copy
  its `read_frame_from_mp4`.
- Per-camera accent colors: **red = wrist, green = ext1, blue = ext2**, matching
  the pointcloud viewers elsewhere in the project.
- Captions go in a bar **below** each tile, never overlaid — overlaying hides
  the mask-to-edge relationship at the frame border.
- The preview frame is the first segment's `register_frame` (where
  FoundationPose registered the object), so the object is visible and
  well-posed in it.
