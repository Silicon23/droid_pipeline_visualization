// One clock for the whole page.
//
// Every time-varying visualization on an episode page — the three input
// videos, the three mask-tracklet videos, and the 4D pointcloud — is driven
// from a single Conductor. There are deliberately no per-video controls: the
// <video> elements are created without `controls`, and if anything pauses one
// of them out from under us (autoplay policy, a stall, the OS) the Conductor
// pulls it back in line on the next frame.
//
// A designated *master* video supplies the clock rather than a wall-clock
// accumulator. If the master stalls to buffer, everything else waits with it
// instead of racing ahead and then snapping back.

const DRIFT_TOLERANCE = 0.08;   // seconds (~1 frame at 15 fps) before we reseek

export class Conductor {
  constructor({ fps = 15, nFrames = 0, rate = 0.5 } = {}) {
    this.fps = fps;
    this.nFrames = nFrames;
    this.rate = rate;
    this.playing = false;
    this.time = 0;
    this._pendingSeek = null;
    this.videos = [];
    this.master = null;
    this.subscribers = [];
    this._listeners = { change: [] };
    this._raf = null;
    this._tick = this._tick.bind(this);
  }

  get duration() {
    if (this.master && Number.isFinite(this.master.el.duration) && this.master.el.duration > 0) {
      return this.master.el.duration;
    }
    return this.nFrames / this.fps;
  }

  get frame() {
    // floor, NOT round: a <video> displays frame i for t in [i/fps, (i+1)/fps),
    // so rounding puts the pointcloud one frame ahead of the picture for the
    // second half of every frame. That showed up as the 3D box disappearing
    // while the video still drew one (or vice versa) on isolated bad frames.
    const f = Math.floor(this.time * this.fps + 1e-6);
    return Math.min(this.nFrames - 1, Math.max(0, f));
  }

  on(evt, fn) { (this._listeners[evt] ||= []).push(fn); return this; }
  _emit(evt) { for (const fn of this._listeners[evt] || []) fn(this); }

  /** Register a <video>. The first master:true element becomes the clock. */
  addVideo(el, { master = false } = {}) {
    el.controls = false;          // no per-video transport, by design
    el.playsInline = true;
    el.muted = true;              // required for programmatic play()
    el.preload = 'auto';
    el.loop = false;              // looping is coordinated, not per element
    const rec = { el };
    this.videos.push(rec);
    if (master || !this.master) this.master = rec;
    el.playbackRate = this.rate;
    return rec;
  }

  /** fn(timeSeconds, frameIndex) — called once per animation frame. */
  addSubscriber(fn) { this.subscribers.push(fn); return this; }

  start() {
    if (this._raf === null) this._raf = requestAnimationFrame(this._tick);
  }

  stop() {
    if (this._raf !== null) { cancelAnimationFrame(this._raf); this._raf = null; }
  }

  play() {
    this.playing = true;
    for (const { el } of this.videos) {
      el.playbackRate = this.rate;
      const p = el.play();
      if (p && p.catch) p.catch(() => { /* autoplay refusal; retried each tick */ });
    }
    this._emit('change');
  }

  pause() {
    this.playing = false;
    for (const { el } of this.videos) el.pause();
    this._emit('change');
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  setRate(rate) {
    this.rate = rate;
    for (const { el } of this.videos) el.playbackRate = rate;
    this._emit('change');
  }

  /** Seek to the middle of frame `idx`'s interval, so floor() lands on it
   *  regardless of decoder rounding at the boundary. */
  seekFrame(idx) { this.seekTime((idx + 0.5) / this.fps); }

  seekTime(t) {
    const dur = this.duration || 0;
    this.time = Math.min(Math.max(t, 0), Math.max(dur - 1e-3, 0));
    // Seeking a <video> is async. Until the master reports the new position,
    // _tick must not overwrite `time` with the stale one — that made a paused
    // scrub snap straight back to where it started.
    this._pendingSeek = this.time;
    for (const { el } of this.videos) {
      if (el.readyState >= 1) el.currentTime = this.time;
    }
    this._pushSubscribers(true);
    this._emit('change');
  }

  _pushSubscribers(force = false) {
    const f = this.frame;
    if (!force && f === this._lastFrame) return;
    this._lastFrame = f;
    for (const fn of this.subscribers) fn(this.time, f);
  }

  _tick() {
    this._raf = requestAnimationFrame(this._tick);

    const m = this.master;
    if (this._pendingSeek !== null && this._pendingSeek !== undefined) {
      const at = m && m.el.readyState >= 1 ? m.el.currentTime : null;
      if (at !== null && Math.abs(at - this._pendingSeek) <= DRIFT_TOLERANCE) {
        this._pendingSeek = null;          // master arrived; resume normal tracking
      } else {
        this.time = this._pendingSeek;     // hold the requested position
        this._pushSubscribers();
        return;
      }
    }

    if (m && m.el.readyState >= 2) {
      this.time = m.el.currentTime;
      // Coordinated loop: when the master runs out, rewind everything together.
      if (m.el.ended || (this.duration && this.time >= this.duration - 1e-3)) {
        this.seekTime(0);
        if (this.playing) this.play();
      }
    }

    for (const { el } of this.videos) {
      if (el.readyState < 1) continue;
      if (el.playbackRate !== this.rate) el.playbackRate = this.rate;
      // Nobody gets to pause or run ahead on their own.
      if (this.playing && el.paused && !el.ended) {
        const p = el.play();
        if (p && p.catch) p.catch(() => {});
      }
      if (!this.playing && !el.paused) el.pause();
      if (el !== (m && m.el) && Math.abs(el.currentTime - this.time) > DRIFT_TOLERANCE) {
        el.currentTime = this.time;
      }
    }

    // Only notify on an actual frame change: emitting every animation frame
    // would rewrite the transport DOM ~60x/s alongside the WebGL render.
    const f = this.frame;
    if (f !== this._lastFrame) {
      this._pushSubscribers();
      this._emit('change');
    }
  }
}
