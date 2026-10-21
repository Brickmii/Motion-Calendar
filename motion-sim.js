// Motion Calendar live simulation: one block of the fractal, filled only by interaction.
// Rules (Ian, 2026-10): a block holds 265 positions and 45 rotations; a shape grows
// point -> line -> triangle -> square -> triangle in a square, then each polygon rotates
// inside the next; the 44th is the last position the topology holds and the 45th blooms.
// A block blooms when a new block is needed, and adding a side is the chooser's choice.
(function () {
  const RAY = Math.PI / 45;            // the minimal choice, 4 degrees
  const RAYS_PER_TURN = 90;
  const BLOCK_POSITIONS = 265;         // 45 + 5 x 44 (diamond closed at 12 shells)
  const BLOCK_ROTATIONS = 45;
  const HELD_POINTS = 44;              // the last position the topology can hold
  const STAGES = ['Heat', 'Polarity', 'Existence', 'Righteousness', 'Order', 'Movement'];
  const SHAPES = ['point', 'line', 'triangle', 'square', 'triangle in a square'];
  const COL = { habitP: '#e8a45c', habitN: '#5c9be8', effort: '#e4e4e7', hold: '#55555c', bloom: '#d4a574' };

  // Diamond cells: shell n holds 4n cells (shell 0 holds the center), labelled a, b, c, d in turn.
  const DIAMOND = [];
  for (let n = 0; n < 12; n++) {
    const ring = [];
    if (n === 0) ring.push([0, 0]);
    else for (let k = 0; k < 4 * n; k++) {
      const side = Math.floor(k / n), t = k % n;
      const p = [[n - t, t], [-t, n - t], [-n + t, -t], [t, -n + t]][side];
      ring.push(p);
    }
    ring.forEach(([x, y]) => DIAMOND.push({ x, y, shell: n, label: 'abcd'[n % 4] }));
  }

  class Block {
    constructor(w, depth, orient, m0, firstAngle) {
      this.w = w; this.depth = depth; this.orient = orient; this.m0 = m0;
      this.points = [];          // defined addresses: {addr, angle}
      this.cells = [];           // filled positions, in order: kind
      this.rotations = 0;
      this.rot = [];             // accumulated rotation (rays) per polygon index
      this.arm = 0;              // index of the point the arm rests on
      this.spin = 0;             // whole-nest rotation, in rays
      if (firstAngle !== undefined) this.define(firstAngle, 'bloom');
    }
    sections() { return Math.pow(10, this.depth); }   // rays sectioned 10 per depth
    address(angle) { return Math.floor(((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (RAY / this.sections())); }
    find(addr) { return this.points.findIndex(p => p.addr === addr); }
    sides() { return this.points.length >= 5 ? this.points.length - 1 : this.points.length; }
    stage() { return Math.min(this.points.length, 6) - 1; }
    define(angle, kind) {
      this.points.push({ addr: this.address(angle), angle });
      this.arm = this.points.length - 1;
      this.fill(kind);
    }
    fill(kind) { this.cells.push(kind); }
    full() { return this.cells.length >= BLOCK_POSITIONS; }
  }

  class Chooser {
    constructor() { this.reset(); }
    reset() {
      this.H = 0; this.N = 0; this.spent = 0;
      this.blocks = [new Block(1, 0, 1, 1)];
      this.log = [];
    }
    get block() { return this.blocks[this.blocks.length - 1]; }
    get T() { return this.N ? this.H / this.N : 1; }

    // One interaction: an observation at an angle with magnitude m (in rays).
    receive(angle, m) {
      this.H += m; this.N += 1;                       // contact at the square; heat never subtracts
      const b = this.block, addr = b.address(angle), at = b.find(addr);
      let choice;
      if (b.points.length === 0) {
        b.define(angle, 'effort'); this.spent += 2; choice = 'effort: the first point';
      } else if (at >= 0) {
        // Familiar: habit. Rotate the arm to it (+ or -), or hold if it is already there.
        const d = ((at - b.arm) % b.points.length + b.points.length) % b.points.length;
        const gate = d === 0 ? 0 : (d <= b.points.length / 2 ? 1 : -1);
        if (gate === 0) { b.fill('hold'); this.spent += 1; choice = 'habit: hold (0)'; }
        else {
          if (b.rotations >= BLOCK_ROTATIONS) return this.bloom(angle, m, gate, 'no rotations left');
          b.arm = at; b.rotations++; b.spin += gate;
          for (let k = 0; k < b.sides() - 3; k++) b.rot[k] = (b.rot[k] || 0) + gate;   // polygons inside the outer one
          b.fill(gate > 0 ? 'habitP' : 'habitN'); this.spent += 1;
          choice = `habit: rotate ${gate > 0 ? '+' : '−'}`;
        }
      } else if (m >= this.T) {
        // New and worth defining: effort, a new address. The 45th point blooms.
        if (b.points.length >= HELD_POINTS) return this.bloom(angle, m, 1, 'the 45th point');
        b.define(angle, 'effort'); this.spent += 2;
        choice = b.points.length <= 5 ? `effort: ${SHAPES[b.points.length - 1]}` : `effort: a side (${b.sides()})`;
      } else {
        // New but below the triangle's temperature: the chooser declines to define it.
        b.fill('hold'); this.spent += 1; choice = 'hold: not defined';
      }
      if (b.full()) return this.bloom(angle, m, 1, 'every position filled', choice);
      this.note(choice);
      return choice;
    }

    // A new block is needed. Its form comes from the information that called for it.
    bloom(angle, m, gate, why, before) {
      const b = this.block;
      const depth = m > b.m0 ? b.depth + 1 : (m < b.m0 ? Math.max(0, b.depth - 1) : b.depth);
      this.blocks.push(new Block(b.w + 1, depth, gate >= 0 ? 1 : -1, m, angle));
      this.spent += 2;
      const choice = `${before ? before + '; ' : ''}bloom (${why}) → w = ${b.w + 1}, depth ${depth}`;
      this.note(choice);
      return choice;
    }
    note(c) { this.log.unshift(c); this.log.length = Math.min(this.log.length, 6); }
  }

  // The world mode: a source that returns to a few places, with drift and varied magnitude.
  class World {
    constructor() { this.home = Array.from({ length: 7 }, () => Math.random() * 2 * Math.PI); this.i = 0; }
    next() {
      if (Math.random() < 0.3) this.i = Math.floor(Math.random() * this.home.length);
      if (Math.random() < 0.08) this.home[this.i] = Math.random() * 2 * Math.PI;
      const a = this.home[this.i] + (Math.random() - 0.5) * RAY * 0.8;
      return [a, 1 + Math.floor(Math.random() * 3)];
    }
  }

  class View {
    constructor() {
      this.modal = document.getElementById('sim-modal');
      this.canvas = document.getElementById('sim-canvas');
      this.ctx = this.canvas.getContext('2d');
      this.dia = document.getElementById('sim-diamond');
      this.dctx = this.dia.getContext('2d');
      this.chooser = new Chooser();
      this.world = new World();
      this.yaw = 0.5; this.pitch = 0.55; this.drag = null;
      this.worldOn = false; this.timer = null; this.anim = 0; this.running = false;
      this.bind();
    }
    bind() {
      const $ = id => document.getElementById(id);
      ['open-sim', 'open-sim-2'].forEach(id => { const el = $(id); if (el) el.addEventListener('click', e => { e.preventDefault(); this.open(); }); });
      $('sim-close').addEventListener('click', () => this.close());
      $('btn-reset').addEventListener('click', () => { this.chooser.reset(); this.anim = 1; });
      $('btn-world').addEventListener('click', () => this.toggleWorld());
      const c = this.canvas;
      c.addEventListener('pointerdown', e => { this.drag = { x: e.clientX, y: e.clientY, moved: false }; c.setPointerCapture(e.pointerId); });
      c.addEventListener('pointermove', e => {
        if (!this.drag) return;
        const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) this.drag.moved = true;
        if (this.drag.moved) {
          this.yaw += dx * 0.01; this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + dy * 0.01));
          this.drag.x = e.clientX; this.drag.y = e.clientY;
        }
      });
      c.addEventListener('pointerup', e => { if (this.drag && !this.drag.moved) this.interact(e); this.drag = null; });
      window.addEventListener('resize', () => { if (this.modal.classList.contains('active')) this.resize(); });
      document.addEventListener('keydown', e => {
        if (!this.modal.classList.contains('active')) return;
        if (e.code === 'Escape') this.close();
        if (e.code === 'KeyW') this.toggleWorld();
        if (e.code === 'KeyR') { this.chooser.reset(); this.anim = 1; }
      });
    }
    open() { this.modal.classList.add('active'); this.resize(); if (!this.running) { this.running = true; this.loop(); } }
    close() { this.modal.classList.remove('active'); if (this.worldOn) this.toggleWorld(); }
    resize() {
      const r = this.canvas.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      this.canvas.width = r.width * dpr; this.canvas.height = r.height * dpr;
      this.W = r.width; this.Hh = r.height; this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    toggleWorld() {
      this.worldOn = !this.worldOn;
      const btn = document.getElementById('btn-world');
      btn.classList.toggle('active', this.worldOn); btn.textContent = this.worldOn ? '■ Stop world' : '▶ World feeds';
      if (this.worldOn) this.timer = setInterval(() => this.feed(...this.world.next()), 450);
      else clearInterval(this.timer);
    }
    // A visitor's click is an observation: its direction is the angle, its distance the magnitude.
    interact(e) {
      const r = this.canvas.getBoundingClientRect(), x = e.clientX - r.left - this.W / 2, y = e.clientY - r.top - this.Hh / 2;
      const R = Math.min(this.W, this.Hh) * 0.42, d = Math.hypot(x, y);
      this.feed(Math.atan2(-y, x), 1 + Math.min(2, Math.floor(d / (R / 1.5))));
    }
    feed(angle, m) {
      const before = this.chooser.blocks.length;
      this.chooser.receive(angle, m);
      if (this.chooser.blocks.length > before) this.anim = 1;
    }
    project(x, y, z) {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
      const y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp;
      const s = 1 / (1 + z2 * 0.0015);
      return [this.W / 2 + x1 * s, this.Hh / 2 - y2 * s];
    }
    draw() {
      const ctx = this.ctx, W = this.W, H = this.Hh, b = this.chooser.block;
      ctx.fillStyle = '#0a0a0b'; ctx.fillRect(0, 0, W, H);
      const R = Math.min(W, H) * 0.42 * (1 - 0.5 * this.anim);
      this.anim = Math.max(0, this.anim - 0.03);
      // x, y, z axes, faint
      ctx.lineWidth = 1;
      [[R, 0, 0, '#3a2a2a'], [0, R, 0, '#2a3a2a'], [0, 0, R * 0.6, '#2a2a3a']].forEach(([x, y, z, c]) => {
        const [a1, a2] = this.project(-x, -y, -z), [b1, b2] = this.project(x, y, z);
        ctx.strokeStyle = c; ctx.beginPath(); ctx.moveTo(a1, a2); ctx.lineTo(b1, b2); ctx.stroke();
      });
      const n = b.points.length, sides = b.sides(), hue = b.orient > 0 ? [212, 165, 116] : [116, 165, 212];
      const spin = b.spin * RAY;
      const poly = (k, r, z, alpha, rot) => {
        ctx.beginPath();
        for (let i = 0; i <= k; i++) {
          const a = rot + i * 2 * Math.PI / k, [px, py] = this.project(r * Math.cos(a), r * Math.sin(a), z);
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.strokeStyle = `rgba(${hue[0]},${hue[1]},${hue[2]},${alpha})`; ctx.stroke();
      };
      if (n >= 3) {
        // The nest: each polygon rotates inside the next, from the triangle out to the current outer polygon.
        for (let k = 3; k <= sides; k++) {
          const frac = sides === 3 ? 1 : (k - 3) / (sides - 3);
          const r = R * (0.22 + 0.78 * frac), z = (sides - k) * 6;
          ctx.lineWidth = k === sides ? 2 : 1;
          poly(k, r, z, k === sides ? 0.95 : 0.25 + 0.5 * frac, (b.rot[k - 3] || 0) * RAY + Math.PI / 2);
        }
      } else if (n === 2) {
        const [a1, a2] = this.project(-R * 0.5, 0, 0), [b1, b2] = this.project(R * 0.5, 0, 0);
        ctx.strokeStyle = `rgb(${hue})`; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(a1, a2); ctx.lineTo(b1, b2); ctx.stroke();
      }
      // The square's boundary, where observations arrive; the defined points sit on it at their own angles.
      const RB = R * 1.12;
      ctx.beginPath();
      for (let i = 0; i <= 90; i++) {
        const a = i * RAY * 2, [px, py] = this.project(RB * Math.cos(a), RB * Math.sin(a), 0);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.strokeStyle = 'rgba(113,113,122,0.25)'; ctx.lineWidth = 1; ctx.stroke();
      b.points.forEach((p, i) => {
        const [px, py] = this.project(RB * Math.cos(p.angle + spin), RB * Math.sin(p.angle + spin), 0);
        ctx.fillStyle = i === b.arm ? '#fff' : `rgba(${hue[0]},${hue[1]},${hue[2]},0.8)`;
        ctx.beginPath(); ctx.arc(px, py, i === b.arm ? 4 : 2.5, 0, 2 * Math.PI); ctx.fill();
      });
      const [cx, cy] = this.project(0, 0, 0);
      if (n) {
        const p = b.points[b.arm], [ax, ay] = this.project(RB * Math.cos(p.angle + spin), RB * Math.sin(p.angle + spin), 0);
        ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ax, ay); ctx.stroke();
      }
      ctx.fillStyle = n >= 5 ? '#fff' : `rgb(${hue})`;
      ctx.beginPath(); ctx.arc(cx, cy, n >= 5 ? 5 : 3, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = '#71717a'; ctx.font = '12px JetBrains Mono, monospace'; ctx.textAlign = 'center';
      if (!n) ctx.fillText('Click anywhere to interact. Nothing fills without information.', W / 2, H - 24);
    }
    drawDiamond() {
      const ctx = this.dctx, w = this.dia.width, h = this.dia.height, b = this.chooser.block, c = 6.2;
      ctx.fillStyle = '#18181b'; ctx.fillRect(0, 0, w, h);
      DIAMOND.forEach((cell, i) => {
        const kind = b.cells[i];
        ctx.fillStyle = kind ? COL[kind] : (cell.shell % 2 ? '#26262b' : '#202024');
        ctx.fillRect(w / 2 + cell.x * c - c / 2 + 0.5, h / 2 - cell.y * c - c / 2 + 0.5, c - 1, c - 1);
      });
    }
    ui() {
      const ch = this.chooser, b = ch.block, set = (id, v) => { document.getElementById(id).textContent = v; };
      set('stat-w', b.w); set('stat-depth', `${b.depth} (ray ÷ ${b.sections()})`);
      set('stat-positions', `${b.cells.length} / ${BLOCK_POSITIONS}`);
      set('stat-rotations', `${b.rotations} / ${BLOCK_ROTATIONS}`);
      set('stat-points', `${b.points.length} / ${HELD_POINTS}`);
      set('stat-sides', b.points.length >= 3 ? b.sides() : '—');
      set('stat-stage', b.points.length ? `${STAGES[b.stage()]} (${b.points.length <= 5 ? SHAPES[b.points.length - 1] : 'rotation'})` : '—');
      set('stat-temp', ch.T.toFixed(2)); set('stat-heat', (ch.H + ch.spent).toFixed(0));
      document.getElementById('sim-log').innerHTML = ch.log.map(l => `<div>${l}</div>`).join('');
    }
    loop() {
      if (!this.modal.classList.contains('active')) { this.running = false; return; }
      this.draw(); this.drawDiamond(); this.ui();
      requestAnimationFrame(() => this.loop());
    }
  }

  window.addEventListener('DOMContentLoaded', () => new View());
})();
