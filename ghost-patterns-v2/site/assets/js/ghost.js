/*
 * In-browser ghost-pattern experiment and the site's SVG charts.
 *
 * A simplified version of the Python package: two-item rules only, one
 * IndependentShuffle null, family-wise calibration against the best rule of
 * each null run. Exposed as window.Ghost.
 */
(function () {
  "use strict";

  const css = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const NS = "http://www.w3.org/2000/svg";
  const MONO = "JetBrains Mono, monospace";

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, str, extra) {
    const t = el("text", Object.assign({ x, y, fill: css("--ink-soft"),
      "font-size": 12, "font-family": MONO }, extra || {}), parent);
    t.textContent = str;
    return t;
  }

  // ------------------------------------------------------------- data

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Column 0 = source (4 values), column 1 = target (no/yes), the rest are
  // noise columns with 3 values each.
  function makeData(nRows, nNoise, effect, rnd) {
    const cols = [], names = [], levels = [];
    const src = new Int8Array(nRows), tgt = new Int8Array(nRows);
    for (let i = 0; i < nRows; i++) {
      src[i] = Math.floor(rnd() * 4);
      const p = src[i] === 0 ? Math.min(0.25 + effect, 0.99) : 0.25;
      tgt[i] = rnd() < p ? 1 : 0;
    }
    cols.push(src, tgt);
    names.push("source", "target");
    levels.push(["a", "b", "c", "d"], ["no", "yes"]);
    for (let c = 0; c < nNoise; c++) {
      const col = new Int8Array(nRows);
      for (let i = 0; i < nRows; i++) col[i] = Math.floor(rnd() * 3);
      cols.push(col);
      names.push("noise" + String(c).padStart(2, "0"));
      levels.push(["v0", "v1", "v2"]);
    }
    return { cols, names, levels, n: nRows };
  }

  function mine(data, th, keepLabels) {
    const { cols, levels, n } = data;
    const out = [];
    const freq = cols.map((c, j) => {
      const f = new Float64Array(levels[j].length);
      for (let i = 0; i < n; i++) f[c[i]]++;
      return f;
    });
    for (let a = 0; a < cols.length; a++) {
      for (let b = a + 1; b < cols.length; b++) {
        const la = levels[a].length, lb = levels[b].length;
        const joint = new Float64Array(la * lb);
        const ca = cols[a], cb = cols[b];
        for (let i = 0; i < n; i++) joint[ca[i] * lb + cb[i]]++;
        for (let x = 0; x < la; x++) {
          for (let y = 0; y < lb; y++) {
            const nxy = joint[x * lb + y];
            if (nxy / n < th.minSupport) continue;
            const pairs = [[a, x, b, y, freq[a][x], freq[b][y]],
                           [b, y, a, x, freq[b][y], freq[a][x]]];
            for (const [i1, v1, i2, v2, nAnte, nCons] of pairs) {
              const conf = nxy / nAnte, lift = conf / (nCons / n);
              if (conf < th.minConf || lift < th.minLift) continue;
              out.push(keepLabels ? {
                ante: data.names[i1] + "=" + levels[i1][v1],
                cons: data.names[i2] + "=" + levels[i2][v2],
                lift, conf, support: nxy / n, cols: [i1, i2],
              } : lift);
            }
          }
        }
      }
    }
    return out;
  }

  function shuffled(data, rnd) {
    const cols = data.cols.map((c) => {
      const s = Int8Array.from(c);
      for (let i = s.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const t = s[i]; s[i] = s[j]; s[j] = t;
      }
      return s;
    });
    return Object.assign({}, data, { cols });
  }

  const DEFAULTS = {
    rows: 800, noise: 16, effect: 0.3, seed: 7, runs: 39, alpha: 0.05,
    minSupport: 0.03, minConf: 0.3, minLift: 1.15,
  };

  function experiment(options) {
    const o = Object.assign({}, DEFAULTS, options);
    const rnd = mulberry32(o.seed);
    const data = makeData(o.rows, o.noise, o.effect, rnd);
    const rules = mine(data, o, true);
    const allNull = [], maxNull = [];
    for (let r = 0; r < o.runs; r++) {
      const lifts = mine(shuffled(data, rnd), o, false);
      for (const l of lifts) allNull.push(l);
      maxNull.push(lifts.length ? Math.max.apply(null, lifts) : 0);
    }
    for (const rule of rules) {
      let hits = 0;
      for (const m of maxNull) if (m >= rule.lift) hits++;
      rule.p = (hits + 1) / (o.runs + 1);
      rule.real = rule.p <= o.alpha;
      rule.planted = rule.cols.every((c) => c < 2);
    }
    rules.sort((x, y) => (y.real - x.real) || (y.lift - x.lift));
    return {
      options: o, rules, allNull, maxNull,
      ghostRate: allNull.length / o.runs,
      ghostMax: allNull.length ? Math.max.apply(null, allNull) : null,
      pFloor: 1 / (o.runs + 1),
    };
  }

  // ------------------------------------------------------------- charts

  function distChart(svg, res, W, H) {
    svg.replaceChildren();
    const pad = { l: 44, r: 16, t: 14, b: 34 };
    const lifts = res.rules.map((r) => r.lift);
    const xmax = Math.max(1.6, ...lifts, ...res.allNull) * 1.05;
    const xmin = Math.min(1.1, res.options.minLift - 0.05);
    const x = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
    const nb = 40, bw = (xmax - xmin) / nb, bins = new Array(nb).fill(0);
    for (const v of res.allNull) {
      bins[Math.min(nb - 1, Math.max(0, Math.floor((v - xmin) / bw)))]++;
    }
    const ymax = Math.max(1, ...bins);
    const y = (c) => H - pad.b - (c / ymax) * (H - pad.t - pad.b - 40);
    const ink = css("--ink-soft");

    el("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b,
      stroke: css("--rule") }, svg);
    for (let t = Math.ceil(xmin * 2) / 2; t <= xmax; t += 0.5) {
      el("line", { x1: x(t), x2: x(t), y1: H - pad.b, y2: H - pad.b + 5,
        stroke: ink }, svg);
      text(svg, x(t), H - pad.b + 19, t.toFixed(1), { "text-anchor": "middle" });
    }
    text(svg, W - pad.r, H - 4, "lift →", { "text-anchor": "end" });

    bins.forEach((c, i) => {
      if (!c) return;
      el("rect", { x: x(xmin + i * bw) + 1,
        width: Math.max(1, x(xmin + bw) - x(xmin) - 2),
        y: y(c), height: H - pad.b - y(c), fill: css("--ghost-fill"),
        class: "bar-grow" }, svg);
    });

    if (res.ghostMax !== null) {
      el("line", { x1: x(res.ghostMax), x2: x(res.ghostMax), y1: pad.t + 12,
        y2: H - pad.b, stroke: ink, "stroke-dasharray": "4 4" }, svg);
      text(svg, x(res.ghostMax) + 6, pad.t + 22,
           "strongest ghost " + res.ghostMax.toFixed(2));
    }
    for (const r of res.rules) {
      el("line", { x1: x(r.lift), x2: x(r.lift), y1: H - pad.b,
        y2: H - pad.b - (r.real ? 70 : 34),
        stroke: r.real ? css("--signal") : css("--ghost"),
        "stroke-width": r.real ? 3 : 1.5 }, svg);
    }
    const best = res.rules.find((r) => r.real);
    if (best) {
      text(svg, Math.min(x(best.lift), W - pad.r - 4), H - pad.b - 78,
           best.ante + " → " + best.cons,
           { "text-anchor": "end", fill: css("--signal"), "font-size": 13,
             "font-weight": 600 });
    }
  }

  // Grouped bars on a log scale: [[label, mined, fromNoise, validated], ...]
  function caseChart(svg, data) {
    svg.replaceChildren();
    const W = 900, H = 300, pad = { l: 56, r: 16, t: 20, b: 44 };
    const ink = css("--ink-soft");
    const colors = [css("--ghost-fill"), css("--ghost"), css("--signal")];
    const top = 5;
    const y = (v) => H - pad.b - (Math.log10(Math.max(v, 1)) / top) * (H - pad.t - pad.b);
    for (let p = 0; p <= 4; p++) {
      el("line", { x1: pad.l, x2: W - pad.r, y1: y(10 ** p), y2: y(10 ** p),
        stroke: css("--rule") }, svg);
      text(svg, pad.l - 8, y(10 ** p) + 4, (10 ** p).toLocaleString("en"),
           { "text-anchor": "end" });
    }
    const gw = (W - pad.l - pad.r) / data.length;
    const bw = Math.min(70, gw / 4.5);
    data.forEach(([name, ...vals], g) => {
      const cx = pad.l + gw * g + gw / 2;
      vals.forEach((v, k) => {
        const bx = cx + (k - 1) * (bw + 6) - bw / 2;
        const h = Math.max(2, H - pad.b - y(v));
        el("rect", { x: bx, width: bw, y: H - pad.b - h, height: h,
          fill: colors[k] }, svg);
        text(svg, bx + bw / 2, H - pad.b - h - 6, v.toLocaleString("en"), {
          "text-anchor": "middle", fill: k === 2 ? css("--signal") : ink,
          "font-weight": k === 2 ? 600 : 400 });
      });
      text(svg, cx, H - pad.b + 24, name, { "text-anchor": "middle",
        fill: css("--ink"), "font-size": 14,
        "font-family": "Bricolage Grotesque, sans-serif", "font-weight": 600 });
    });
    el("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b,
      stroke: ink }, svg);
    text(svg, 12, pad.t + 4, "rules, log scale");
  }

  // Line chart: [{x, y}] series with optional reference line.
  function lineChart(svg, points, opts) {
    svg.replaceChildren();
    const W = 900, H = 260, pad = { l: 56, r: 20, t: 40, b: 44 };
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymax = opts.ymax != null ? opts.ymax : Math.max(...ys) * 1.1;
    const x = (v) => pad.l + ((v - xmin) / (xmax - xmin || 1)) * (W - pad.l - pad.r);
    const y = (v) => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
    const ink = css("--ink-soft");
    (opts.yticks || []).forEach((t) => {
      el("line", { x1: pad.l, x2: W - pad.r, y1: y(t), y2: y(t), stroke: css("--rule") }, svg);
      text(svg, pad.l - 8, y(t) + 4, opts.yfmt ? opts.yfmt(t) : t, { "text-anchor": "end" });
    });
    points.forEach((p) => text(svg, x(p.x), H - pad.b + 20, opts.xfmt ? opts.xfmt(p.x) : p.x,
                                { "text-anchor": "middle" }));
    if (opts.ref != null) {
      el("line", { x1: pad.l, x2: W - pad.r, y1: y(opts.ref), y2: y(opts.ref),
        stroke: css("--signal"), "stroke-dasharray": "5 4" }, svg);
      text(svg, pad.l + 6, y(opts.ref) + 16, opts.refLabel || "",
           { fill: css("--signal") });
    }
    el("polyline", { points: points.map((p) => x(p.x) + "," + y(p.y)).join(" "),
      fill: "none", stroke: css("--cool"), "stroke-width": 3 }, svg);
    points.forEach((p) => {
      el("circle", { cx: x(p.x), cy: y(p.y), r: 5, fill: css("--cool") }, svg);
      if (opts.labels) text(svg, x(p.x), y(p.y) - 12, opts.labels(p), {
        "text-anchor": "middle", fill: css("--ink"), "font-weight": 600 });
    });
    text(svg, W - pad.r, H - 4, opts.xlabel || "", { "text-anchor": "end" });
    text(svg, 12, 16, opts.ylabel || "");
  }

  window.Ghost = { experiment, distChart, caseChart, lineChart, DEFAULTS };
})();
