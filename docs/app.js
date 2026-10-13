/* Страница проекта: всё строится из window.DATA (scripts/export_demo.py). */
(function () {
  const D = window.DATA;
  const NAMES = D.names;
  const COLORS = ["#f4d03f", "#d4ac6e", "#9aa5a8", "#a9cf8f", "#c39bd3", "#5dade2",
                  "#cfe8f7", "#f1948a", "#8e44ad", "#e67e22", "#1c1c1c", "#8b4513"];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (x, d = 3) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d).replace(".", ",");
  const pct = (x, d = 0) => (x * 100).toFixed(d).replace(".", ",") + " %";
  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  const lad = (key, split) => D.ladder.find((r) => r.key === key && r.split === split);
  const H = D.hidden, E = D.experiments;
  const cvW = lad("lgbm_full", "CV по скважинам"), cvP = lad("lgbm_full", "CV по точкам"),
        open = lad("lgbm_full", "open test");
  const worst = D.per_well_hidden.slice().sort((a, b) => a.score - b.score)[0];
  const rest = D.posthoc_info;

  /* ---------- KPI и вывод ---------- */
  el("kpis").innerHTML = [
    ["Скрытые скважины (финал)", fmt(H.score), `точность ${pct(H.accuracy)} · 10 скважин`],
    ["Место среди финалистов", `${H.rank_among_finalists} из ${H.finalists.length + 1}`,
     `победитель ${fmt(H.finalists[0][1])}`],
    ["CV по случайным точкам обещал", fmt(cvP.score), `точность ${pct(cvP.accuracy)} — самообман`],
    [`Без скважины ${worst.well}`, fmt(rest.score_without_worst), "9 скважин — уровень победителя"],
  ].map(([t, v, s]) => `<div class="kpi"><small>${t}</small><b>${v}</b><span>${s}</span></div>`).join("");

  el("verdict").innerHTML = `
    <p><b>Итог.</b> Модель, выбранная по плану, на скрытых скважинах набрала ${fmt(H.score)} (точность ${pct(H.accuracy)}) —
      ${H.rank_among_finalists}-е место из ${H.finalists.length + 1}: впереди ${H.rank_among_finalists - 1} финалистов соревнования,
      позади один. Соревнование она бы не выиграла — и проект не про это. Кросс-валидация
      по скважинам предсказала результат честно (${fmt(cvW.score)}, открытый тест ${fmt(open.score)});
      случайное разбиение по точкам обещало ${fmt(cvP.score)} — в ${(H.score / cvP.score).toFixed(1).replace(".", ",")} раза лучше реальности.</p>
    <p>Треть всего штрафа дала одна скважина — <b>${worst.well}</b>, вскрывшая цехштейновые соли: галит модель
      назвала глиной, а примеров галита в обучении — три скважины из 98. CV это предвещал (галит распознан лишь
      в ${pct(E.halite_recall_cv)} случаев). Сервис помечает такую скважину сам — по низкой уверенности модели.</p>`;

  /* ---------- SVG-хелперы ---------- */
  function svg(w, h, body) { return `<svg viewBox="0 0 ${w} ${h}" role="img">${body}</svg>`; }
  function hbars(rows, opt) {
    const W = 760, L = opt.left || 230, R = 70, rowH = opt.rowH || 30, top = 24;
    const Hh = top + rows.length * rowH + 28;
    const x = (v) => L + (v - opt.min) / (opt.max - opt.min) * (W - L - R);
    let b = "";
    for (const t of opt.ticks) {
      b += `<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${Hh - 24}"/>`;
      b += `<text class="muted" x="${x(t)}" y="${Hh - 8}" text-anchor="middle">${fmt(t, 1)}</text>`;
    }
    if (opt.band) {
      b += `<rect x="${x(opt.band[0])}" y="${top - 6}" width="${x(opt.band[1]) - x(opt.band[0])}" height="${Hh - top - 18}"
              fill="${css("--accent")}" opacity="0.10"/>`;
      b += `<text class="muted" x="${x((opt.band[0] + opt.band[1]) / 2)}" y="${top - 10}" text-anchor="middle" font-size="11">${opt.bandLabel}</text>`;
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH, x0 = x(opt.min), x1 = x(r.v);
      b += `<text x="${L - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" ${r.bold ? 'font-weight="700"' : ""}>${esc(r.label)}</text>`;
      b += `<rect x="${Math.min(x0, x1)}" y="${y + 5}" width="${Math.abs(x1 - x0)}" height="${rowH - 10}" rx="4" fill="${r.color}"/>`;
      b += `<text x="${x1 + 6}" y="${y + rowH / 2 + 4}" ${r.bold ? 'font-weight="700"' : ""}>${fmt(r.v)}</text>`;
    });
    return svg(W, Hh, b);
  }

  /* ---------- валидация ---------- */
  el("chart-valid").innerHTML = hbars([
    { label: "CV по случайным точкам", v: cvP.score, color: css("--bad") },
    { label: "CV по скважинам (5 фолдов)", v: cvW.score, color: css("--accent") },
    { label: "Открытый тест, 10 скважин", v: open.score, color: css("--accent") },
    { label: "Скрытый тест, 10 скважин", v: H.score, color: css("--text"), bold: true },
  ], { min: -1, max: 0, ticks: [-1, -0.8, -0.6, -0.4, -0.2, 0], band: [E.subset10_p05, E.subset10_p95],
       bandLabel: "разброс для 10 скважин (5–95 %)" });

  /* ---------- финалисты ---------- */
  const lb = H.finalists.map(([n, s]) => ({ label: n, v: s, color: css("--muted") }))
    .concat([{ label: "Эта модель", v: H.score, color: css("--accent"), bold: true }])
    .sort((a, b) => b.v - a.v);
  el("chart-lb").innerHTML = hbars(lb, { min: -1, max: -0.3, ticks: [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3], rowH: 24 });

  /* ---------- лестница ---------- */
  const keys = ["const", "logreg", "lgbm_raw", "lgbm_strat", "lgbm_full"];
  el("ladder").innerHTML = `<table><thead><tr><th>Модель</th><th>CV по скважинам</th><th>Открытый тест</th>
      <th>Точность (open)</th><th>Macro-F1 (open)</th></tr></thead><tbody>${keys.map((k) => {
        const c = lad(k, "CV по скважинам"), o = lad(k, "open test");
        return `<tr class="${k === "lgbm_full" ? "hl" : ""}"><td>${esc(c.model)}</td><td>${fmt(c.score)}</td><td>${fmt(o.score)}</td>
          <td>${pct(o.accuracy, 1)}</td><td>${fmt(o.macro_f1, 2)}</td></tr>`;
      }).join("")}</tbody></table>
    <p class="note">Каждая ступень улучшает CV больше чем на 0,005 (порог из плана). Выбор класса с минимальным ожидаемым
      штрафом вместо самого вероятного дал +${fmt(cvW.score_bayes - cvW.score)} на CV — меньше порога, поэтому в финале
      обычный argmax. На скрытых скважинах штрафной выбор оказался бы хуже (${fmt(H.all_metrics.score_bayes)}).</p>`;

  /* ---------- калибровка ---------- */
  (function () {
    const W = 360, Hh = 300, L = 44, B = 36, T = 10, R = 10;
    const x = (v) => L + v * (W - L - R), y = (v) => Hh - B - v * (Hh - B - T);
    let b = "";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      b += `<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="${y(0)}" y2="${y(1)}"/><line class="gridline" x1="${x(0)}" x2="${x(1)}" y1="${y(t)}" y2="${y(t)}"/>`;
      b += `<text class="muted" x="${x(t)}" y="${Hh - B + 16}" text-anchor="middle">${pct(t)}</text>`;
      b += `<text class="muted" x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${pct(t)}</text>`;
    }
    b += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="${css("--muted")}" stroke-dasharray="4 4"/>`;
    const pts = D.reliability.map((r) => `${x(r.conf)},${y(r.acc)}`).join(" ");
    b += `<polyline points="${pts}" fill="none" stroke="${css("--bad")}" stroke-width="2.5"/>`;
    D.reliability.forEach((r) => { b += `<circle cx="${x(r.conf)}" cy="${y(r.acc)}" r="${2 + Math.sqrt(r.n) / 120}" fill="${css("--bad")}"/>`; });
    b += `<text x="${x(0.5)}" y="${Hh - 4}" text-anchor="middle" class="muted">заявленная уверенность</text>`;
    b += `<text x="${x(0.02)}" y="${y(0.95)}" class="muted">доля верных</text>`;
    el("chart-rel").innerHTML = svg(W, Hh, b) +
      `<p class="note" style="margin-top:2px">ECE ${fmt(E.ece, 3)} → ${fmt(E.ece_cv_scaled, 3)} после калибровки на CV;
       на открытом тесте ${fmt(E.ece_open, 3)} → ${fmt(E.ece_open_scaled, 3)}.</p>`;
  })();

  el("coverage").innerHTML = `<table><thead><tr><th>Модель решает сама</th><th>Точность</th><th>Score</th></tr></thead><tbody>${
    D.coverage.map((r) => `<tr class="${r.coverage === 1 ? "hl" : ""}"><td>${pct(r.coverage)} точек</td><td>${pct(r.accuracy, 1)}</td><td>${fmt(r.score)}</td></tr>`).join("")
  }</tbody></table>`;

  const ic = Object.fromEntries(D.interp.map((r) => [r.interp_conf, r]));
  el("interp").innerHTML = `<p style="margin:0"><b>Модель ошибается там же, где сомневался интерпретатор.</b>
    В разметке у каждой точки есть уверенность геолога (1 — уверен, 2–3 — сомневался). Где геолог уверен, модель права
    в ${pct(ic[1].accuracy)} случаев; где сомневался — в ${pct(ic[2].accuracy)}–${pct(ic[3].accuracy)}.
    Часть «ошибок» — спорные интервалы, где и эталон под вопросом.</p>`;

  /* ---------- разбор 17/4-1 и варианты ---------- */
  const ph = D.posthoc || [];
  const phRows = [...new Set(ph.map((r) => r.key))].map((k) => {
    const c = ph.find((r) => r.key === k && r.split === "CV по скважинам"), o = ph.find((r) => r.key === k && r.split === "open test");
    return `<tr><td>${esc(c.variant)}</td><td>${fmt(c.score)}</td><td>${fmt(o.score)}</td><td>${pct(c.halite_recall)}</td></tr>`;
  }).join("");
  el("posthoc").innerHTML = `
    <p style="margin-top:0">На скважине <b>${worst.well}</b> score ${fmt(worst.score)} при ${fmt(rest.score_without_worst)} на остальных девяти.
      ${pct(rest.worst_zechstein_share)} её интервала — цехштейновые соли; в обучении галит есть лишь в ${rest.halite_train_wells} скважинах из 98.
      ${rest.halite_as_shale.toLocaleString("ru")} точек галита модель назвала глиной: гамма-каротаж там ≈ ${rest.halite_gr_median} API —
      у глины так не бывает, но плотность и нейтронный каротаж записаны лишь на ${pct(rest.halite_rhob_share)} точек,
      а свита Smith Bank в обучающих скважинах почти всегда глинистая. Гипотеза: модель поверила стратиграфии больше,
      чем физике, — переобучилась на категориальный признак «свита».</p>
    <p>Скрытые скважины уже вскрыты, поэтому гипотезу проверяю только кросс-валидацией и открытым тестом —
      не подгоняя под финал:</p>
    <div class="scroll"><table><thead><tr><th>Вариант</th><th>CV по скважинам</th><th>Открытый тест</th><th>Галит распознан (CV)</th></tr></thead>
      <tbody>${phRows}</tbody></table></div>
    <p class="note">${esc(rest.conclusion)}</p>`;

  /* ---------- предупреждения сервиса на тестовых скважинах ---------- */
  el("warn-table").innerHTML = `<table><thead><tr><th>Скважина</th><th>Тест</th><th>Score</th><th style="text-align:left">Предупреждения сервиса</th></tr></thead><tbody>${
    D.warnings.map((r) => `<tr class="${r.warnings.length ? "hl" : ""}"><td>${esc(r.well)}</td><td>${r.split}</td><td>${fmt(r.score)}</td>
      <td style="text-align:left;font-weight:400;font-size:0.82rem">${r.warnings.length ? r.warnings.map(esc).join("<br>") : "—"}</td></tr>`).join("")
  }</tbody></table>`;

  /* ---------- пример ответа API ---------- */
  el("api-example").firstChild.textContent = JSON.stringify(D.api_example, null, 1);

  /* ---------- просмотр скважин ---------- */
  const sel = el("well"), cv = el("tracks"), tip = el("tip");
  D.wells.forEach((w, i) => sel.add(new Option(`Скважина ${w.well}`, i)));
  sel.value = String(D.wells.findIndex((w) => w.well === worst.well));
  el("chips").innerHTML = NAMES.map((n, i) => `<span><i style="background:${COLORS[i]}"></i>${n}</span>`).join("");
  const TRACKS = [
    { key: "GR", title: "GR, API", min: 0, max: 150, color: "#16a34a" },
    { key: "RHOB", title: "RHOB, г/см³", min: 1.95, max: 2.95, color: "#dc2626", key2: "NPHI", min2: 0.45, max2: -0.15, color2: "#2563eb", title2: "NPHI" },
    { key: "RDEP", title: "RDEP, Ом·м (лог)", min: -0.7, max: 3, color: "#111827", log: true },
    { key: "true", title: "Геолог", lith: true },
    { key: "pred", title: "Модель", lith: true },
    { key: "conf", title: "Уверенность", min: 0, max: 1, color: "#7c3aed" },
  ];
  let layout = null;

  function draw() {
    const w = D.wells[+sel.value];
    const pw = D.per_well_hidden.find((r) => r.well === w.well);
    el("well-stat").innerHTML = `score <b>${fmt(pw.score)}</b> · точность <b>${pct(pw.accuracy)}</b> · ${w.depth[0].toFixed(0)}–${w.depth[w.depth.length - 1].toFixed(0)} м`;
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, Hh = cv.clientHeight;
    cv.width = W * dpr; cv.height = Hh * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, Hh);
    const top = 38, bot = 8, left = 46, gap = 6;
    const n = w.depth.length, d0 = w.depth[0], d1 = w.depth[n - 1];
    const yOf = (d) => top + (d - d0) / (d1 - d0) * (Hh - top - bot);
    const tw = (W - left - gap * (TRACKS.length - 1)) / TRACKS.length;
    g.font = "11px -apple-system, Segoe UI, sans-serif"; g.fillStyle = css("--muted"); g.textAlign = "right";
    for (let k = 0; k <= 6; k++) {
      const d = d0 + (d1 - d0) * k / 6;
      g.fillText(d.toFixed(0), left - 6, yOf(d) + 4);
    }
    TRACKS.forEach((t, j) => {
      const x0 = left + j * (tw + gap);
      g.fillStyle = css("--card"); g.fillRect(x0, top, tw, Hh - top - bot);
      g.strokeStyle = css("--border"); g.strokeRect(x0, top, tw, Hh - top - bot);
      g.fillStyle = css("--text"); g.textAlign = "center"; g.font = "600 11px -apple-system, Segoe UI, sans-serif";
      g.fillText(t.title, x0 + tw / 2, 14);
      if (t.key2) { g.fillStyle = t.color2; g.fillText(t.title2, x0 + tw / 2, 28); }
      g.font = "11px -apple-system, Segoe UI, sans-serif";
      if (t.lith) {
        const arr = w[t.key];
        for (let i = 0; i < n; i++) {
          const ya = yOf(w.depth[i]), yb = i + 1 < n ? yOf(w.depth[i + 1]) : ya + 1;
          g.fillStyle = COLORS[arr[i]]; g.fillRect(x0 + 1, ya, tw - 2, Math.max(1, yb - ya + 0.5));
        }
        return;
      }
      const line = (key, mn, mx, color, log) => {
        g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); let pen = false;
        for (let i = 0; i < n; i++) {
          let v = w[key][i];
          if (v == null || (log && v <= 0)) { pen = false; continue; }
          if (log) v = Math.log10(v);
          const xx = x0 + Math.max(0, Math.min(1, (v - mn) / (mx - mn))) * tw, yy = yOf(w.depth[i]);
          if (pen) g.lineTo(xx, yy); else g.moveTo(xx, yy);
          pen = true;
        }
        g.stroke();
      };
      line(t.key, t.min, t.max, t.color, t.log);
      if (t.key2) line(t.key2, t.min2, t.max2, t.color2);
    });
    layout = { w, top, bot, left, tw, gap, Hh, d0, d1, n };
  }

  cv.addEventListener("mousemove", (e) => {
    if (!layout) return;
    const r = cv.getBoundingClientRect(), y = e.clientY - r.top, { w, top, bot, Hh, d0, d1, n } = layout;
    if (y < top || y > Hh - bot) { tip.style.display = "none"; return; }
    const d = d0 + (y - top) / (Hh - top - bot) * (d1 - d0);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (w.depth[m] < d) lo = m; else hi = m; }
    const i = Math.abs(w.depth[lo] - d) < Math.abs(w.depth[hi] - d) ? lo : hi;
    const v = (k, dd) => (w[k][i] == null ? "—" : w[k][i].toFixed(dd).replace(".", ","));
    const ok = w.true[i] === w.pred[i];
    tip.innerHTML = `<b>${w.depth[i].toFixed(1).replace(".", ",")} м</b> · ${esc(w.group[i] || "")}<br>
      GR ${v("GR", 0)} · RHOB ${v("RHOB", 2)} · NPHI ${v("NPHI", 2)} · RDEP ${v("RDEP", 1)}<br>
      геолог: <b>${NAMES[w.true[i]]}</b><br>модель: <b style="color:${ok ? css("--good") : css("--bad")}">${NAMES[w.pred[i]]}</b> (${pct(w.conf[i])})`;
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 260) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  cv.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  sel.addEventListener("change", draw);
  window.addEventListener("resize", draw);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  draw();
})();
