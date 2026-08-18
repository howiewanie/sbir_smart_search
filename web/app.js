const el = (id) => document.getElementById(id);

const state = { data: null, theme: null };

/* ---------- formatting ---------- */

const money = (v) => {
  if (!v) return "$0";
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${Math.round(v / 1e3)}K`;
  return `$${Math.round(v)}`;
};

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

const titleCase = (s) =>
  s.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());

/* ---------- charts, drawn as plain elements ---------- */

function barRows(target, rows, { valueKey = "awards", label = (r) => r.name,
                                 format = (v) => v } = {}) {
  const max = Math.max(...rows.map((r) => r[valueKey]), 1);
  el(target).innerHTML = rows
    .map(
      (r) => `<div class="bar-row">
        <span class="bar-label" title="${esc(label(r))}">${esc(label(r))}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${(r[valueKey] / max) * 100}%"></span></span>
        <span class="bar-value">${format(r[valueKey])}</span>
      </div>`
    )
    .join("");
}

function columnChart(target, points) {
  if (!points.length) {
    el(target).innerHTML = `<p class="hint">Not enough dated awards to plot.</p>`;
    return;
  }
  const max = Math.max(...points.map((p) => p.awards), 1);
  const step = points.length > 18 ? Math.ceil(points.length / 9) : 1;
  el(target).innerHTML =
    `<div class="cols">${points
      .map((p, i) => {
        const h = Math.max((p.awards / max) * 100, p.awards ? 4 : 0);
        const tick = i % step === 0 || i === points.length - 1 ? String(p.year).slice(2) : "";
        return `<span class="col" title="${p.year}: ${p.awards} awards, ${money(p.funding)}">
                  <span class="col-fill" style="height:${h}%"></span>
                  <span class="col-tick">${tick}</span>
                </span>`;
      })
      .join("")}</div>`;
}

function interpretation(d) {
  el("interpretation").innerHTML = (d.reading || [])
    .map((s) => `<li>${esc(s)}</li>`).join("");
}

/* ---------- award cards ---------- */

function awardCard(a, rank) {
  const site = a.website && /^https?:\/\//i.test(a.website) ? a.website : null;
  const company = esc(titleCase(a.company || "Unknown"));
  const tags = [a.agency, a.branch, a.phase, a.program, a.year,
                [a.city, a.state].filter(Boolean).join(", ")].filter(Boolean);
  const extra = a.related_awards > 1
    ? `<span class="tag related">+${a.related_awards - 1} related award${a.related_awards > 2 ? "s" : ""} &middot; ${money(a.related_funding)} total</span>`
    : "";
  return `<li class="hit" data-id="${a.id}">
    <div class="hit-top">
      <h4>${rank}. ${esc(a.title || "Untitled award")}</h4>
      <span class="tag amount">${money(a.amount)}</span>
    </div>
    <p class="company">${site ? `<a href="${esc(site)}" target="_blank" rel="noopener">${company}</a>` : company}</p>
    <div class="meta">${tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}${extra}</div>
    ${a.abstract ? `<div class="abstract"><p class="clamp">${esc(a.abstract)}</p>
       <button type="button" data-expand>Show full abstract</button></div>` : ""}
  </li>`;
}

function renderAwards() {
  const d = state.data;
  const shown = state.theme
    ? d.awards.filter((a) => state.theme.award_ids.includes(a.id))
    : d.awards;
  el("awards").innerHTML = shown.map((a, i) => awardCard(a, i + 1)).join("");
  const ev = d.evidence;
  el("evidence-note").innerHTML = state.theme
    ? `${shown.length} awards mentioning <b>${esc(state.theme.label)}</b>. <button type="button" class="link" id="clear-theme">Show all ${d.awards.length}</button>`
    : `The ${ev.size} awards below were selected from the ${ev.considered} closest matches, after collapsing ${ev.duplicates_removed} repeat filings of the same project and limiting any one company to ${ev.per_company_cap}.`;
  const clear = el("clear-theme");
  if (clear) clear.addEventListener("click", () => { state.theme = null; renderThemes(); renderAwards(); });
}

function renderThemes() {
  el("themes").innerHTML = state.data.themes
    .map((t) => `<button type="button" class="theme${state.theme && state.theme.label === t.label ? " on" : ""}"
        data-theme="${esc(t.label)}">${esc(t.label)} <span>${t.awards}</span></button>`)
    .join("");
}

/* ---------- page assembly ---------- */

function render(d) {
  state.data = d;
  state.theme = null;
  const t = d.totals;

  el("topic").textContent = titleCase(d.query);
  el("lede").textContent =
    `Across the ${t.awards} most closely related SBIR/STTR awards, ${money(t.funding)} in federal funding reached ` +
    `${t.companies} companies through ${t.agencies} agencies between ${t.years[0]} and ${t.years[1]}.`;

  el("figures").innerHTML = [
    [t.awards, "awards examined"],
    [money(t.funding), "identified funding"],
    [t.agencies, "agencies"],
    [t.companies, "companies"],
  ].map(([v, k]) => `<div class="figure"><b>${v}</b><span>${k}</span></div>`).join("");

  const cov = d.coverage || {};
  el("basis").textContent =
    `Evidence drawn from awards recorded between ${cov.first_year} and ${cov.complete_through}.` +
    (cov.partial_years && cov.partial_years.length
      ? ` ${cov.partial_years.join(", ")} is present in the export but incomplete, so it is excluded from totals and charts.`
      : "");

  columnChart("chart-timeline", d.timeline.points);
  barRows("chart-agency", d.agencies.slice(0, 6), { valueKey: "funding", format: money });
  barRows("chart-phase", d.phases);
  barRows("chart-program", d.programs);

  renderThemes();

  const company = (c, detail) => `<div class="firm">
      <b>${esc(titleCase(c.company))}</b>
      <span>${detail(c)}</span>
    </div>`;
  el("recurring").innerHTML = d.ecosystem.recurring.length
    ? d.ecosystem.recurring.map((c) => company(c,
        (x) => `${x.awards_here} award${x.awards_here > 1 ? "s" : ""} here &middot; ${x.total_awards} overall &middot; ${money(x.total_funding)} &middot; ${x.first_year}-${x.last_year}`)).join("")
    : `<p class="hint">No firm in this evidence has a broad award history.</p>`;
  el("progressed").innerHTML = d.ecosystem.progressed.length
    ? d.ecosystem.progressed.map((c) => company(c,
        (x) => `${x.topic_progressed} project${x.topic_progressed > 1 ? "s" : ""} advanced in this area &middot; ${x.total_awards} awards overall`)).join("")
    : `<p class="hint">No Phase I to Phase II progression found within this technology area.</p>`;

  renderAwards();
  interpretation(d);

  el("report").hidden = false;
  el("loading").hidden = true;
  el("ask").classList.add("compact");
}

/* ---------- running a search ---------- */

const STEPS = [
  "Searching historical awards…",
  "Identifying relevant companies and programs…",
  "Analysing funding patterns…",
  "Assembling the evidence…",
];

async function run(query) {
  if (!query.trim()) return;
  el("report").hidden = true;
  el("loading").hidden = false;
  el("ask").classList.add("compact");

  let step = 0;
  el("loading-step").textContent = STEPS[0];
  const ticker = setInterval(() => {
    step = Math.min(step + 1, STEPS.length - 1);
    el("loading-step").textContent = STEPS[step];
  }, 700);

  try {
    const res = await fetch(`/api/research?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error((await res.json()).detail || `Request failed (${res.status})`);
    render(await res.json());
    const url = new URL(window.location);
    url.searchParams.set("q", query);
    history.replaceState(null, "", url);
  } catch (err) {
    el("loading").innerHTML = `<div class="loading-card"><p>${esc(err.message)}</p></div>`;
  } finally {
    clearInterval(ticker);
  }
}

/* ---------- events ---------- */

el("research-form").addEventListener("submit", (e) => {
  e.preventDefault();
  run(el("query").value);
});

el("examples").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-q]");
  if (!b) return;
  el("query").value = b.dataset.q;
  run(b.dataset.q);
});

el("themes").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-theme]");
  if (!b) return;
  const picked = state.data.themes.find((t) => t.label === b.dataset.theme);
  state.theme = state.theme && state.theme.label === picked.label ? null : picked;
  renderThemes();
  renderAwards();
});

el("awards").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-expand]");
  if (!b) return;
  const clamped = b.previousElementSibling.classList.toggle("clamp");
  b.textContent = clamped ? "Show full abstract" : "Hide abstract";
});

function downloadReport(ext) {
  if (!state.data) return;
  const link = document.createElement("a");
  link.href = `/api/research.${ext}?q=${encodeURIComponent(state.data.query)}`;
  link.download = "";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

el("download-pdf").addEventListener("click", () => downloadReport("pdf"));
el("download-docx").addEventListener("click", () => downloadReport("docx"));
el("brief").addEventListener("click", () => window.print());

el("restart").addEventListener("click", () => {
  el("report").hidden = true;
  el("ask").classList.remove("compact");
  el("query").value = "";
  el("query").focus();
  history.replaceState(null, "", window.location.pathname);
});

/* ---------- boot ---------- */

(async function boot() {
  try {
    const stats = await fetch("/api/stats").then((r) => r.json());
    if (!stats.ready) {
      el("dataset").textContent = "No index yet";
      el("loading").hidden = false;
      el("loading").innerHTML =
        `<div class="loading-card"><p>Build the index first: <code>python -m sbir setup</code></p></div>`;
      return;
    }
    const c = stats.coverage || {};
    const t = stats.totals || {};
    el("dataset").innerHTML =
      `${(t.awards || stats.awards).toLocaleString()} awards &middot; ${c.first_year}&ndash;${c.complete_through}
       &middot; ${(t.companies || 0).toLocaleString()} companies`;

    const incoming = new URLSearchParams(window.location.search).get("q");
    if (incoming) {
      el("query").value = incoming;
      run(incoming);
    }
  } catch {
    el("dataset").textContent = "Unavailable";
  }
})();
