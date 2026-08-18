const el = (id) => document.getElementById(id);

const state = {
  offset: 0,
  hits: [],
  total: 0,
  ready: false,
};

const CHECKBOX_FACETS = ["program", "phase", "agency"];

const money = (value) => {
  if (!value) return "Amount not listed";
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `$${Math.round(value / 1e3)}K`;
  return `$${value.toLocaleString()}`;
};

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

/* ---------- reading the form ---------- */

function checkedValues(facet) {
  return Array.from(
    document.querySelectorAll(`#facet-${facet} input:checked`)
  ).map((input) => input.value);
}

function currentParams() {
  const params = new URLSearchParams();
  const query = el("query").value.trim();
  if (query) params.set("q", query);
  params.set("mode", el("mode").value);
  params.set("sort", el("sort").value);

  for (const facet of CHECKBOX_FACETS) {
    for (const value of checkedValues(facet)) params.append(facet, value);
  }
  if (el("facet-state").value) params.append("state", el("facet-state").value);

  for (const field of ["year_min", "year_max", "amount_min", "amount_max"]) {
    const value = el(field).value.trim();
    if (value !== "") params.set(field, value);
  }
  for (const flag of ["women_owned", "hubzone", "disadvantaged"]) {
    if (el(flag).checked) params.set(flag, "true");
  }
  return params;
}

/* ---------- rendering ---------- */

function renderHit(hit, rank) {
  const website = hit.website && /^https?:\/\//i.test(hit.website) ? hit.website : null;
  const company = escapeHtml(hit.company || "Unknown company");
  const tags = [
    hit.agency,
    hit.branch,
    hit.phase,
    hit.program,
    hit.year,
    [hit.city, hit.state].filter(Boolean).join(", "),
  ].filter(Boolean);

  const abstract = hit.abstract
    ? `<div class="abstract"><p class="clamp">${escapeHtml(hit.abstract)}</p>
         <button type="button" data-expand>Show full abstract</button></div>`
    : "";

  const match = hit.similarity != null
    ? `<span class="match">${Math.round(hit.similarity * 100)}% match</span>`
    : "";

  return `<li class="hit">
    <div class="hit-top">
      <h3>${rank}. ${escapeHtml(hit.title || "Untitled award")}</h3>
      ${match}
    </div>
    <p class="company">${website ? `<a href="${escapeHtml(website)}" rel="noopener" target="_blank">${company}</a>` : company}</p>
    <div class="meta">
      <span class="tag amount">${money(hit.amount)}</span>
      ${tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}
    </div>
    ${abstract}
  </li>`;
}

function renderNotice(title, body) {
  el("hits").innerHTML = `<div class="notice"><h3>${title}</h3>${body}</div>`;
}

function renderEmpty(query, mode) {
  if (mode === "company" && query) {
    // Usually this is either a typo or one of the big primes, which are not
    // eligible for these programs in the first place.
    renderNotice(
      "No company matched",
      `<p>Nothing in the database is named like <b>${escapeHtml(query)}</b>.</p>
       <p>SBIR and STTR fund small businesses, so large prime contractors never
       appear. Try a shorter form of the name, or switch back to
       <b>Topic</b> to search by subject instead.</p>`
    );
  } else if (!query) {
    renderNotice(
      "Nothing to show",
      "<p>These filters exclude every award. Try clearing one.</p>"
    );
  } else {
    renderNotice("No matches", "<p>Try broader wording or clear a filter.</p>");
  }
}

function render() {
  el("hits").innerHTML = state.hits.map((hit, i) => renderHit(hit, i + 1)).join("");
  el("load-more").hidden = state.hits.length >= state.total;
}

/* ---------- searching ---------- */

async function runSearch({ append = false } = {}) {
  if (!state.ready) return;
  state.offset = append ? state.offset + 20 : 0;

  const params = currentParams();
  params.set("limit", "20");
  params.set("offset", String(state.offset));

  document.body.classList.add("busy");
  try {
    const response = await fetch(`/api/search?${params}`);
    if (!response.ok) throw new Error(`search failed (${response.status})`);
    const data = await response.json();

    state.hits = append ? state.hits.concat(data.results) : data.results;
    state.total = data.total;

    if (!state.hits.length) {
      renderEmpty(el("query").value.trim(), el("mode").value);
      el("load-more").hidden = true;
      el("summary").innerHTML = "No results.";
    } else {
      render();
      const shown = state.hits.length;
      const cap = data.truncated ? "+" : "";
      el("summary").innerHTML =
        `Showing <b>${shown}</b> of <b>${state.total.toLocaleString()}${cap}</b> matches
         &middot; ${data.took_ms} ms`;
    }

    const exportParams = currentParams();
    exportParams.set("limit", "500");
    const exportLink = el("export");
    exportLink.href = `/api/export.csv?${exportParams}`;
    exportLink.setAttribute("aria-disabled", state.hits.length ? "false" : "true");

    const url = new URL(window.location);
    url.search = params.toString();
    history.replaceState(null, "", url);
  } catch (error) {
    renderNotice("Something went wrong", `<p>${escapeHtml(error.message)}</p>`);
  } finally {
    document.body.classList.remove("busy");
  }
}

/* ---------- setup ---------- */

function buildCheckboxes(facet, values) {
  el(`facet-${facet}`).innerHTML = values
    .map(
      (value) => `<label><input type="checkbox" value="${escapeHtml(value)}">
        <span>${escapeHtml(value)}</span></label>`
    )
    .join("");
}

async function loadIndex() {
  const stats = await fetch("/api/stats").then((r) => r.json());

  if (!stats.ready) {
    el("dataset").textContent = "No index yet";
    renderNotice(
      "Build the index to get started",
      `<p>Download the award data and embed it once:</p>
       <p><code>python -m sbir setup</code></p>
       <p>Want to try it quickly? <code>python -m sbir setup --since 2020</code>
       indexes the most recent awards in a couple of minutes.</p>`
    );
    el("summary").textContent = "Index not built.";
    return;
  }

  const totals = stats.totals || {};
  const years = stats.years || [];
  el("dataset").innerHTML =
    `${(totals.awards || stats.awards).toLocaleString()} awards &middot;
     ${years[0]}&ndash;${years[1]} &middot;
     ${(totals.companies || 0).toLocaleString()} companies &middot;
     $${((totals.funding || 0) / 1e9).toFixed(1)}B`;

  const facets = await fetch("/api/facets").then((r) => r.json());
  for (const facet of CHECKBOX_FACETS) buildCheckboxes(facet, facets[facet] || []);
  el("facet-state").innerHTML =
    '<option value="">Any state</option>' +
    (facets.state || [])
      .map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`)
      .join("");

  if (years.length === 2) {
    el("year_min").placeholder = years[0];
    el("year_max").placeholder = years[1];
  }

  state.ready = true;

  // Restore a shared link, otherwise show the newest awards as a starting point.
  const incoming = new URLSearchParams(window.location.search);
  if (incoming.toString()) {
    applyParams(incoming);
  } else {
    el("sort").value = "newest";
  }
  runSearch();
}

function applyParams(params) {
  el("query").value = params.get("q") || "";
  if (params.get("mode")) el("mode").value = params.get("mode");
  if (params.get("sort")) el("sort").value = params.get("sort");
  for (const facet of CHECKBOX_FACETS) {
    const wanted = new Set(params.getAll(facet));
    document
      .querySelectorAll(`#facet-${facet} input`)
      .forEach((input) => (input.checked = wanted.has(input.value)));
  }
  if (params.get("state")) el("facet-state").value = params.get("state");
  for (const field of ["year_min", "year_max", "amount_min", "amount_max"]) {
    el(field).value = params.get(field) || "";
  }
  for (const flag of ["women_owned", "hubzone", "disadvantaged"]) {
    el(flag).checked = params.get(flag) === "true";
  }
}

function clearFilters() {
  document.querySelectorAll(".filters input[type=checkbox]").forEach((c) => (c.checked = false));
  document.querySelectorAll(".filters input[type=number]").forEach((i) => (i.value = ""));
  el("facet-state").value = "";
  runSearch();
}

el("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
});

el("filters").addEventListener("change", () => runSearch());

el("sort").addEventListener("change", () => runSearch());
el("mode").addEventListener("change", () => runSearch());
el("clear-filters").addEventListener("click", clearFilters);
el("load-more").addEventListener("click", () => runSearch({ append: true }));

el("examples").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-q]");
  if (!button) return;
  el("query").value = button.dataset.q;
  el("mode").value = "auto";
  el("sort").value = "relevance";
  runSearch();
});

el("hits").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-expand]");
  if (!button) return;
  const paragraph = button.previousElementSibling;
  const clamped = paragraph.classList.toggle("clamp");
  button.textContent = clamped ? "Show full abstract" : "Hide abstract";
});

loadIndex();
