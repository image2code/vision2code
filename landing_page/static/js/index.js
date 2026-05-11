const paths = {
  leaderboard: "static/data/leaderboard_data.json",
  examples: "static/data/examples_data.json",
  viewer: "static/data/viewer_data.json",
};

const state = {
  viewerPage: 1,
  viewerPageSize: 6,
  filteredViewerSamples: [],
};

const datasetOrder = [
  "ChartQA",
  "dvqa",
  "figureqa",
  "matplotlib",
  "Geoperception",
  "GEOQA_8K_R1V",
  "geometry3k",
  "Graph-Algorithms",
  "GraphVQA-Swift",
  "ChemVQA-2K",
  "EEE-Bench",
  "Physics",
  "OlympiadBench",
  "DocVQA",
  "spatialvlm_qa",
];

const datasetLabels = {
  ChartQA: "ChartQA",
  dvqa: "DVQA",
  figureqa: "FigQA",
  matplotlib: "MPL",
  Geoperception: "Geoper.",
  GEOQA_8K_R1V: "GEOQA",
  geometry3k: "Geom3K",
  "Graph-Algorithms": "Graph Alg.",
  "GraphVQA-Swift": "GraphVQA",
  "ChemVQA-2K": "ChemVQA",
  "EEE-Bench": "EEE",
  Physics: "Physics",
  OlympiadBench: "Olymp.",
  DocVQA: "DocVQA",
  spatialvlm_qa: "Spatial",
};

function byId(id) {
  return document.getElementById(id);
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

function formatNumber(value) {
  return Number(value).toLocaleString("en-US");
}

function formatScore(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "NA";
  }
  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "NA";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function scoreBandClass(value) {
  const score = Number(value);
  if (Number.isNaN(score)) return "score-na";
  if (score < 1) return "score-band-red";
  if (score < 2) return "score-band-orange";
  if (score < 3) return "score-band-yellow-low";
  if (score < 4) return "score-band-yellow-high";
  return "score-band-green";
}

function escapeHtml(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function splitLabel(split) {
  const labels = {
    filtered_test: "Test",
    filtered_test_mini: "Test mini",
  };
  return labels[split] || split;
}

function initLeaderboard(data) {
  const splitSelect = byId("leaderboard-split");
  splitSelect.innerHTML = data.splits
    .map((split) => `<option value="${escapeHtml(split)}">${escapeHtml(splitLabel(split))}</option>`)
    .join("");
  splitSelect.value = data.default_split;
  splitSelect.addEventListener("change", () => renderLeaderboard(data));
  renderLeaderboard(data);
}

function renderLeaderboard(data) {
  const split = byId("leaderboard-split").value;
  const rows = data.rows
    .filter((row) => row.split === split)
    .slice()
    .sort((a, b) => b.score - a.score);

  const availableDatasets = rows.length ? new Set(rows[0].datasets.map((item) => item.dataset)) : new Set();
  const datasets = datasetOrder.filter((dataset) => availableDatasets.has(dataset));
  byId("leaderboard-head").innerHTML = `
    <tr>
      <th>Rank</th>
      <th class="model-col">Model</th>
      ${datasets.map((dataset) => `<th>${escapeHtml(datasetLabels[dataset] || dataset)}</th>`).join("")}
      <th>Render success</th>
      <th>Final score</th>
    </tr>
  `;

  const tbody = byId("leaderboard-table").querySelector("tbody");
  tbody.innerHTML = rows
    .map((row, index) => {
      const scoresByDataset = new Map(row.datasets.map((item) => [item.dataset, item.score]));
      return `
        <tr>
          <td>${index + 1}</td>
          <td><strong>${escapeHtml(row.model)}</strong></td>
          ${datasets
            .map((dataset) => {
              const score = scoresByDataset.get(dataset);
              return `<td class="score-cell ${scoreBandClass(score)}">${formatScore(score)}</td>`;
            })
            .join("")}
          <td class="render-cell">${formatPercent(row.render_success_rate)}</td>
          <td class="score-cell final-score ${scoreBandClass(row.score)}">${formatScore(row.score)}</td>
        </tr>
      `;
    })
    .join("");
}

function initExamples(data) {
  const datasets = [...new Set(data.examples.map((item) => item.dataset))];
  const select = byId("example-filter");
  select.innerHTML = datasets
    .map((dataset) => `<option value="${escapeHtml(dataset)}">${escapeHtml(dataset)}</option>`)
    .join("");
  select.value = datasets[0] || "";
  select.addEventListener("change", () => renderExamples(data));
  renderExamples(data);
}

function renderExamples(data) {
  const selected = byId("example-filter").value;
  const examples = data.examples.filter((item) => item.dataset === selected);

  byId("examples-grid").innerHTML = examples
    .map(
      (item) => `
        <article class="example-card">
          <img src="${escapeHtml(item.image)}" alt="Composite recreation and rating example for ${escapeHtml(
            item.dataset,
          )}" loading="lazy">
          <div class="example-body">
            <div class="meta-line">
              <span class="tag">${escapeHtml(item.domain)}</span>
              <span class="tag alt">${escapeHtml(item.dataset)}</span>
            </div>
          </div>
        </article>
      `,
    )
    .join("");
}

function initViewer(data) {
  const domainSelect = byId("viewer-domain");
  const datasetSelect = byId("viewer-dataset");

  domainSelect.innerHTML = ["All domains", ...data.domains.map((item) => item.name)]
    .map((domain) => `<option value="${escapeHtml(domain)}">${escapeHtml(domain)}</option>`)
    .join("");

  function refreshDatasetOptions() {
    const domain = domainSelect.value;
    const datasets = data.datasets
      .filter((item) => domain === "All domains" || item.domain === domain)
      .map((item) => item.name);
    datasetSelect.innerHTML = ["All datasets", ...datasets]
      .map((dataset) => `<option value="${escapeHtml(dataset)}">${escapeHtml(dataset)}</option>`)
      .join("");
  }

  domainSelect.addEventListener("change", () => {
    refreshDatasetOptions();
    state.viewerPage = 1;
    renderViewer(data);
  });
  datasetSelect.addEventListener("change", () => {
    state.viewerPage = 1;
    renderViewer(data);
  });
  byId("viewer-prev").addEventListener("click", () => {
    state.viewerPage = Math.max(1, state.viewerPage - 1);
    renderViewer(data);
  });
  byId("viewer-next").addEventListener("click", () => {
    state.viewerPage += 1;
    renderViewer(data);
  });

  byId("viewer-grid").addEventListener("click", (event) => {
    const card = event.target.closest("[data-image-src]");
    if (!card) return;
    openModal(card.dataset.imageSrc, card.dataset.imageAlt);
  });

  refreshDatasetOptions();
  renderViewer(data);
}

function renderViewer(data) {
  const domain = byId("viewer-domain").value;
  const dataset = byId("viewer-dataset").value;

  const filtered = data.samples.filter((sample) => {
    const matchesDomain = domain === "All domains" || sample.domain === domain;
    const matchesDataset = dataset === "All datasets" || sample.dataset === dataset;
    return matchesDomain && matchesDataset;
  });

  state.filteredViewerSamples = filtered;
  const pageCount = Math.max(1, Math.ceil(filtered.length / state.viewerPageSize));
  state.viewerPage = Math.min(state.viewerPage, pageCount);
  const start = (state.viewerPage - 1) * state.viewerPageSize;
  const pageItems = filtered.slice(start, start + state.viewerPageSize);

  byId("viewer-status").textContent = `${formatNumber(filtered.length)} of ${formatNumber(
    data.sample_count,
  )} test samples`;

  byId("viewer-grid").innerHTML = pageItems
    .map((sample) => {
      const image = sample.images[0];
      const imageHtml = image
        ? `<img src="${escapeHtml(image.src)}" alt="Source image for ${escapeHtml(
            sample.dataset,
          )}" loading="lazy">`
        : `<div class="missing-image">No image</div>`;
      const imageSrc = image ? image.src : "";
      return `
        <article class="viewer-card" data-image-src="${escapeHtml(imageSrc)}" data-image-alt="${escapeHtml(
          sample.dataset,
        )} source image">
          ${imageHtml}
          <div class="viewer-body">
            <div class="meta-line">
              <span class="tag">${escapeHtml(sample.domain)}</span>
              <span class="tag alt">${escapeHtml(sample.dataset)}</span>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  byId("viewer-page").textContent = `Page ${state.viewerPage} of ${pageCount}`;
  byId("viewer-prev").disabled = state.viewerPage <= 1;
  byId("viewer-next").disabled = state.viewerPage >= pageCount;
}

function openModal(src, alt) {
  if (!src) return;
  const modal = byId("image-modal");
  const image = byId("modal-image");
  image.src = src;
  image.alt = alt || "Source image preview";
  modal.classList.add("is-active");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  const modal = byId("image-modal");
  const image = byId("modal-image");
  modal.classList.remove("is-active");
  modal.setAttribute("aria-hidden", "true");
  image.src = "";
}

function initModal() {
  byId("image-modal").addEventListener("click", (event) => {
    if (event.target.id === "image-modal" || event.target.classList.contains("modal-background")) {
      closeModal();
    }
  });
  document.querySelector(".modal-close").addEventListener("click", closeModal);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModal();
  });
}

async function main() {
  try {
    const [leaderboardData, examplesData, viewerData] = await Promise.all([
      loadJson(paths.leaderboard),
      loadJson(paths.examples),
      loadJson(paths.viewer),
    ]);

    initLeaderboard(leaderboardData);
    initExamples(examplesData);
    initViewer(viewerData);
    initModal();
  } catch (error) {
    console.error(error);
    byId("viewer-status").textContent = "Site data failed to load.";
  }
}

main();
