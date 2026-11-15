/*
  NexusEd AI frontend (plain JavaScript, no framework).

  Sections, in order:
    1. Utilities        – small helpers ($, escapeHtml, formatBytes, toast)
    2. API helpers      – every call to the FastAPI backend
    3. Screens / state  – which screen is visible + shared app state
    4. Health chips     – Ollama / Qdrant status in the top bar
    5. Upload           – drag & drop, recent documents
    6. Polling          – processing screen + stepper
    7. Graph rendering  – Cytoscape mind map, radial layout, hover effects
    8. Drawer           – concept details panel
    9. Quiz             – one-question check with instant feedback
   10. Progress         – "N of M explored", saved in the browser
   11. Start-up         – wire everything together
*/

// =====================================================================
// 1. Utilities
// =====================================================================
const $ = (id) => document.getElementById(id);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Make text safe to put inside HTML (document text could contain "<script>"). */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

/** 1536000 -> "1.5 MB" */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Show a small notification in the corner for a few seconds. */
function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast glass-strong ${type}`;
  el.innerHTML = `${type === "error" ? "⚠️" : "✨"} ${escapeHtml(message)}`;
  $("toasts").appendChild(el);
  setTimeout(() => el.classList.add("hide"), 4500);
  setTimeout(() => el.remove(), 4900);
}

/** Read/write localStorage safely (it can be blocked in private windows). */
const storage = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
  },
};

// =====================================================================
// 2. API helpers
// =====================================================================
/** Call the backend and return JSON; throws an Error with the server's message. */
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Can't reach the NexusEd server. Is it running?");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    // FastAPI puts error text in "detail" (a string, or a list for validation errors).
    const detail = typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

const API = {
  health: () => api("/api/health"),
  listDocuments: () => api("/api/documents"),
  getDocument: (id) => api(`/api/documents/${id}`),
  retry: (id) => api(`/api/documents/${id}/retry`, { method: "POST" }),
  remove: (id) => api(`/api/documents/${id}`, { method: "DELETE" }),
  upload(file) {
    const form = new FormData();
    form.append("file", file);
    return api("/api/upload", { method: "POST", body: form });
  },
  explain: (docId, nodeId, level) =>
    api("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, node_id: nodeId, level }),
    }),
};

// =====================================================================
// 3. Screens / state
// =====================================================================
const state = {
  docId: null, // document currently shown
  doc: null, // its full data incl. mind map
  cy: null, // the Cytoscape graph instance
  pollTimer: null,
  selectedNode: null, // node open in the drawer
  explainRequest: 0, // counter so a slow old answer can't overwrite a newer one
};

/** Show one screen (upload | processing | map) and hide the others. */
function showScreen(name) {
  for (const s of ["upload", "processing", "map"]) {
    $(`screen-${s}`).classList.toggle("hidden", s !== name);
  }
  $("map-toolbar").classList.toggle("hidden", name !== "map");
  $("map-toolbar").classList.toggle("flex", name === "map");
  if (name !== "map") closeDrawer();
  if (name !== "processing") stopPolling();
  if (name === "upload") {
    history.replaceState(null, "", location.pathname); // clear #doc=… from the URL
    loadRecent();
  }
}

/** Open a document: show the map if ready, otherwise the processing screen. */
async function openDocument(docId, fileInfo = null) {
  state.docId = docId;
  history.replaceState(null, "", `#doc=${docId}`); // so a page refresh comes back here
  try {
    const doc = await API.getDocument(docId);
    if (doc.status === "ready") return renderMap(doc);
    showProcessing(doc, fileInfo);
  } catch (err) {
    toast(err.message, "error");
    showScreen("upload");
  }
}

// =====================================================================
// 4. Health chips
// =====================================================================
function setChip(id, status, tooltip) {
  const chip = $(id);
  chip.classList.remove("ok", "warn", "bad");
  chip.classList.add(status);
  chip.title = tooltip;
}

async function refreshHealth() {
  try {
    const h = await API.health();
    if (!h.ollama) setChip("chip-ollama", "bad", "Ollama is not reachable. Start it with: ollama serve");
    else if (!h.model_available) setChip("chip-ollama", "warn", `Ollama is running, but model "${h.model}" is missing. Run: ollama pull ${h.model}`);
    else setChip("chip-ollama", "ok", `Ollama is running with ${h.model}`);
    setChip("chip-qdrant", h.qdrant ? "ok" : "bad", h.qdrant ? "Qdrant vector database is running" : "Qdrant is not reachable. Start it with docker compose up -d qdrant");
  } catch {
    setChip("chip-ollama", "bad", "NexusEd server is not reachable");
    setChip("chip-qdrant", "bad", "NexusEd server is not reachable");
  }
}

// =====================================================================
// 5. Upload
// =====================================================================
const ALLOWED = [".pdf", ".docx", ".pptx"];
const MAX_MB = 25;

/** Check the file in the browser first, so obvious mistakes fail instantly. */
function validateFile(file) {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (ext === ".doc" || ext === ".ppt") return `Old ${ext} files aren't supported. Please "Save As" ${ext}x first.`;
  if (!ALLOWED.includes(ext)) return "Please choose a PDF, DOCX or PPTX file.";
  if (file.size > MAX_MB * 1024 * 1024) return `That file is larger than ${MAX_MB} MB.`;
  if (file.size === 0) return "That file is empty.";
  return null;
}

async function handleFile(file) {
  const problem = validateFile(file);
  if (problem) return toast(problem, "error");
  $("drop-title").textContent = file.name;
  $("drop-sub").textContent = `${formatBytes(file.size)} · uploading…`;
  try {
    const { doc_id } = await API.upload(file);
    await openDocument(doc_id, { name: file.name, size: file.size });
  } catch (err) {
    toast(err.message, "error");
  } finally {
    resetDropzone();
  }
}

function resetDropzone() {
  $("drop-title").textContent = "Drop your file here";
  $("drop-sub").textContent = `or click to browse · PDF, DOCX or PPTX · up to ${MAX_MB} MB`;
  $("file-input").value = "";
}

function setupDropzone() {
  const zone = $("dropzone");
  // dragenter/dragover must call preventDefault, or the browser opens the file itself.
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("drag-over"); })
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("drag-over"); })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); }
  });
  $("file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });
}

const STATUS_TEXT = {
  queued: "Waiting to start", parsing: "Reading", embedding: "Understanding",
  keywords: "Finding concepts", building_map: "Building map", ready: "Ready", failed: "Failed",
};

/** Fill the "Recent documents" list. */
async function loadRecent() {
  const list = $("recent-list");
  let docs;
  try {
    docs = await API.listDocuments();
  } catch {
    return; // health chips already show the server is down
  }
  if (!docs.length) {
    list.innerHTML = `<p class="p-5 text-sm text-slate-500">No documents yet — upload one above.</p>`;
    return;
  }
  list.innerHTML = docs.map((d) => {
    const badgeClass = d.status === "ready" ? "ready" : d.status === "failed" ? "failed" : "working";
    const icon = { pdf: "📕", docx: "📘", pptx: "📙" }[d.filename.split(".").pop().toLowerCase()] || "📄";
    const date = new Date(d.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    return `
      <div class="flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors">
        <button class="flex-1 min-w-0 flex items-center gap-3 text-left" data-open="${d.doc_id}">
          <span class="text-xl">${icon}</span>
          <span class="min-w-0">
            <span class="block truncate font-medium">${escapeHtml(d.filename)}</span>
            <span class="block text-xs text-slate-500">${date}</span>
          </span>
        </button>
        <span class="badge ${badgeClass}">${STATUS_TEXT[d.status] || d.status}</span>
        <button class="text-slate-500 hover:text-rose-300 px-2" title="Delete" data-delete="${d.doc_id}">🗑</button>
      </div>`;
  }).join("");
}

function setupRecentList() {
  // One click handler for the whole list ("event delegation").
  $("recent-list").addEventListener("click", async (e) => {
    const open = e.target.closest("[data-open]");
    const del = e.target.closest("[data-delete]");
    if (open) openDocument(open.dataset.open);
    if (del && confirm("Delete this document and its mind map?")) {
      try {
        await API.remove(del.dataset.delete);
        toast("Document deleted");
        loadRecent();
      } catch (err) {
        toast(err.message, "error");
      }
    }
  });
}

// =====================================================================
// 6. Polling (processing screen)
// =====================================================================
// Which stepper step is active for each backend status.
const STEP_FOR_STATUS = { queued: 0, parsing: 0, embedding: 1, keywords: 2, building_map: 3, ready: 4 };

function showProcessing(doc, fileInfo) {
  showScreen("processing");
  $("proc-filename").textContent = doc.filename;
  $("proc-filesize").textContent = fileInfo ? formatBytes(fileInfo.size) : "";
  updateProcessing(doc);
  if (doc.status !== "failed") startPolling();
}

function updateProcessing(doc) {
  const failed = doc.status === "failed";
  const active = STEP_FOR_STATUS[doc.status] ?? 0;
  document.querySelectorAll(".step").forEach((el) => {
    const i = Number(el.dataset.step);
    el.classList.toggle("done", !failed && i < active);
    el.classList.toggle("active", !failed && i === active);
  });
  $("proc-bar").style.width = `${failed ? 0 : doc.progress}%`;
  $("proc-percent").textContent = failed ? "" : `${doc.progress}%`;
  $("proc-status").textContent = STATUS_TEXT[doc.status] || doc.status;
  $("proc-error").classList.toggle("hidden", !failed);
  $("proc-error-msg").textContent = doc.error || "";
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    try {
      const doc = await API.getDocument(state.docId);
      updateProcessing(doc);
      if (doc.status === "ready") {
        stopPolling();
        setTimeout(() => renderMap(doc), 500); // let the last check mark show briefly
      } else if (doc.status === "failed") {
        stopPolling();
      }
    } catch (err) {
      stopPolling();
      toast(err.message, "error");
    }
  }, 1500);
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

// =====================================================================
// 7. Graph rendering (Cytoscape)
// =====================================================================
const PALETTE = ["#818cf8", "#e879f9", "#22d3ee", "#34d399", "#fbbf24", "#fb7185", "#a78bfa"];

/** Mix a hex colour with white; amount 0..1 (used for the children's lighter tint). */
function tint(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const mix = (c) => Math.round(c + (255 - c) * amount);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(mix);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Turn the backend's flat node list into Cytoscape nodes + edges with colours and sizes. */
function buildElements(mindmap) {
  const colourOf = {};
  let branchIndex = 0;
  const elements = [];
  for (const node of mindmap.nodes) {
    if (node.level === 1) colourOf[node.id] = PALETTE[branchIndex++ % PALETTE.length];
    if (node.level >= 2) colourOf[node.id] = tint(colourOf[node.parent_id] || PALETTE[0], 0.55);
    const color = colourOf[node.id] || "#6366f1";
    // Width grows with label length, capped so long labels wrap onto more lines.
    const fontSize = [22, 15, 13][Math.min(node.level, 2)];
    const maxWidth = [220, 170, 150][Math.min(node.level, 2)];
    const width = Math.min(maxWidth, Math.max(70, node.label.length * fontSize * 0.58 + 28));
    const lines = Math.ceil((node.label.length * fontSize * 0.58) / (width - 24));
    elements.push({
      group: "nodes",
      data: {
        ...node,
        display: node.label,
        color,
        branchColor: colourOf[node.parent_id] || color,
        w: node.level === 0 ? Math.max(170, width) : width,
        h: node.level === 0 ? Math.max(170, width) : Math.max(38, lines * fontSize * 1.3 + 18),
        fontSize,
      },
      classes: `level${node.level}`,
    });
    if (node.parent_id) {
      elements.push({ group: "edges", data: { id: `e-${node.id}`, source: node.parent_id, target: node.id, color } });
    }
  }
  return elements;
}

const GRAPH_STYLE = [
  {
    selector: "node",
    style: {
      label: "data(display)",
      width: "data(w)",
      height: "data(h)",
      shape: "round-rectangle",
      "font-family": "Inter, system-ui, sans-serif",
      "font-size": "data(fontSize)",
      "font-weight": 600,
      "text-wrap": "wrap",
      "text-max-width": (n) => n.data("w") - 20,
      "text-valign": "center",
      "text-halign": "center",
      "background-color": "data(color)",
      color: "#0f172a",
      "border-width": 0,
      "underlay-color": "data(color)", // underlay = a soft glow behind the node
      "underlay-padding": 0,
      "underlay-opacity": 0,
      "underlay-shape": "round-rectangle",
      "transition-property": "opacity, underlay-opacity, underlay-padding, border-width",
      "transition-duration": 250,
    },
  },
  {
    selector: "node.level0",
    style: {
      shape: "ellipse",
      color: "#ffffff",
      "font-weight": 800,
      "background-fill": "linear-gradient",
      "background-gradient-stop-colors": "#6366f1 #d946ef #06b6d4",
      "background-gradient-direction": "to-bottom-right",
      "underlay-color": "#d946ef",
      "underlay-opacity": 0.35,
      "underlay-padding": 18,
      "underlay-shape": "ellipse",
      "text-outline-color": "#4c1d95",
      "text-outline-width": 1,
    },
  },
  { selector: "node.level1", style: { "font-weight": 700 } },
  { selector: "node.level2", style: { "font-weight": 500 } },
  {
    selector: "edge",
    style: {
      width: 2.5,
      "curve-style": "unbundled-bezier",
      "control-point-distances": 30,
      "line-color": "data(color)",
      "line-opacity": 0.45,
      "transition-property": "line-opacity, width",
      "transition-duration": 250,
    },
  },
  { selector: ".dim", style: { opacity: 0.18 } },
  { selector: "node.hl", style: { "underlay-opacity": 0.45, "underlay-padding": 10 } },
  { selector: "edge.hl", style: { "line-opacity": 1, width: 4 } },
  { selector: "node.explored", style: { "border-width": 4, "border-color": "#34d399", "border-opacity": 1 } },
  { selector: "node:selected", style: { "underlay-opacity": 0.6, "underlay-padding": 14 } },
];

/**
 * Our own radial layout: branches around the centre, each branch's
 * children fanned out in that branch's slice of the circle. Every branch
 * gets a slice proportional to how many children it has.
 */
function radialPositions(mindmap) {
  const branches = mindmap.nodes.filter((n) => n.level === 1);
  const childrenOf = (id) => mindmap.nodes.filter((n) => n.parent_id === id);
  const totalLeaves = branches.reduce((sum, b) => sum + Math.max(1, childrenOf(b.id).length), 0);
  // Outer ring radius grows with node count so children have ~165px of arc each.
  const r2 = Math.max(340, (totalLeaves * 165) / (2 * Math.PI));
  const r1 = Math.max(200, r2 * 0.5);
  const positions = { n0: { x: 0, y: 0 } };
  let angle = -Math.PI / 2; // start at the top (12 o'clock)
  for (const branch of branches) {
    const kids = childrenOf(branch.id);
    const slice = (2 * Math.PI * Math.max(1, kids.length)) / totalLeaves;
    const mid = angle + slice / 2;
    positions[branch.id] = { x: r1 * Math.cos(mid), y: r1 * Math.sin(mid) };
    kids.forEach((kid, i) => {
      const a = angle + (slice * (i + 0.5)) / kids.length;
      positions[kid.id] = { x: r2 * Math.cos(a), y: r2 * Math.sin(a) };
    });
    angle += slice;
  }
  return positions;
}

/** (Re-)arrange nodes in the radial layout, animated unless reduced motion is on. */
function runLayout(animate = true) {
  const positions = radialPositions(state.doc.mindmap);
  state.cy.layout({
    name: "preset",
    positions: (node) => positions[node.id()] || { x: 0, y: 0 },
    animate: animate && !prefersReducedMotion,
    animationDuration: 700,
    animationEasing: "ease-out-cubic",
    fit: true,
    padding: 60,
  }).run();
}

/** First render: nodes grow out of the centre and fade in. */
function introAnimation(cy) {
  if (prefersReducedMotion) return runLayout(false);
  cy.nodes().positions(() => ({ x: 0, y: 0 }));
  cy.elements().style("opacity", 0);
  runLayout(true);
  // When each fade finishes, remove the inline opacity so hover dimming (.dim) works again.
  cy.nodes().forEach((node) => {
    const delay = node.data("level") * 180;
    node.delay(delay).animate({ style: { opacity: 1 } }, { duration: 500, complete: () => node.removeStyle("opacity") });
  });
  cy.edges().delay(300).animate({ style: { opacity: 1 } }, { duration: 700, complete: () => cy.edges().removeStyle("opacity") });
}

function setupGraphEvents(cy) {
  // Hover: light up the node and its neighbours, dim everything else.
  cy.on("mouseover", "node", (e) => {
    const near = e.target.closedNeighborhood(); // the node + its edges + connected nodes
    cy.elements().not(near).addClass("dim");
    near.addClass("hl");
    $("cy").style.cursor = "pointer";
  });
  cy.on("mouseout", "node", () => {
    cy.elements().removeClass("dim hl");
    $("cy").style.cursor = "default";
  });
  cy.on("tap", "node", (e) => openDrawer(e.target));
  cy.on("tap", (e) => {
    if (e.target !== cy) return; // only react to taps on empty space
    cy.elements().removeClass("dim hl"); // touch screens never send "mouseout"
    closeDrawer();
  });
}

/** Show the mind-map screen for a ready document. */
function renderMap(doc) {
  state.doc = doc;
  state.docId = doc.doc_id;
  history.replaceState(null, "", `#doc=${doc.doc_id}`);
  showScreen("map");
  $("map-filename").textContent = doc.filename;
  if (state.cy) state.cy.destroy();
  state.cy = cytoscape({
    container: $("cy"),
    elements: buildElements(doc.mindmap),
    style: GRAPH_STYLE,
    minZoom: 0.15,
    maxZoom: 2.5,
    wheelSensitivity: 0.25,
  });
  setupGraphEvents(state.cy);
  restoreExplored();
  introAnimation(state.cy);
  if (doc.mindmap.generated_by === "fallback") {
    toast("The AI model was unavailable, so this map was built from headings and keywords.");
  }
}

function exportPng() {
  const png = state.cy.png({ full: true, scale: 2, bg: "#020617" });
  const link = document.createElement("a");
  link.href = png;
  link.download = `${state.doc.filename.replace(/\.[^.]+$/, "")}-mindmap.png`;
  link.click();
}

// =====================================================================
// 8. Drawer (concept details)
// =====================================================================
function openDrawer(node) {
  state.selectedNode = node;
  state.cy.nodes().unselect();
  node.select();
  const parent = state.cy.getElementById(node.data("parent_id") || "");
  $("drawer-title").textContent = node.data("label");
  $("drawer-parent").textContent = parent.nonempty() ? `${parent.data("label")} ›` : "Central topic";
  $("drawer-swatch").style.background = node.data("level") === 0 ? "linear-gradient(135deg,#6366f1,#d946ef)" : node.data("color");
  setSimpleToggle(false);
  $("drawer").classList.add("open");
  $("drawer-backdrop").classList.add("open");
  keepNodeVisible(node);
  loadExplanation("standard");
}

/** On wide screens, slide the graph left if the clicked node would hide behind the drawer. */
function keepNodeVisible(node) {
  if (window.innerWidth < 768) return;
  const drawerLeft = $("drawer").getBoundingClientRect().left || window.innerWidth - 470;
  const overlap = node.renderedPosition().x + 100 - drawerLeft;
  if (overlap > 0) state.cy.animate({ panBy: { x: -overlap, y: 0 } }, { duration: 350 });
}

function closeDrawer() {
  $("drawer").classList.remove("open");
  $("drawer-backdrop").classList.remove("open");
  state.selectedNode = null;
  state.explainRequest++; // ignore any answer still on its way
  if (state.cy) state.cy.nodes().unselect();
}

function setSimpleToggle(on) {
  $("simple-toggle").setAttribute("aria-checked", String(on));
}

/** Shimmer placeholders while the local AI works (can take several seconds). */
function showSkeleton() {
  const bar = (w, h = "h-3.5") => `<div class="skeleton ${h} ${w}"></div>`;
  const card = (rows) => `<div class="section-card space-y-2.5">${bar("w-32", "h-4")}${rows}</div>`;
  $("drawer-body").innerHTML = `
    <p class="text-xs text-slate-400 flex items-center gap-2"><span class="inline-block w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse"></span>Reading the document and thinking…</p>
    ${card(bar("w-full") + bar("w-11/12") + bar("w-4/5"))}
    ${card(bar("w-full") + bar("w-3/4"))}
    ${card(bar("w-5/6") + bar("w-4/6") + bar("w-3/4"))}
    ${card(bar("w-full", "h-9") + bar("w-full", "h-9"))}`;
}

async function loadExplanation(level) {
  const node = state.selectedNode;
  if (!node) return;
  const requestId = ++state.explainRequest;
  $("simple-toggle").disabled = true;
  showSkeleton();
  try {
    const data = await API.explain(state.docId, node.id(), level);
    if (requestId !== state.explainRequest) return; // user clicked something else meanwhile
    renderExplanation(data);
    markExplored(node);
  } catch (err) {
    if (requestId !== state.explainRequest) return;
    $("drawer-body").innerHTML = `
      <div class="section-card border-rose-400/30">
        <p class="font-semibold text-rose-200">Couldn't explain this concept</p>
        <p class="text-sm text-rose-100/80 mt-1">${escapeHtml(err.message)}</p>
        <button class="tool-btn tool-btn-primary mt-4" id="btn-explain-retry">Try again</button>
      </div>`;
    $("btn-explain-retry").onclick = () => loadExplanation(level);
    toast(err.message, "error");
  } finally {
    if (requestId === state.explainRequest) $("simple-toggle").disabled = false;
  }
}

function section(icon, title, inner, delay) {
  return `
    <div class="section-card" style="animation-delay:${delay}ms">
      <div class="section-title"><span class="icon">${icon}</span>${title}</div>
      ${inner}
    </div>`;
}

function renderExplanation(data) {
  const points = data.key_points.map((p) => `<li class="flex gap-2"><span class="text-indigo-300">▸</span><span>${escapeHtml(p)}</span></li>`).join("");
  const sources = data.sources.length
    ? data.sources.map((s) => {
        const loc = s.location.charAt(0).toUpperCase() + s.location.slice(1); // "slide 4" -> "Slide 4"
        const tip = [s.source, s.heading].filter(Boolean).join(" · ");
        return `<span class="source-chip" title="${escapeHtml(tip)}">📄 ${escapeHtml(loc)}${s.heading ? ` · ${escapeHtml(s.heading.slice(0, 24))}` : ""}</span>`;
      }).join("")
    : `<span class="text-sm text-slate-500">No matching passages found.</span>`;

  $("drawer-body").innerHTML = [
    data.level === "simple" ? `<p class="text-xs text-fuchsia-300">✨ Simpler version</p>` : "",
    section("💡", "Simple explanation", `<p class="text-[15px] leading-relaxed text-slate-200">${escapeHtml(data.simple_explanation)}</p>`, 0),
    section("🌍", "Real-life example", `<p class="text-[15px] leading-relaxed text-slate-200">${escapeHtml(data.real_life_example)}</p>`, 80),
    section("📌", "Key points", `<ul class="space-y-1.5 text-[15px] text-slate-200">${points}</ul>`, 160),
    section("🧠", "Quick check", renderQuiz(data.quiz), 240),
    section("📄", "Sources", `<div class="flex flex-wrap gap-2">${sources}</div>`, 320),
  ].join("");
  setupQuiz(data.quiz);
}

// =====================================================================
// 9. Quiz
// =====================================================================
function renderQuiz(quiz) {
  const letters = ["A", "B", "C", "D"];
  const options = quiz.options.map((opt, i) =>
    `<button class="quiz-option" data-index="${i}"><span class="font-semibold text-slate-400 mr-2">${letters[i]}</span>${escapeHtml(opt)}</button>`
  ).join("");
  return `
    <p class="font-medium mb-3">${escapeHtml(quiz.question)}</p>
    <div class="space-y-2" id="quiz-options">${options}</div>
    <p id="quiz-why" class="hidden mt-3 text-sm text-slate-300"></p>`;
}

function setupQuiz(quiz) {
  const buttons = [...document.querySelectorAll(".quiz-option")];
  buttons.forEach((btn) => btn.addEventListener("click", () => {
    const chosen = Number(btn.dataset.index);
    const correct = chosen === quiz.correct_index;
    btn.classList.add(correct ? "correct" : "wrong");
    buttons[quiz.correct_index].classList.add("correct"); // always reveal the right one
    buttons.forEach((b) => (b.disabled = true));
    const why = $("quiz-why");
    why.innerHTML = `${correct ? "🎉 <b>Correct!</b>" : "❌ <b>Not quite.</b>"} ${escapeHtml(quiz.why)}`;
    why.classList.remove("hidden");
  }));
}

// =====================================================================
// 10. Progress ("N of M explored", remembered per document)
// =====================================================================
const exploredKey = () => `nexus-explored-${state.docId}`;

function markExplored(node) {
  if (node.hasClass("explored")) return;
  const explored = new Set(storage.get(exploredKey(), []));
  explored.add(node.id());
  storage.set(exploredKey(), [...explored]);
  node.addClass("explored");
  node.data("display", `✓ ${node.data("label")}`);
  updateProgress();
  if (state.cy.nodes(".explored").length === state.cy.nodes().length) {
    toast("You explored every concept. Brilliant work! 🎓");
  }
}

function restoreExplored() {
  const explored = new Set(storage.get(exploredKey(), []));
  state.cy.nodes().forEach((node) => {
    if (explored.has(node.id())) {
      node.addClass("explored");
      node.data("display", `✓ ${node.data("label")}`);
    }
  });
  updateProgress();
}

function updateProgress() {
  const total = state.cy.nodes().length;
  const done = state.cy.nodes(".explored").length;
  $("explored-label").textContent = `${done} of ${total} explored`;
  $("explored-bar").style.width = `${total ? (done / total) * 100 : 0}%`;
}

// =====================================================================
// 11. Start-up
// =====================================================================
function setupButtons() {
  $("brand").onclick = () => showScreen("upload");
  $("btn-new").onclick = () => showScreen("upload");
  $("btn-fit").onclick = () => state.cy?.animate({ fit: { padding: 60 } }, { duration: 400 });
  $("btn-relayout").onclick = () => state.cy && runLayout(true);
  $("btn-export").onclick = () => state.cy && exportPng();
  $("drawer-close").onclick = closeDrawer;
  $("drawer-backdrop").onclick = closeDrawer;
  $("btn-other").onclick = () => showScreen("upload");
  $("btn-retry").onclick = async () => {
    try {
      const doc = await API.retry(state.docId);
      openDocument(doc.doc_id);
    } catch (err) {
      toast(err.message, "error");
    }
  };
  $("simple-toggle").onclick = () => {
    const on = $("simple-toggle").getAttribute("aria-checked") !== "true";
    setSimpleToggle(on);
    loadExplanation(on ? "simple" : "standard");
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  // Re-fit the graph after the window size changes (e.g. phone rotated).
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => state.cy?.resize().fit(undefined, 40), 200);
  });
}

function start() {
  if (typeof cytoscape === "undefined") {
    toast("Couldn't load the graph library (no internet for the CDN?).", "error");
  }
  setupDropzone();
  setupRecentList();
  setupButtons();
  refreshHealth();
  setInterval(refreshHealth, 15000);
  // If the URL has #doc=…, reopen that document (e.g. after a page refresh).
  const match = location.hash.match(/doc=([a-f0-9]+)/);
  if (match) openDocument(match[1]);
  else showScreen("upload");
}

start();
