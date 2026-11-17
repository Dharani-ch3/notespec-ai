// Knowledge AI Engine web console. Plain JS, no build step.
// All user and model text is inserted with textContent, never innerHTML.

const $ = (id) => document.getElementById(id);
const state = { project: null, run: null, auth: false, busy: false };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function storedKey() {
  try { return localStorage.getItem("kae-key") || ""; } catch { return ""; }
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const key = $("apiKey").value || storedKey();
  if (key) headers["X-API-Key"] = key;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { ...opts, headers });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = text; }
  if (!res.ok) throw new Error((data && data.detail) ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText);
  return data;
}

function toast(msg) {
  const t = el("div", { class: "toast", role: "status" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 3500);
}

async function busy(btn, fn) {
  if (state.busy) return;
  state.busy = true;
  const label = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "Working…"; }
  try { await fn(); } catch (e) { toast(e.message); } finally {
    state.busy = false;
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

// ---------------- projects ----------------

async function loadProjects(selectId) {
  let projects = await api("/api/projects");
  if (!projects.length) {
    await api("/api/projects", { method: "POST", json: { name: "Riverside Clinic Intake App" } });
    projects = await api("/api/projects");
  }
  const sel = $("projectSelect");
  sel.replaceChildren(...projects.map((p) => el("option", { value: p.id }, `${p.name}`)));
  sel.value = selectId || projects[0].id;
  await selectProject(sel.value);
}

async function selectProject(id) {
  state.project = id;
  state.run = null;
  await Promise.all([loadArtifacts(), loadLatestRun()]);
}

async function loadArtifacts() {
  const arts = await api(`/api/projects/${state.project}/artifacts`);
  $("artCount").textContent = arts.length ? `(${arts.length})` : "";
  $("artifactList").replaceChildren(
    ...arts.map((a) => el("li", { title: a.filename }, el("span", {}, a.filename), el("span", {}, `${a.chunks} ${a.chunks === 1 ? "passage" : "passages"}`)))
  );
  $("sampleBtn").hidden = arts.some((a) => a.filename.startsWith("01_kickoff"));
  return arts;
}

async function loadLatestRun() {
  const runs = await api(`/api/projects/${state.project}/runs`);
  if (runs.length) await showRun(await api(`/api/runs/${runs[0].id}`), false);
  else renderRun(null);
}

// ---------------- upload and search ----------------

async function upload(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await api(`/api/projects/${state.project}/artifacts`, { method: "POST", body: form });
  const errors = res.filter((r) => r.error);
  toast(errors.length ? errors.map((e) => `${e.filename}: ${e.error}`).join(" · ") : `Added ${res.length} file(s)`);
  await loadArtifacts();
}

const clip = (t, n) => (t.length > n ? t.slice(0, n) + "…" : t);

function highlight(text, query) {
  const words = query.toLowerCase().match(/[a-z0-9]{3,}/g) || [];
  if (!words.length) return [text];
  const re = new RegExp(`(${words.map((w) => w.replace(/[^a-z0-9]/g, "")).join("|")})`, "gi");
  return text.split(re).map((part, i) => (i % 2 ? el("mark", {}, part) : part));
}

async function search(q) {
  const hits = await api(`/api/projects/${state.project}/search?q=${encodeURIComponent(q)}&k=6`);
  $("searchResults").replaceChildren(
    ...(hits.length
      ? hits.map((h) =>
          el("li", { onclick: () => openSource(h.chunk_id) },
            el("div", { class: "meta" }, el("span", {}, h.filename),
              el("span", {}, `bm25 #${h.bm25_rank ?? "–"} · vec #${h.vector_rank ?? "–"}`)),
            el("div", { class: "bar", style: `width:${Math.round(h.score * 100)}%` }),
            el("div", {}, ...highlight(clip(h.text.replace(/^#+\s*/gm, "").replace(/\s+/g, " "), 220), q))
          ))
      : [el("li", {}, "No matches.")])
  );
}

async function openSource(chunkId) {
  const c = chunkId.startsWith("answers#") || chunkId.startsWith("feedback#")
    ? { filename: chunkId.startsWith("answers#") ? "your answer" : "reviewer feedback", text: sourceFromRun(chunkId) }
    : await api(`/api/chunks/${encodeURIComponent(chunkId)}`);
  $("srcTitle").textContent = `${c.filename} · ${chunkId}`;
  $("srcText").textContent = c.text;
  $("sourceDialog").showModal();
}

function sourceFromRun(chunkId) {
  const run = state.run;
  if (chunkId.startsWith("answers#")) {
    const qa = run.answers[chunkId.slice(8)];
    return qa ? `Q: ${qa.question}\nA: ${qa.answer}` : chunkId;
  }
  const fb = run.feedback[Number(chunkId.slice(9))];
  return fb ? `${fb.by}: ${fb.note}` : chunkId;
}

// ---------------- runs ----------------

async function showRun(run, scroll = true) {
  state.run = run;
  if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
  renderRun(run);
  const events = await api(`/api/runs/${run.id}/events`);
  $("logBox").hidden = false;
  $("log").replaceChildren(
    ...events.map((e) =>
      el("li", {}, `${e.at.replace("T", " ").slice(0, 19)}  `, el("b", {}, e.action), `  by ${e.actor}`,
        Object.keys(e.detail).length ? `  ${summarize(e.detail)}` : "")
    )
  );
}

function summarize(d) {
  const keep = Object.entries(d).filter(([k]) => !["chunk_ids", "questions"].includes(k));
  return keep.map(([k, v]) => `${k}=${Array.isArray(v) ? v.length : String(v).slice(0, 40)}`).join(" ");
}

function citeLabel(s) {
  if (s.chunk_id.startsWith("answers#")) return "your answer";
  if (s.chunk_id.startsWith("feedback#")) return "reviewer note";
  const name = (s.artifact || s.chunk_id).replace(/^\d+_/, "").replace(/\.(md|txt|pdf|docx|csv|json)$/, "");
  const part = Number(s.chunk_id.split("#")[1]);
  return Number.isFinite(part) ? `${name} ¶${part + 1}` : name;
}

function cites(sources) {
  if (!sources || !sources.length) return null;
  const uniq = [...new Map(sources.map((s) => [s.chunk_id, s])).values()];
  return el("div", { class: "cites" },
    ...uniq.map((s) => el("button", { class: "cite", type: "button", title: "Show the source passage", onclick: () => openSource(s.chunk_id) },
      citeLabel(s)))
  );
}

function section(title, items, render) {
  if (!items || !items.length) return null;
  return el("section", { class: "section" },
    el("h4", {}, title, el("span", { class: "count" }, items.length)),
    el("div", { class: "grid" }, ...items.map(render)));
}

function setStepper(run) {
  const order = ["ingest", "context", "spec", "questions", "review", "approved"];
  let now = "ingest";
  if (run) now = { needs_answers: "questions", in_review: "review", approved: "approved", failed: "spec" }[run.status];
  else if ($("artifactList").children.length) now = "context";
  const idx = order.indexOf(now);
  for (const li of $("stepper").children) {
    const i = order.indexOf(li.dataset.step);
    li.className = i < idx || (now === "approved" && i === idx) ? "done" : i === idx ? "now" : "";
  }
}

function renderRun(run) {
  setStepper(run);
  $("empty").hidden = !!run;
  $("questionsBox").hidden = !run || run.status !== "needs_answers";
  $("reviewBox").hidden = !run || run.status !== "in_review";
  $("approvedBox").hidden = !run || run.status !== "approved";
  $("logBox").hidden = !run;
  $("warnings").replaceChildren(...(run ? run.warnings.map((w) => el("p", { class: "warn" }, w)) : []));
  if (!run) {
    $("spec").replaceChildren();
    $("runMeta").textContent = "Add notes, then generate a spec.";
    return;
  }
  const spec = run.spec || {};
  $("runMeta").textContent = `Run ${run.id} · revision ${run.revision} · ${run.provider} · ${run.status.replace("_", " ")}`;

  if (run.status === "approved") {
    $("approvedBox").replaceChildren(
      el("h3", {}, "Approved"),
      el("p", {}, `Signed off by ${run.approved_by}. Every item below links to its source.`),
      el("p", {}, el("a", { href: `/api/runs/${run.id}/export.md`, target: "_blank", style: "color:#fff" }, "Download spec as Markdown →"))
    );
  }

  const qs = spec.open_questions || [];
  $("qCount").textContent = qs.length;
  $("answersForm").replaceChildren(
    ...qs.map((q) =>
      el("div", { class: "q" },
        el("label", { for: `ans-${q.id}` }, q.question),
        q.why_it_matters ? el("small", {}, q.why_it_matters) : null,
        cites(q.sources),
        el("input", { id: `ans-${q.id}`, name: q.id, placeholder: "Your answer (leave blank to skip for now)" }))
    ),
    el("button", { class: "btn", type: "submit", id: "answerBtn" }, "Submit answers and regenerate")
  );

  $("spec").replaceChildren(
    el("div", { class: "spec-head" }, el("h3", {}, spec.title || ""), el("p", {}, spec.summary || "")),
    section("Users", spec.users, (u) =>
      el("div", { class: "item" }, el("div", { class: "title" }, u.name),
        u.goals.length ? el("ul", {}, ...u.goals.map((g) => el("li", {}, g))) : null, cites(u.sources))),
    section("Features", spec.features, (f) =>
      el("div", { class: "item" },
        el("div", { class: "title" }, el("span", { class: "fid" }, f.id), f.name, el("span", { class: `prio ${f.priority}` }, f.priority)),
        el("p", {}, f.description), cites(f.sources))),
    section("Data entities", spec.entities, (e) =>
      el("div", { class: "item" }, el("div", { class: "title" }, e.name),
        e.fields.length ? el("div", { class: "tags" }, ...e.fields.map((f) => el("span", { class: "tag" }, f))) : null,
        cites(e.sources))),
    section("Screens", spec.screens, (s) =>
      el("div", { class: "item" }, el("div", { class: "title" }, s.name), el("p", {}, s.purpose), cites(s.sources))),
    section("Constraints", spec.constraints, (c) =>
      el("div", { class: "item" }, el("span", { class: "kind" }, c.kind), el("p", {}, c.text), cites(c.sources)))
  );
}

// ---------------- wire up ----------------

async function init() {
  try {
    const h = await api("/api/health");
    state.auth = h.auth;
    $("health").textContent = `${h.provider} · ${h.embedder}`;
    $("apiKey").hidden = !h.auth;
    $("apiKey").value = storedKey();
    const providers = await api("/api/providers").catch(() => []);
    $("providerSelect").replaceChildren(
      ...providers.map((p) => el("option", { value: p.name, selected: p.default || null, disabled: p.configured ? null : "" },
        `${p.name}${p.configured ? "" : " (not configured)"}`))
    );
    await loadProjects();
  } catch (e) {
    $("health").textContent = "offline";
    toast(e.message);
  }
}

$("apiKey").addEventListener("change", () => {
  try { localStorage.setItem("kae-key", $("apiKey").value); } catch {}
  init();
});
$("projectSelect").addEventListener("change", (e) => busy(null, () => selectProject(e.target.value)));
$("newProjectBtn").addEventListener("click", () => {
  const name = prompt("Project name");
  if (name) busy($("newProjectBtn"), async () => {
    const p = await api("/api/projects", { method: "POST", json: { name } });
    await loadProjects(p.id);
  });
});
$("sampleBtn").addEventListener("click", (e) => busy(e.target, async () => {
  await api(`/api/projects/${state.project}/sample`, { method: "POST" });
  await loadArtifacts();
  setStepper(state.run);
  toast("Loaded 4 sample notes from a community clinic project");
}));
$("fileInput").addEventListener("change", (e) => busy(null, () => upload([...e.target.files])));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => busy(null, () => upload([...e.dataTransfer.files])));
$("searchForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("searchInput").value.trim();
  if (q) busy(null, () => search(q));
});
$("generateBtn").addEventListener("click", (e) => busy(e.target, async () => {
  const run = await api(`/api/projects/${state.project}/runs`, { method: "POST", json: { provider: $("providerSelect").value || null } });
  await showRun(run);
}));
$("answersForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const answers = Object.fromEntries([...new FormData(e.target)].filter(([, v]) => v.trim()));
  if (!Object.keys(answers).length) return toast("Type at least one answer");
  busy($("answerBtn"), async () => showRun(await api(`/api/runs/${state.run.id}/answers`, { method: "POST", json: { answers } })));
});
$("approveBtn").addEventListener("click", (e) => busy(e.target, async () =>
  showRun(await api(`/api/runs/${state.run.id}/review`, { method: "POST", json: { decision: "approve", note: $("reviewNote").value } }))));
$("changesBtn").addEventListener("click", (e) => busy(e.target, async () => {
  const run = await api(`/api/runs/${state.run.id}/review`, { method: "POST", json: { decision: "request_changes", note: $("reviewNote").value } });
  $("reviewNote").value = "";
  await showRun(run);
}));

init();
