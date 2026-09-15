import {
  api,
  query,
  ready,
  upload,
  download,
  assetURL,
  loadImages,
} from "./api.js";
import {
  esc,
  icon,
  button,
  ib,
  toast,
  dialog,
  close,
  field,
  input,
  option,
  empty,
  time,
} from "./ui.js";
import { Canvas } from "./canvas.js";
import { createWorkspace } from "./workspace.js";
import { createLibrary } from "./library.js";
import { createGraphEditor } from "./graph.js";

const root = document.querySelector("#app");
const state = {
  page: "workspace",
  config: null,
  characters: [],
  workspaces: [],
  assets: [],
  more: false,
  kind: "history",
  offset: 0,
  jobs: [],
  provider: null,
  dirty: false,
  configDirty: false,
  canvasDirty: false,
  picked: [],
  cast: [],
  outfits: {},
  assetRoles: {},
  assetTargets: {},
  prompt: "",
  output: "3:4 4K",
  generating: false,
  planners: [],
  plans: [],
};
let canvas,
  saveTimer,
  pickCallback = null,
  characterDraft = null,
  lookIndex = 0;
Object.defineProperty(state, "dirty", {
  get: () => state.configDirty || state.canvasDirty,
  set: (value) => {
    if (state.page === "providers") state.configDirty = value;
    else state.canvasDirty = value;
  },
});
let canvasSaves = Promise.resolve();
let cropRect, cropAsset;
const shoot = createWorkspace({
  state,
  getCanvas: () => canvas,
  render: renderPage,
  shell,
  saveCanvas,
  picker,
  viewAsset,
  changed: onCanvasChange,
});
const library = createLibrary({
  state,
  render: renderPage,
  getAssets,
  viewAsset,
  refresh,
  saveCanvas,
});
const graph = createGraphEditor({
  state,
  render: renderPage,
  picker,
  viewAsset,
});
const pages = [
  ["workspace", "工作区", "canvas"],
  ["flows", "工作流", "flow"],
  ["gallery", "画廊", "gallery"],
  ["characters", "形象库", "people"],
  ["providers", "服务商", "settings"],
  ["jobs", "任务", "tasks"],
];
const chainNames = {
  draw: "文生图",
  edit: "改图",
  selfie: "自拍",
  video: "视频",
};
const providerName = (p) => p.label || p.id || "新服务商";
const providerOptions = (selected) =>
  `<option value="">按功能链路</option>${state.config.providers
    .filter((p) => !p.__template_key.includes("video"))
    .map((p) => option(p.id, providerName(p), selected))
    .join("")}`;
function shell() {
  root.innerHTML = `<aside class="sidebar"><a class="brand" href="#"><span class="brand-mark">${icon("canvas")}</span><strong>映像</strong><small>IMAGE STUDIO</small></a><nav>${pages.map(([id, label, symbol]) => button(label, "page", symbol, `data-page="${id}" class="${state.page === id ? "active" : ""}"`)).join("")}</nav><div class="sidebar-foot"><span class="connection-dot"></span>已连接 AstrBot</div></aside><main><header class="page-header"><div><span class="eyebrow">创作空间</span><h1>${pages.find((p) => p[0] === state.page)[1]}</h1></div><div class="header-actions">${state.dirty ? '<span class="save-status">有未保存修改</span>' : ""}${state.page === "providers" ? button("保存配置", "save-config", "check", 'class="primary"') : ""}${state.page === "characters" ? button("添加人物", "new-character", "plus", 'class="primary"') : ""}${state.page === "gallery" ? button("上传", "upload", "upload") : ""}${state.page === "workspace" ? `<select id="workspace-select" aria-label="选择工作区">${state.workspaces.map((w) => option(w.id, w.name, canvas?.doc.id)).join("")}</select>${ib("新建工作区", "new-workspace", "plus")}${ib("导出画布", "export", "download")}` : ""}${ib("刷新", "refresh-page", "refresh")}</div></header><div id="page-content"></div></main>`;
  document
    .querySelector("#workspace-select")
    ?.insertAdjacentHTML(
      "afterend",
      ib("重命名工作区", "rename-workspace", "edit"),
    );
  renderPage();
}
function renderPage() {
  const el = document.querySelector("#page-content");
  if (state.page === "workspace") renderWorkspace(el);
  if (state.page === "gallery") renderGallery(el);
  if (state.page === "characters") renderCharacters(el);
  if (state.page === "providers") renderProviders(el);
  if (state.page === "jobs") renderJobs(el);
  if (state.page === "flows") graph.render(el);
  loadImages();
}
async function refresh() {
  const data = await api("state");
  state.config = data.config;
  state.characters = data.characters;
  state.workspaces = data.workspaces;
  state.planners = data.planners || [];
  state.plans = data.plans || [];
  state.graphs = data.graphs || [];
  state.graphRuns = data.graph_runs || [];
  state.nodeTypes = data.node_types || {};
  state.storage = data.storage;
  graph.restore();
  if (!canvas) {
    let doc = state.workspaces[0];
    if (!doc) {
      doc = await api("workspace", {
        name: "我的工作区",
        layers: [],
        prompt: "",
      });
      state.workspaces.push(doc);
    }
    setCanvas(doc);
  } else if (!state.canvasDirty) {
    const updated = state.workspaces.find((w) => w.id === canvas.doc.id);
    if (updated) setCanvas(updated);
  }
  shell();
}
function setCanvas(doc) {
  canvas = new Canvas(doc, onCanvasChange);
  state.prompt = doc.prompt || "";
  shoot.restore(doc);
}
function onCanvasChange(doc) {
  state.canvasDirty = true;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(
    () => saveCanvas().catch((e) => toast(e.message)),
    700,
  );
}
function saveCanvas() {
  const current = canvas,
    draft = structuredClone(current.doc);
  draft.prompt = state.prompt;
  draft.shoot = structuredClone(shoot.capture());
  canvasSaves = canvasSaves
    .catch(() => {})
    .then(async () => {
      const status = document.querySelector("#workspace-status");
      if (status) status.textContent = "保存中";
      try {
        const saved = await api("workspace", {
          ...draft,
          revision: current.doc.revision,
        });
        current.doc.revision = saved.revision;
        const i = state.workspaces.findIndex((w) => w.id === saved.id);
        if (i >= 0) state.workspaces[i] = saved;
        if (
          current === canvas &&
          JSON.stringify(current.doc.layers) === JSON.stringify(draft.layers) &&
          JSON.stringify(shoot.capture()) === JSON.stringify(draft.shoot) &&
          state.prompt === draft.prompt
        )
          state.canvasDirty = false;
        if (status) status.textContent = "已保存";
      } catch (error) {
        if (status) status.textContent = "保存失败";
        throw error;
      }
    });
  return canvasSaves;
}
function renderWorkspace(el) {
  shoot.renderWorkspace(el);
}
function renderGallery(el) {
  library.render(el);
}
async function getAssets(reset = true) {
  if (reset) {
    state.offset = 0;
    state.assets = [];
  }
  const r = await query("library", { kind: state.kind, offset: state.offset });
  state.assets.push(...r.items);
  state.more = r.more;
  state.offset += r.items.length;
}
async function viewAsset(id) {
  const a = (await query("asset", { id, thumbnail: "1" })).asset;
  const cast = a.characters?.length
    ? "<h3>人物与穿搭</h3>" +
      a.characters
        .map(
          (c) =>
            `<p>${esc(c.name)} · ${esc(c.look)}<small>${esc(c.outfit)}</small></p>`,
        )
        .join("")
    : "";
  dialog(
    "图片",
    `<div class="asset-detail"><div class="large-image"><img id="large-image" alt="原图"></div><div><p class="muted">${esc(a.model || a.name || "图片")} · ${a.width} × ${a.height}</p><h3>生成提示词</h3><pre class="prompt-view">${esc(a.prompt || "旧缓存未保存提示词")}</pre><div class="metadata"><span>${esc(a.provider || "")}</span><span>${time(a.created)}</span></div>${cast}</div></div>`,
    `${button("下载原图", "download", "download", `data-id="${id}"`)}${button("做变体", "shoot-from", "copy", `data-mode="variants" data-id="${id}"`)}${button("仿拍", "shoot-from", "canvas", `data-mode="recreate" data-id="${id}"`)}${button("继续创作", "reuse", "canvas", `data-id="${id}" class="primary"`)}`,
  );
  document
    .querySelector(".sheet footer")
    ?.insertAdjacentHTML("afterbegin", library.assetActions(a));
  const img = document.querySelector("#large-image");
  try {
    img.src = await assetURL(id, "preview");
  } catch (e) {
    toast(e.message);
  }
}
async function picker(callback, title = "选择图片") {
  pickCallback = callback;
  const r = await query("library", { offset: 0 });
  dialog(
    title,
    `<div class="picker-toolbar">${button("上传图片", "upload", "upload")}</div><div class="asset-grid picker-grid">${r.items.map((a) => `<button class="asset-card" data-action="pick-asset" data-id="${a.id}"><div><img data-asset="${a.id}" alt="${esc(a.name || a.model || "图片")}"></div></button>`).join("")}</div>${r.more ? button("更多图片", "picker-more", "", 'data-offset="48"') : ""}${!r.items.length ? '<p class="muted">先上传一张清晰的参考图.</p>' : ""}`,
  );
  loadImages(document.querySelector("#overlay"));
}
function renderCharacters(el) {
  el.className = "standard-page";
  el.innerHTML = `<div class="characters-grid">${state.characters
    .map((c) => {
      const look = c.looks.find((l) => l.name === c.active_look) || c.looks[0];
      return `<article class="character-card"><button class="character-cover" data-action="edit-character" data-id="${c.id}"><img data-asset="${look.assets[0]}" alt="${esc(c.name)}"><span>${c.kind === "bot" ? "Bot" : "人物"}</span></button><div class="character-info"><h2>${esc(c.name)}</h2><select data-character-look="${c.id}" aria-label="${esc(c.name)}的默认形象">${c.looks.map((l) => option(l.name, l.name, c.active_look)).join("")}</select>${ib("编辑人物", "edit-character", "edit", `data-id="${c.id}"`)}</div></article>`;
    })
    .join(
      "",
    )}</div>${!state.characters.length ? empty("让每个人都有自己的形象.", "new-character", "添加人物") : ""}<div class="quiet-help"><code>/形象</code><span>查看</span><code>/换形象 形象名</code><span>本会话切换</span><code>/人物</code><span>查看可用人物</span></div>`;
}
function characterEditor(c) {
  characterDraft = structuredClone(
    c || {
      name: "",
      kind: "bot",
      bot_id: "",
      owner_sender: "",
      allowed_senders: [],
      scopes: [],
      looks: [{ name: "日常", assets: [] }],
      active_look: "日常",
    },
  );
  lookIndex = 0;
  renderCharacterEditor();
}
function renderCharacterEditor() {
  const c = characterDraft,
    l = c.looks[lookIndex];
  dialog(
    c.id ? "编辑人物" : "添加人物",
    `<form id="character-form"><div class="form-grid">${field("人物名称", input("name", c.name, "text", 'placeholder="例如 木有知" required'))}${field("身份", `<select name="kind">${option("bot", "Bot 自己", c.kind)}${option("person", "其他人物", c.kind)}</select>`)}${field("Bot 账号", input("bot_id", c.bot_id), "多个 Bot 时分别绑定, 单 Bot 可留空")}${field("本人用户 ID", input("owner_sender", c.owner_sender), "将聊天里的“我”对应到此人物")}${field("允许使用的用户 ID", input("allowed_senders", c.allowed_senders.join(",")), "多个用逗号分隔; 其他人物必填")}${field("限定会话", input("scopes", c.scopes.join(",")), "可选, 填 AstrBot 会话标识")}</div><div class="look-heading"><h3>形象</h3>${button("添加一套", "add-look", "plus")}</div><div class="look-tabs">${c.looks.map((look, i) => button(look.name, "edit-look", "", `data-index="${i}" class="${i === lookIndex ? "selected" : ""}"`)).join("")}</div>${field("形象名称", input("look_name", l.name))}<div class="look-images">${l.assets.map((id) => `<button type="button" data-action="remove-look-asset" data-id="${id}" title="移除参考图"><img data-asset="${id}" alt="身份参考">${icon("close")}</button>`).join("")}${button("从资产库选择", "look-assets", "plus")}</div><p class="field-help">只锁定此人的身份. 服装由每次拍摄分别决定.</p>${field("默认形象", `<select name="active_look">${c.looks.map((x) => option(x.name, x.name, c.active_look)).join("")}</select>`)}</form>`,
    (c.id ? ib("删除人物", "delete-character", "trash") : "") +
      button("保存人物", "save-character", "check", 'class="primary"'),
  );
  if (c.looks.length > 1)
    document
      .querySelector(".look-heading")
      .insertAdjacentHTML(
        "beforeend",
        ib("移除当前形象", "remove-look", "trash"),
      );
  loadImages(document.querySelector("#overlay"));
}
function readCharacterForm() {
  const form = document.querySelector("#character-form");
  if (!form) return;
  const d = new FormData(form);
  for (const k of ["name", "kind", "bot_id", "owner_sender", "active_look"])
    characterDraft[k] = String(d.get(k) || "");
  for (const k of ["allowed_senders", "scopes"])
    characterDraft[k] = String(d.get(k) || "")
      .split(/[,，\n]/)
      .map((x) => x.trim())
      .filter(Boolean);
  const prev = characterDraft.looks[lookIndex].name,
    next = String(d.get("look_name") || "");
  characterDraft.looks[lookIndex].name = next;
  if (characterDraft.active_look === prev) characterDraft.active_look = next;
}
function renderProviders(el) {
  el.className = "providers-page";
  el.innerHTML = `<div class="provider-list"><div class="list-heading"><span>${state.config.providers.length} 个服务商</span>${ib("添加服务商", "new-provider", "plus")}</div>${state.config.providers.map((p) => `<button class="provider-row ${state.provider === p.id ? "selected" : ""}" data-action="edit-provider" data-id="${esc(p.id)}"><span class="protocol-mark">${p.__template_key.startsWith("gemini") ? "G" : p.__template_key.includes("chat") ? "C" : "O"}</span><span><strong>${esc(providerName(p))}</strong><small>${esc(p.model || p.default_model || p.__template_key)}</small></span>${icon("arrow")}</button>`).join("")}<div class="chain-heading"><h3>回退链路</h3><small>从上到下尝试</small></div>${Object.entries(
    chainNames,
  )
    .map(
      ([id, label]) =>
        `<section class="chain"><header><strong>${label}</strong>${ib("添加" + label + "服务商", "chain-add", "plus", `data-feature="${id}"`)}</header><div class="chain-list" data-feature="${id}">${(state.config.features[id]?.chain || []).map((x, i) => `<div class="chain-row" data-index="${i}"><button class="drag-handle" aria-label="拖动排序" data-drag="${id}" data-index="${i}">${icon("grip")}</button><span class="chain-number">${i + 1}</span><span class="chain-label">${esc(providerName(state.config.providers.find((p) => p.id === x.provider_id) || { id: x.provider_id }))}${x.output ? `<small>${esc(x.output)}</small>` : ""}</span><div>${ib("上移", "chain-up", "up", `data-feature="${id}" data-index="${i}" ${i === 0 ? "disabled" : ""}`)}${ib("下移", "chain-down", "down", `data-feature="${id}" data-index="${i}"`)}${ib("移除", "chain-remove", "close", `data-feature="${id}" data-index="${i}"`)}</div></div>`).join("") || '<p class="chain-empty">尚未添加</p>'}</div></section>`,
    )
    .join("")}</div><div class="provider-editor" id="provider-editor"></div>`;
  renderProviderEditor();
  bindChains();
}
function renderProviderEditor() {
  const el = document.querySelector("#provider-editor"),
    p = state.config.providers.find((p) => p.id === state.provider);
  if (!p) {
    el.innerHTML = empty("连接你的图像模型.", "new-provider", "添加服务商");
    return;
  }
  const t = state.config.templates[p.__template_key];
  const mainKeys = new Set([
    "label",
    "base_url",
    "api_url",
    "server_url",
    "api_keys",
    "api_key",
    "apikey",
    "model",
    "default_model",
    "default_size",
    "default_resolution",
  ]);
  function renderField([k, m]) {
    let v = p[k] ?? m.default ?? "",
      control;
    const secret = /api.?key|token|password|cookie/i.test(k);
    if (k === "model" || k === "default_model") {
      control = `<div class="model-input">${input(k, v, "text", 'list="model-list" placeholder="获取后选择, 或手动填写"')}${ib("获取模型列表", "fetch-models", "refresh")}</div><datalist id="model-list"></datalist>`;
    } else if (m.type === "bool")
      control = `<select name="${k}">${option("true", "开启", String(v))}${option("false", "关闭", String(v))}</select>`;
    else if (m.type === "dict" || m.type === "object")
      control = `<textarea name="${k}" rows="2">${esc(JSON.stringify(v, null, 2))}</textarea>`;
    else if (m.type === "list") {
      control =
        typeof v === "string"
          ? input(k, v, "password", `placeholder="每行一个"`)
          : `<textarea name="${k}" rows="2" ${secret ? 'class="secret-input"' : ""}>${esc(v.join("\n"))}</textarea>`;
    } else if (m.options && !secret)
      control = `<input name="${k}" value="${esc(v)}" list="options-${k}"><datalist id="options-${k}">${m.options.map((x) => `<option value="${esc(x)}">`).join("")}</datalist>`;
    else
      control = input(
        k,
        v,
        secret
          ? "password"
          : ["int", "float"].includes(m.type)
            ? "number"
            : "text",
      );
    if (k === "id") control = input(k, v, "text", "readonly");
    return field(m.description || k, control);
  }
  const items = Object.entries(t?.items || {}).filter(
    ([k]) => k !== "extra_body",
  );
  el.innerHTML = `<div class="editor-title"><div><span class="eyebrow">${esc(t?.name || p.__template_key)}</span><h2>${esc(providerName(p))}</h2></div>${ib("复制服务商", "copy-provider", "copy")}${ib("移除服务商", "remove-provider", "trash")}</div><form id="provider-form"><div class="form-grid">${items
    .filter(([k]) => mainKeys.has(k))
    .map(renderField)
    .join(
      "",
    )}</div><div id="model-status" role="status"></div><details class="extra-options"><summary>连接与请求选项</summary><div class="form-grid advanced-grid">${items
    .filter(([k]) => !mainKeys.has(k))
    .map(renderField)
    .join(
      "",
    )}</div></details><details class="extra-options"><summary>额外参数 <span>${Object.keys(p.extra_body || {}).length}</span></summary><p class="field-help">按模型协议填写, 如 quality 或 generationConfig.</p><textarea name="extra_body" spellcheck="false" rows="7" aria-label="额外请求体 JSON">${esc(JSON.stringify(p.extra_body || {}, null, 2))}</textarea></details></form>`;
}
function readProvider() {
  const p = state.config.providers.find((p) => p.id === state.provider),
    form = document.querySelector("#provider-form");
  if (!p || !form) return;
  const t = state.config.templates[p.__template_key];
  const data = new FormData(form),
    next = { ...p };
  for (const [k, m] of Object.entries(t.items)) {
    if (!data.has(k)) continue;
    const v = String(data.get(k));
    if (v === "******** (已保存)") {
      next[k] = v;
      continue;
    }
    if (m.type === "bool") next[k] = v === "true";
    else if (m.type === "int" || m.type === "float") next[k] = Number(v);
    else if (m.type === "list")
      next[k] = v
        .split("\n")
        .map((x) => x.trim())
        .filter(Boolean);
    else if (m.type === "dict" || m.type === "object")
      next[k] = JSON.parse(v || "{}");
    else next[k] = v;
  }
  if (
    !next.extra_body ||
    Array.isArray(next.extra_body) ||
    typeof next.extra_body !== "object"
  )
    throw Error("额外请求体必须是 JSON 对象");
  if (next.id !== p.id) {
    for (const f of Object.keys(chainNames)) {
      for (const link of state.config.features[f]?.chain || [])
        if (link.provider_id === p.id) link.provider_id = next.id;
    }
    state.provider = next.id;
  }
  Object.assign(p, next);
}
function bindChains() {
  let drag;
  root.onpointerdown = (e) => {
    const h = e.target.closest("[data-drag]");
    if (!h) return;
    const row = h.closest(".chain-row");
    drag = {
      feature: h.dataset.drag,
      index: Number(h.dataset.index),
      id: e.pointerId,
      row,
    };
    h.setPointerCapture(e.pointerId);
    row.classList.add("dragging");
  };
  root.onpointerup = (e) => {
    if (!drag) return;
    const target = document
        .elementFromPoint(e.clientX, e.clientY)
        ?.closest(".chain-row"),
      list = target?.closest(".chain-list");
    if (list?.dataset.feature === drag.feature) {
      const dest = Number(target.dataset.index),
        rows = state.config.features[drag.feature].chain;
      const [item] = rows.splice(drag.index, 1);
      rows.splice(dest, 0, item);
      state.dirty = true;
    }
    drag.row.classList.remove("dragging");
    drag = null;
    readProvider();
    renderPage();
  };
  root.onpointercancel = () => {
    if (drag) drag.row.classList.remove("dragging");
    drag = null;
  };
}
const jobState = {
  queued: "排队中",
  running: "正在生成",
  completed: "已完成",
  partial: "部分完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  expired: "记录已过期",
};
function renderJobs(el) {
  el.className = "standard-page";
  el.innerHTML = `<div class="job-list">${state.jobs.map((j) => `<article class="job-row"><span class="job-state ${j.state}">${["running", "queued"].includes(j.state) ? '<span class="spinner"></span>' : icon(j.state === "completed" ? "check" : "tasks")}</span><div><strong>${esc(j.prompt.slice(0, 100))}</strong><small>${time(j.created)} · ${jobState[j.state] || j.state}</small>${j.error ? `<details class="job-error"><summary>错误详情</summary><p>${esc(j.error)}</p></details>` : ""}</div>${j.asset_id ? button("查看", "view-asset", "", `data-id="${j.asset_id}"`) : ""}${["queued", "running"].includes(j.state) ? button("取消", "cancel-job", "", `data-id="${j.task_id}"`) : ""}</article>`).join("")}</div>${!state.jobs.length ? empty("生成任务会留在这里.", "go-workspace", "开始创作") : ""}`;
}
async function pollJobs() {
  try {
    const jobs = await api("jobs");
    for (const job of jobs)
      for (const item of job.items || [])
        if (item.deleted) item.state = "deleted";
    const changed = JSON.stringify(jobs) !== JSON.stringify(state.jobs);
    state.jobs = jobs;
    if (state.page === "jobs" && changed) renderPage();
    if (state.page === "workspace") {
      const active = state.jobs.filter((j) =>
        ["running", "queued"].includes(j.state),
      );
      const target = document.querySelector("#active-job");
      if (target)
        target.innerHTML = active.length
          ? `<button data-action="go-jobs"><span class="spinner"></span>${active.length} 个任务进行中 ${icon("arrow")}</button>`
          : "";
    }
    for (const j of state.jobs) {
      const assets = j.items?.length
        ? j.items.filter((i) => i.asset_id).map((i) => i.asset_id)
        : j.asset_id
          ? [j.asset_id]
          : [];
      for (const id of assets) {
        if (
          j.workspace_id === canvas.doc.id &&
          !canvas.doc.layers.some((l) => l.asset_id === id)
        ) {
          const a = (await query("asset", { id, thumbnail: "1" })).asset;
          await canvas.add(a);
        }
      }
    }
    await shoot.poll();
    await graph.poll();
    pollJobs.initialized = true;
  } catch {
    /* Retry polling without submitting another generation. */
  }
}
pollJobs.seen = new Set();

async function act(action, el) {
  if (await library.action(action, el)) return;
  if (await graph.action(action, el)) return;
  if (action === "shoot-run") action = "generate";
  if (await shoot.action(action, el)) return;
  if (action === "page") {
    if (state.page === "providers") readProvider();
    if (state.page === "workspace") await saveCanvas();
    state.page = el.dataset.page;
    if (state.page === "gallery") await getAssets();
    if (state.page === "jobs") state.jobs = await api("jobs");
    shell();
  }
  if (action === "refresh-page") {
    if (state.dirty) {
      dialog(
        "有未保存的修改",
        "<p>重新载入会放弃本页未保存的修改.</p>",
        button("重新载入", "force-refresh", "refresh"),
      );
      return;
    }
    await refresh();
    if (state.page === "gallery") {
      await getAssets();
      renderPage();
    }
  }
  if (action === "force-refresh") {
    close();
    state.configDirty = false;
    state.canvasDirty = false;
    canvas = null;
    await refresh();
  }
  if (action === "close") {
    close();
    pickCallback = null;
  }
  if (action === "upload") {
    document.querySelector("#upload").click();
  }
  if (action === "gallery-kind") {
    state.kind = el.dataset.kind;
    await getAssets();
    renderPage();
  }
  if (action === "more-assets") {
    await getAssets(false);
    renderPage();
  }
  if (action === "view-asset") await viewAsset(el.dataset.id);
  if (action === "download") await download(el.dataset.id);
  if (action === "reuse") {
    const a = (await query("asset", { id: el.dataset.id, thumbnail: "1" }))
      .asset;
    close();
    state.page = "workspace";
    state.picked = [a.id];
    state.prompt = a.user_prompt || a.prompt || "";
    await canvas.add(a);
    await shoot.selectMode("edit", a.id);
    shell();
  }
  if (action === "pick-reference")
    await picker(async (a) => {
      if (state.picked.length >= 8) throw Error("最多选择 8 张参考图");
      if (!state.picked.includes(a.id)) state.picked.push(a.id);
      close();
      renderPage();
    });
  if (action === "unpick") {
    state.picked = state.picked.filter((id) => id !== el.dataset.id);
    renderPage();
  }
  if (action === "pick-asset") {
    const a = (await query("asset", { id: el.dataset.id, thumbnail: "1" }))
      .asset;
    if (pickCallback) {
      const cb = pickCallback;
      pickCallback = null;
      await cb(a);
    }
  }
  if (action === "picker-more") {
    const offset = Number(el.dataset.offset),
      r = await query("library", { offset });
    document
      .querySelector(".picker-grid")
      .insertAdjacentHTML(
        "beforeend",
        r.items
          .map(
            (a) =>
              `<button class="asset-card" data-action="pick-asset" data-id="${a.id}"><div><img data-asset="${a.id}" alt="图片"></div></button>`,
          )
          .join(""),
      );
    el.dataset.offset = offset + r.items.length;
    if (!r.more) el.remove();
    loadImages(document.querySelector("#overlay"));
  }
  if (action === "canvas-library")
    await picker(async (a) => {
      close();
      await canvas.add(a);
    });
  if (action === "use-selected") {
    const l = canvas.doc.layers.find((l) => l.id === canvas.selected);
    if (l && !state.picked.includes(l.asset_id)) state.picked.push(l.asset_id);
    renderPage();
  }
  if (action === "new-workspace") {
    await saveCanvas();
    const w = await api("workspace", {
      name: "工作区 " + (state.workspaces.length + 1),
      layers: [],
      prompt: "",
    });
    state.workspaces.push(w);
    setCanvas(w);
    shell();
  }
  if (action === "rename-workspace") {
    dialog(
      "工作区名称",
      field(
        "名称",
        input("workspace_name", canvas.doc.name, "text", 'maxlength="80"'),
      ),
      button("保存", "save-workspace-name", "check", 'class="primary"'),
    );
  }
  if (action === "save-workspace-name") {
    const name = document.querySelector('[name="workspace_name"]').value.trim();
    if (!name) throw Error("请输入工作区名称");
    canvas.doc.name = name;
    state.canvasDirty = true;
    await saveCanvas();
    close();
    shell();
  }
  if (action === "hand") {
    canvas.tool = canvas.tool === "hand" ? "select" : "hand";
    el.classList.toggle("selected", canvas.tool === "hand");
  }
  if (action === "undo" || action === "redo") canvas.undo(action === "redo");
  if (action.startsWith("layer-")) canvas.mutate(action.slice(6));
  if (action === "fit") canvas.fit();
  if (action === "crop") {
    const layer = canvas.doc.layers.find((l) => l.id === canvas.selected);
    if (!layer) throw Error("先选择需要裁剪的图片");
    cropAsset = layer.asset_id;
    cropRect = null;
    dialog(
      "裁剪",
      '<div class="crop-surface"><img id="crop-image" alt="裁剪原图"><div id="crop-rect"></div></div>',
      button("裁剪为新图", "confirm-crop", "crop", 'class="primary"'),
    );
    const image = document.querySelector("#crop-image");
    image.src = await assetURL(cropAsset, false);
    await image.decode();
    const surface = image.parentElement,
      selection = document.querySelector("#crop-rect");
    let start;
    const point = (e) => {
      const r = image.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
        y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
      };
    };
    surface.onpointerdown = (e) => {
      surface.setPointerCapture(e.pointerId);
      start = point(e);
    };
    surface.onpointermove = (e) => {
      if (!start) return;
      const p = point(e);
      cropRect = [
        Math.min(start.x, p.x),
        Math.min(start.y, p.y),
        Math.abs(p.x - start.x),
        Math.abs(p.y - start.y),
      ];
      Object.assign(selection.style, {
        display: "block",
        left: cropRect[0] * 100 + "%",
        top: cropRect[1] * 100 + "%",
        width: cropRect[2] * 100 + "%",
        height: cropRect[3] * 100 + "%",
      });
    };
    surface.onpointerup = () => {
      start = null;
    };
  }
  if (action === "confirm-crop") {
    if (!cropRect || cropRect[2] < 0.01 || cropRect[3] < 0.01)
      throw Error("请拖动选择裁剪区域");
    const a = await api("crop", { id: cropAsset, rect: cropRect });
    close();
    await canvas.add(a);
    toast("裁剪已保存, 原图保留");
  }
  if (action === "export") {
    const blob = await canvas.exportBlob(),
      url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = canvas.doc.name + ".png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  if (action === "pick-cast") {
    dialog(
      "出镜人物",
      `<div class="cast-list">${state.characters.map((c) => `<label><img data-asset="${c.looks[0].assets[0]}" alt="${esc(c.name)}"><span>${esc(c.name)}</span><input type="checkbox" name="cast" value="${c.id}" ${state.cast.includes(c.id) ? "checked" : ""}></label>`).join("") || "<p>先在形象库添加人物.</p>"}</div>`,
      button("使用这些人物", "confirm-cast", "check", 'class="primary"'),
    );
    loadImages(document.querySelector("#overlay"));
  }
  if (action === "confirm-cast") {
    state.cast = [...document.querySelectorAll("input[name=cast]:checked")].map(
      (x) => x.value,
    );
    if (state.cast.length > 4) throw Error("最多选择 4 位人物");
    close();
    onCanvasChange(canvas.doc);
    renderPage();
  }
  if (action === "generate") {
    if (state.generating) return;
    if (["generate", "edit"].includes(shoot.mode) && !state.prompt.trim())
      throw Error("请先填写创作指令");
    state.generating = true;
    el.disabled = true;
    try {
      await saveCanvas();
      const payload = {
        ...shoot.payload(),
        prompt: state.prompt,
        assets: state.picked,
        asset_roles: state.picked.map(
          (id) => state.assetRoles[id] || "background",
        ),
        asset_targets: state.picked.map((id) => state.assetTargets[id] || ""),
        characters: state.cast,
        outfits: state.cast.map((id) => state.outfits[id] || ""),
        provider: state.generationProvider || "",
        output: state.output,
        workspace_id: canvas.doc.id,
      };
      const result = await shoot.request("generate", payload);
      shoot.submitted(result.task_id);
      toast("已加入任务, 可以继续创作");
      await pollJobs();
    } finally {
      state.generating = false;
      el.disabled = false;
      if (state.page === "workspace") renderPage();
    }
  }
  if (action === "go-jobs" || action === "go-workspace") {
    state.page = action === "go-jobs" ? "jobs" : "workspace";
    shell();
  }
  if (action === "cancel-job") {
    await api("cancel", { task_id: el.dataset.id });
    await pollJobs();
    toast("已停止等待, 上游已开始的请求可能仍计费");
  }
  if (action === "new-character") characterEditor();
  if (action === "edit-character")
    characterEditor(state.characters.find((c) => c.id === el.dataset.id));
  if (action === "add-look") {
    readCharacterForm();
    characterDraft.looks.push({
      name: "形象 " + (characterDraft.looks.length + 1),
      assets: [],
    });
    lookIndex = characterDraft.looks.length - 1;
    renderCharacterEditor();
  }
  if (action === "edit-look") {
    readCharacterForm();
    lookIndex = Number(el.dataset.index);
    renderCharacterEditor();
  }
  if (action === "remove-look") {
    readCharacterForm();
    if (characterDraft.looks.length <= 1) throw Error("至少保留一套形象");
    const removed = characterDraft.looks.splice(lookIndex, 1)[0];
    if (characterDraft.active_look === removed.name)
      characterDraft.active_look = characterDraft.looks[0].name;
    lookIndex = Math.max(0, lookIndex - 1);
    renderCharacterEditor();
  }
  if (action === "delete-character") {
    dialog(
      "删除人物",
      `<p>删除 ${esc(characterDraft.name)}? 参考图与生成历史会保留.</p>`,
      button("返回", "back-character") +
        button("确认删除", "confirm-delete-character", "trash"),
    );
  }
  if (action === "back-character") renderCharacterEditor();
  if (action === "confirm-delete-character") {
    const id = characterDraft.id;
    await api("character", {
      id,
      revision: characterDraft.revision,
      delete: true,
    });
    state.characters = state.characters.filter((c) => c.id !== id);
    state.cast = state.cast.filter((cid) => cid !== id);
    close();
    shell();
    toast("人物已删除, 图片已保留");
  }
  if (action === "look-assets") {
    readCharacterForm();
    await picker((a) => {
      const ids = characterDraft.looks[lookIndex].assets;
      if (ids.length >= 4) throw Error("每套最多 4 张身份图");
      if (!ids.includes(a.id)) ids.push(a.id);
      renderCharacterEditor();
    }, "选择身份参考");
  }
  if (action === "remove-look-asset") {
    readCharacterForm();
    characterDraft.looks[lookIndex].assets = characterDraft.looks[
      lookIndex
    ].assets.filter((id) => id !== el.dataset.id);
    renderCharacterEditor();
  }
  if (action === "save-character") {
    readCharacterForm();
    const c = await api("character", characterDraft);
    const i = state.characters.findIndex((x) => x.id === c.id);
    if (i < 0) state.characters.push(c);
    else state.characters[i] = c;
    close();
    shell();
    toast("人物已保存");
  }
  if (action === "edit-provider") {
    readProvider();
    state.provider = el.dataset.id;
    renderPage();
    document
      .querySelector("#provider-editor")
      .scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (action === "new-provider") {
    readProvider();
    dialog(
      "添加服务商",
      `<div class="template-options">${Object.entries(state.config.templates)
        .map(
          ([id, t], i) =>
            `<button data-action="add-provider" data-template="${id}"><span class="protocol-mark">${i < 3 ? ["O", "G", "C"][i] : icon("plus")}</span><div><strong>${esc(t.name)}</strong><small>${esc((t.description || "").slice(0, 70))}</small></div>${icon("arrow")}</button>`,
        )
        .join("")}</div>`,
    );
  }
  if (action === "add-provider") {
    const key = el.dataset.template,
      t = state.config.templates[key],
      p = Object.fromEntries(
        Object.entries(t.items).map(([k, m]) => [
          k,
          structuredClone(m.default ?? ""),
        ]),
      );
    p.__template_key = key;
    p.id = key + "_" + crypto.randomUUID().slice(0, 6);
    state.config.providers.push(p);
    state.provider = p.id;
    state.dirty = true;
    close();
    shell();
  }
  if (action === "copy-provider") {
    readProvider();
    const p = structuredClone(
      state.config.providers.find((p) => p.id === state.provider),
    );
    p.id += "_" + crypto.randomUUID().slice(0, 6);
    p.label = (p.label || p.id) + " 副本";
    for (const k of Object.keys(p))
      if (/api.?key|token|cookie/i.test(k))
        p[k] = Array.isArray(
          state.config.templates[p.__template_key].items[k]?.default,
        )
          ? []
          : "";
    state.config.providers.push(p);
    state.provider = p.id;
    state.dirty = true;
    shell();
  }
  if (action === "remove-provider") {
    const id = state.provider;
    if (
      Object.keys(chainNames).some((k) =>
        state.config.features[k]?.chain?.some((x) => x.provider_id === id),
      )
    )
      throw Error("请先从回退链路移除此服务商");
    state.config.providers = state.config.providers.filter((p) => p.id !== id);
    state.provider = null;
    state.dirty = true;
    shell();
  }
  if (action === "fetch-models") {
    readProvider();
    el.disabled = true;
    const selected = state.provider;
    try {
      const r = await api("models", {
        provider: state.config.providers.find((p) => p.id === selected),
      });
      if (state.provider !== selected) return;
      document.querySelector("#model-list").innerHTML = r.models
        .map((id) => `<option value="${esc(id)}">`)
        .join("");
      const modelInput = document.querySelector("input[list=model-list]");
      modelInput.focus();
      document.querySelector("#model-status").textContent =
        "已获取 " + r.models.length + " 个模型, 点击模型输入框选择";
      toast("模型列表已更新");
    } finally {
      el.disabled = false;
    }
  }
  if (action === "chain-add") {
    readProvider();
    const f = el.dataset.feature;
    dialog(
      "添加到" + chainNames[f],
      `<div class="template-options">${state.config.providers
        .filter((p) =>
          f === "video"
            ? p.__template_key.includes("video")
            : !p.__template_key.includes("video"),
        )
        .map((p) =>
          button(
            providerName(p),
            "chain-select",
            "plus",
            `data-feature="${f}" data-id="${esc(p.id)}"`,
          ),
        )
        .join("")}</div>`,
    );
  }
  if (action === "chain-select") {
    const f = el.dataset.feature;
    state.config.features[f] ??= {};
    state.config.features[f].chain ??= [];
    state.config.features[f].chain.push({
      __template_key: "provider",
      provider_id: el.dataset.id,
    });
    state.dirty = true;
    close();
    shell();
  }
  if (["chain-up", "chain-down", "chain-remove"].includes(action)) {
    readProvider();
    const rows = state.config.features[el.dataset.feature].chain,
      i = Number(el.dataset.index),
      n = action === "chain-up" ? i - 1 : i + 1;
    if (action === "chain-remove") rows.splice(i, 1);
    else if (n >= 0 && n < rows.length) [rows[i], rows[n]] = [rows[n], rows[i]];
    state.dirty = true;
    shell();
  }
  if (action === "save-config") {
    readProvider();
    el.disabled = true;
    try {
      state.config = await api("config", {
        revision: state.config.revision,
        providers: state.config.providers,
        chains: Object.fromEntries(
          Object.keys(chainNames).map((k) => [
            k,
            state.config.features[k]?.chain || [],
          ]),
        ),
      });
      state.dirty = false;
      shell();
      toast("配置已保存并生效");
    } finally {
      el.disabled = false;
    }
  }
}
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (el && !el.disabled)
    act(el.dataset.action, el).catch((error) => toast(error.message));
});
document.addEventListener("input", (e) => {
  shoot.onInput(e);
  if (e.target.id === "prompt") {
    state.prompt = e.target.value;
    canvas.doc.prompt = state.prompt;
    onCanvasChange(canvas.doc);
  }
  if (e.target.dataset.outfit)
    state.outfits[e.target.dataset.outfit] = e.target.value;
  if (e.target.closest("#provider-form")) state.dirty = true;
});
document.addEventListener("change", async (e) => {
  try {
    if (state.page === "workspace") shoot.change(e);
    if (e.target.dataset.role)
      state.assetRoles[e.target.dataset.role] = e.target.value;
    if (e.target.dataset.target)
      state.assetTargets[e.target.dataset.target] = e.target.value;
    if (e.target.id === "output") state.output = e.target.value;
    if (e.target.id === "generation-provider")
      state.generationProvider = e.target.value;
    if (e.target.id === "workspace-select") {
      if (shoot.submitting) {
        e.target.value = canvas.doc.id;
        throw Error("请求正在提交, 请稍候切换工作区");
      }
      await saveCanvas();
      setCanvas(state.workspaces.find((w) => w.id === e.target.value));
      shell();
      canvas.fit();
    }
    if (e.target.dataset.characterLook) {
      const c = await api("appearance", {
        character_id: e.target.dataset.characterLook,
        look: e.target.value,
      });
      state.characters[state.characters.findIndex((x) => x.id === c.id)] = c;
      renderPage();
    }
  } catch (error) {
    toast(error.message);
  }
});
document.querySelector("#upload").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  e.target.value = "";
  try {
    for (const file of files) {
      toast("正在上传 " + file.name);
      const a = await upload(file);
      if (pickCallback) {
        const cb = pickCallback;
        pickCallback = null;
        await cb(a);
      } else if (state.page === "workspace") await canvas.add(a);
    }
    if (state.page === "gallery") {
      await getAssets();
      renderPage();
    }
    toast("图片已保存");
  } catch (error) {
    toast(error.message);
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") close();
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  if (
    state.page === "workspace" &&
    (e.metaKey || e.ctrlKey) &&
    e.key.toLowerCase() === "z"
  ) {
    e.preventDefault();
    canvas.undo(e.shiftKey);
  }
  if (state.page === "workspace" && e.key === "Delete") canvas.mutate("remove");
});
window.addEventListener("beforeunload", (e) => {
  if (state.dirty || graph.dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
try {
  await ready();
  await refresh();
  await pollJobs();
  setInterval(pollJobs, 4000);
} catch (error) {
  root.innerHTML = `<div class="boot error"><h2>工作台暂时无法打开</h2><p>${esc(error.message)}</p>${button("重试", "refresh-page", "refresh")}</div>`;
}
