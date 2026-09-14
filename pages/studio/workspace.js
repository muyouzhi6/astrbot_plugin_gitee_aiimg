import { api, query, loadImages } from "./api.js";
import {
  esc,
  icon,
  button,
  ib,
  field,
  input,
  option,
  dialog,
  close,
  toast,
  time,
} from "./ui.js";

const modes = [
  ["generate", "自由生成", "从想法开始", "plus"],
  ["edit", "图片编辑", "用一句话修改", "edit"],
  ["variants", "成片变体", "保留风格, 探索镜头", "copy"],
  ["recreate", "严格仿拍", "复现构图与氛围", "canvas"],
  ["outfit", "换装拍摄", "只换衣服, 保留人物", "people"],
];
const statuses = {
  queued: "排队中",
  running: "生成中",
  planning: "正在规划",
  completed: "已完成",
  partial: "部分完成",
  failed: "失败",
  interrupted: "已中断",
  cancelled: "已取消",
  expired: "已过期",
};
const special = (mode) => ["variants", "recreate", "outfit"].includes(mode);
const sourceNames = {
  variants: "满意的成片",
  recreate: "仿拍目标",
  outfit: "服装参考",
};
const placeholders = {
  generate: "描述你想拍的画面...",
  edit: "想改哪里? 例如把背景换成雨后街道...",
  variants: "想变化什么? 例如保留穿搭, 多一些近景和自然抓拍...",
  recreate: "哪些细节一定要保留? 可留空...",
  outfit: "穿上它去哪里拍? 例如傍晚公园, 轻松自然...",
};

export function createWorkspace({
  state,
  getCanvas,
  render,
  shell,
  saveCanvas,
  picker,
  viewAsset,
  changed,
}) {
  let pending = {},
    submitting = false;
  async function sendOnce(endpoint, payload) {
    if (submitting) throw Error("上一条请求正在提交, 请稍候");
    submitting = true;
    try {
      const fingerprint = JSON.stringify(payload);
      let entry = pending[endpoint];
      if (!entry || entry.fingerprint !== fingerprint)
        entry = { fingerprint, id: crypto.randomUUID() };
      pending[endpoint] = entry;
      persist();
      await saveCanvas();
      const result = await api(endpoint, { ...payload, request_id: entry.id });
      delete pending[endpoint];
      try {
        await saveCanvas();
      } catch (error) {
        pending[endpoint] = entry;
        throw error;
      }
      return result;
    } finally {
      submitting = false;
    }
  }
  let view = "board",
    mode = "generate",
    count = 1,
    source = "",
    planner = "",
    target = "",
    plan = null;
  let planning = false,
    generationGroup = "",
    focused = "",
    lastBoard = "";
  const ids = () => [...new Set(getCanvas().doc.layers.map((l) => l.asset_id))];
  const currentJobs = () =>
    state.jobs.filter((j) => j.workspace_id === getCanvas().doc.id);
  const modeName = () => modes.find((m) => m[0] === mode)?.[1] || "自由生成";
  const sourceView = () =>
    source
      ? `<div class="source-preview"><img data-asset="${esc(source)}" alt="${sourceNames[mode]}"><span>${sourceNames[mode]}</span>${ib("更换参考图", "shoot-source", "refresh")}</div>`
      : `<button class="source-upload" data-action="shoot-source">${icon("upload")}<strong>选择${sourceNames[mode]}</strong><span>从画廊选择或上传</span></button>`;
  const planMatches = () =>
    plan?.state === "ready" &&
    plan.workflow === mode &&
    plan.source_asset === source &&
    plan.prompt === state.prompt.trim() &&
    plan.output === state.output &&
    JSON.stringify(plan.characters) === JSON.stringify(state.cast) &&
    (plan.target_character || "") === target &&
    plan.count === count;

  function restore(doc) {
    const s = doc.shoot || {};
    pending = s.pending || {};
    mode = modes.some((m) => m[0] === s.mode) ? s.mode : "generate";
    count = s.count || 1;
    source = s.source || "";
    planner = s.planner || "";
    target = s.target || "";
    view = s.view || "board";
    focused = s.focused || "";
    generationGroup = s.group || "";
    plan = state.plans?.find((p) => p.id === s.plan_id) || null;
    if (plan?.state === "ready" && s.plan_shots) plan.shots = s.plan_shots;
    state.picked = s.picked || [];
    state.cast = s.cast || [];
    state.outfits = s.outfits || {};
    state.output = s.output || "3:4 4K";
    state.generationProvider = s.provider || "";
    state.assetRoles = s.assetRoles || {};
    state.assetTargets = s.assetTargets || {};
  }
  function capture() {
    return {
      pending,
      mode,
      count,
      source,
      planner,
      target,
      view,
      focused,
      group: generationGroup,
      plan_id: plan?.id || "",
      plan_shots: plan?.shots,
      picked: state.picked,
      cast: state.cast,
      outfits: state.outfits,
      output: state.output,
      provider: state.generationProvider,
      assetRoles: state.assetRoles,
      assetTargets: state.assetTargets,
    };
  }
  function persist() {
    changed(getCanvas().doc);
  }
  function modeTabs() {
    return `<div class="mode-tabs" role="tablist" aria-label="创作方式">${modes.map(([id, label, subtitle, symbol]) => `<button role="tab" aria-selected="${mode === id}" data-action="shoot-mode" data-mode="${id}" class="${mode === id ? "active" : ""}">${icon(symbol)}<span>${label}</span></button>`).join("")}</div>`;
  }
  function composer() {
    const busy = planning || plan?.state === "planning";
    const people = state.cast
      .map((id) => state.characters.find((c) => c.id === id))
      .filter(Boolean);
    const needsPeople = ["recreate", "outfit"].includes(mode);
    const list = `<div class="cast-pills">${people.map((c) => `<button data-action="pick-cast"><img data-asset="${esc(c.looks.find((l) => l.name === c.active_look)?.assets[0] || c.looks[0].assets[0])}" alt="${esc(c.name)}">${esc(c.name)}</button>`).join("")}${button(people.length ? "调整人物" : needsPeople ? "选择出镜人物" : "添加人物", "pick-cast", "people")}</div>`;
    const providers = state.config.providers.filter(
      (p) => !p.__template_key.includes("video"),
    );
    return `<aside class="shoot-composer"><div class="composer-heading"><div><span class="eyebrow">创作</span><h2>${modeName()}</h2></div><span class="count-label">${count} 张</span></div>
      ${special(mode) ? sourceView() : `<div class="input-assets">${state.picked.map((id) => `<button data-action="unpick" data-id="${esc(id)}" title="移除参考图"><img data-asset="${esc(id)}" alt="参考图">${icon("close")}</button>`).join("")}${button(mode === "edit" ? "选择要编辑的图片" : "添加参考图", "pick-reference", "plus")}</div>`}
      ${list}${state.cast.length > 1 && special(mode) ? field("参考图对应人物", `<select id="shoot-target">${option("", "请选择", target)}${people.map((c) => option(c.id, c.name, target)).join("")}</select>`) : ""}
      <label class="brief-field"><span>${special(mode) ? "补充要求" : "画面描述"}</span><textarea id="prompt" rows="4" placeholder="${placeholders[mode]}">${esc(state.prompt)}</textarea></label>
      ${people.length ? `<details class="wardrobe-options"><summary>独立穿搭 <span>${people.length} 位人物</span></summary>${people.map((c) => field(c.name, input("outfit-" + c.id, state.outfits[c.id] || "", "text", `data-outfit="${c.id}" placeholder="${c.kind === "bot" ? "默认使用日程穿搭" : "按语义单独设计"}"`))).join("")}</details>` : ""}
      ${!special(mode) && state.picked.length && people.length ? `<details class="wardrobe-options"><summary>参考图用途</summary>${state.picked.map((id, i) => `<div class="reference-roles"><span>参考 ${i + 1}</span><select data-role="${id}" aria-label="参考 ${i + 1} 用途">${["background", "style", "clothing", "pose", "subject"].map((r, j) => option(r, ["背景", "画风", "服装", "姿态", "构图"][j], state.assetRoles[id] || "background")).join("")}</select><select data-target="${id}" aria-label="参考 ${i + 1} 所属人物">${option("", "所属人物", "")}${people.map((c) => option(c.id, c.name, state.assetTargets[id])).join("")}</select></div>`).join("")}</details>` : ""}
      <div class="shoot-options">${field("生成模型", `<select id="generation-provider" aria-label="生成服务商">${option("", "按回退链路", state.generationProvider)}${providers.map((p) => option(p.id, p.label || p.id, state.generationProvider)).join("")}</select>`)}
      ${special(mode) ? field("镜头规划模型", `<select id="shoot-planner">${option("", "选择视觉对话模型", planner)}${(state.planners || []).map((p) => option(p.id, p.model + " · " + p.id, planner)).join("")}</select>`) : ""}
      <div class="format-count">${field("画幅", `<select id="output" aria-label="图片规格">${["3:4 4K", "1:1 4K", "16:9 4K", "9:16 4K", "4:3 4K", "3:4 2K", "1:1 2K", "16:9 2K", "1024x1024"].map((v) => option(v, v, state.output)).join("")}</select>`)}${field("数量", `<div class="stepper">${ib("减少数量", "shoot-minus", "back")}<input id="shoot-count" aria-label="生成数量" type="number" min="1" max="12" value="${count}">${ib("增加数量", "shoot-plus", "plus")}</div>`)}</div></div>
      <div class="shoot-submit">${button(busy ? "正在规划镜头" : special(mode) ? (planMatches() ? "重新规划镜头" : "规划镜头") : state.generating ? "正在提交" : `生成 ${count} 张`, special(mode) ? "shoot-plan" : "generate", busy ? "" : "arrow", `class="primary" ${busy || state.generating ? "disabled" : ""}`)}<span>${special(mode) ? "先看镜头清单, 确认后再生成" : "生成后会自动收进当前作品集"}</span></div></aside>`;
  }
  function shotBoard() {
    const jobs = currentJobs(),
      job = jobs.find((j) => j.task_id === generationGroup) || jobs[0];
    const review = plan && !generationGroup;
    if (review) {
      if (plan.state === "planning")
        return `<div class="shoot-wait"><span class="spinner"></span><h3>正在拆解画面与镜头</h3><p>可以离开页面, 回来继续查看.</p></div>`;
      if (plan.state !== "ready")
        return `<div class="shoot-wait"><h3>镜头规划未完成</h3><p>${esc(plan.error || "请重新规划")}</p></div>`;
      return `<div class="review-heading"><div><h3>确认这组镜头</h3><p>每张都可以修改, 也可以取消不需要的镜头.</p></div>${button(`生成 ${plan.shots.filter((s) => s.selected !== false).length} 张`, "shoot-run", "arrow", `class="primary" ${planMatches() ? "" : "disabled"}`)}</div><div class="shot-grid">${plan.shots.map((s, i) => `<article class="planned-shot"><div class="shot-number"><span>${String(i + 1).padStart(2, "0")}</span><input type="checkbox" aria-label="选择镜头 ${i + 1}" data-shot-check="${i}" ${s.selected !== false ? "checked" : ""}></div><h3>${esc(s.title)}</h3><textarea aria-label="镜头 ${i + 1} 提示词" data-shot-prompt="${i}">${esc(s.prompt)}</textarea><div class="shot-tags">${s.variation_focus.map((x) => `<span>${esc(x)}</span>`).join("")}</div></article>`).join("")}</div>`;
    }
    const items = job?.items?.length
      ? job.items
      : job?.asset_id
        ? [
            {
              item_id: "legacy",
              asset_id: job.asset_id,
              state: "completed",
              title: "生成图片",
            },
          ]
        : [];
    const generated = new Set(
      jobs
        .flatMap((j) => [...(j.items || []).map((i) => i.asset_id), j.asset_id])
        .filter(Boolean),
    );
    const localAssets = ids().filter((id) => !generated.has(id));
    return `${job ? `<div class="results-heading"><div><h3>${job.workflow ? modes.find((m) => m[0] === job.workflow)?.[1] || "作品" : "作品"}<span>${items.filter((i) => i.state === "completed").length} / ${items.length}</span></h3><small>${time(job.created)} · ${statuses[job.state] || job.state}</small></div><div>${["running", "queued"].includes(job.state) ? button("停止剩余任务", "cancel-job", "", `data-id="${job.task_id}"`) : ""}${items.some((i) => i.state === "failed") ? button("重试失败项", "shoot-retry", "refresh", `data-job="${job.task_id}"`) : ""}</div></div>` : ""}
      ${
        items.length || localAssets.length
          ? `<div class="shot-grid">${items.map((s, i) => (s.asset_id ? resultCard(s.asset_id, s.title, i + 1) : `<article class="result-card pending ${s.state}"><div class="pending-image">${["running", "queued"].includes(s.state) ? '<span class="spinner"></span>' : icon("info")}<span>${statuses[s.state] || s.state}</span></div><div class="result-caption"><span>${esc(s.title)}</span><span>${String(i + 1).padStart(2, "0")}</span></div>${s.error ? `<details><summary>查看原因</summary><p>${esc(s.error)}</p></details>` : ""}</article>`)).join("")}${localAssets.map((id, i) => resultCard(id, "已选图片", items.length + i + 1)).join("")}</div>`
          : `<div class="studio-empty"><div class="empty-frames"><span></span><span></span><span></span></div><h2>下一张, 从这里开始.</h2><p>描述画面直接生成, 或选一张照片继续创作.</p><div>${button("从画廊选图", "shoot-library", "gallery")}${button("上传图片", "upload", "upload")}</div><div class="empty-paths">${modes
              .slice(2)
              .map(
                ([id, label, subtitle]) =>
                  `<button data-action="shoot-mode" data-mode="${id}"><strong>${label}</strong><span>${subtitle}</span>${icon("arrow")}</button>`,
              )
              .join("")}</div></div>`
      }`;
  }
  function resultCard(id, title, index) {
    return `<article class="result-card ${focused === id ? "focused" : ""}"><button class="result-image" data-action="view-asset" data-id="${esc(id)}"><img data-asset="${esc(id)}" alt="${esc(title)}" loading="lazy"><span>查看大图 ${icon("zoom")}</span></button><div class="result-caption"><strong>${esc(title)}</strong><span>${String(index).padStart(2, "0")}</span></div><div class="result-actions">${button("变体", "shoot-from", "copy", `data-mode="variants" data-id="${id}"`)}${button("仿拍", "shoot-from", "canvas", `data-mode="recreate" data-id="${id}"`)}${button("编辑", "shoot-from", "edit", `data-mode="edit" data-id="${id}"`)}${ib("下载原图", "download", "download", `data-id="${id}"`)}</div></article>`;
  }
  function renderBoard() {
    const el = document.querySelector("#shoot-board");
    if (!el) return;
    const html = shotBoard();
    if (html === lastBoard && el.innerHTML) return;
    lastBoard = html;
    el.innerHTML = html;
    loadImages(el);
  }
  function renderWorkspace(el) {
    el.className = "shoot-page";
    el.innerHTML = `${modeTabs()}<div class="shoot-layout">${composer()}<section class="shoot-results"><div class="board-toolbar"><div class="view-switch">${button("选片", "shoot-view", "gallery", `data-view="board" class="${view === "board" ? "active" : ""}"`)}${button("自由画布", "shoot-view", "canvas", `data-view="canvas" class="${view === "canvas" ? "active" : ""}"`)}</div><div class="board-actions">${
      currentJobs().length
        ? `<select id="shoot-group" aria-label="选择生成批次"><option value="">最近一组</option>${currentJobs()
            .map((j) =>
              option(
                j.task_id,
                `${time(j.created)} · ${j.count || 1} 张`,
                generationGroup,
              ),
            )
            .join("")}</select>`
        : ""
    }${button("添加图片", "shoot-library", "plus")}</div></div>${view === "canvas" ? `<div class="workspace-stage"><div id="canvas" aria-label="创作画布" tabindex="0"></div>${getCanvas().toolbar()}<div class="workspace-corner"><span id="workspace-status">自动保存</span></div></div><div class="canvas-context"><span>点选图片, 选择下一步</span>${button("做变体", "shoot-selected", "copy", 'data-mode="variants"')}${button("仿拍", "shoot-selected", "canvas", 'data-mode="recreate"')}${button("改图", "shoot-selected", "edit", 'data-mode="edit"')}</div>` : '<div id="shoot-board"></div>'}</section></div>`;
    if (view === "canvas") {
      getCanvas().mount(el.querySelector("#canvas"));
      requestAnimationFrame(() => getCanvas().fit());
    } else renderBoard();
  }
  async function selectMode(next, asset) {
    mode = next;
    plan = null;
    generationGroup = "";
    if (asset) {
      focused = asset;
      if (special(mode)) source = asset;
      else {
        state.picked = [asset];
        state.prompt = "";
      }
    }
    if (special(mode)) {
      count = Math.max(count, 4);
      state.prompt = "";
    }
    persist();
    render();
    document
      .querySelector(".shoot-composer")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function payload() {
    if (special(mode) && !planMatches())
      throw Error("配置已变化, 请重新规划镜头");
    return {
      workflow: mode,
      count,
      source_asset: source,
      plan_id: plan?.id,
      shots: plan?.shots.filter((s) => s.selected !== false),
      target_character: target,
    };
  }
  async function action(action, el) {
    if (!action.startsWith("shoot-")) return false;
    if (action === "shoot-mode") await selectMode(el.dataset.mode);
    if (action === "shoot-from") {
      const a = (await query("asset", { id: el.dataset.id, thumbnail: "1" }))
        .asset;
      if (!ids().includes(a.id)) await getCanvas().add(a);
      state.page = "workspace";
      close();
      await selectMode(el.dataset.mode, a.id);
      shell();
    }
    if (action === "shoot-selected") {
      const l = getCanvas().doc.layers.find(
        (l) => l.id === getCanvas().selected,
      );
      if (!l) throw Error("先点选一张画布图片");
      await selectMode(el.dataset.mode, l.asset_id);
    }
    if (action === "shoot-view") {
      view = el.dataset.view;
      persist();
      render();
    }
    if (action === "shoot-library")
      await picker(async (a) => {
        close();
        await getCanvas().add(a);
        focused = a.id;
        render();
      });
    if (action === "shoot-source")
      await picker(async (a) => {
        source = a.id;
        plan = null;
        generationGroup = "";
        close();
        persist();
        render();
      }, "选择" + sourceNames[mode]);
    if (action === "shoot-plus" || action === "shoot-minus") {
      count = Math.max(
        1,
        Math.min(12, count + (action === "shoot-plus" ? 1 : -1)),
      );
      persist();
      render();
    }
    if (action === "shoot-plan") {
      if (planning) return true;
      if (!source) throw Error("先选择" + sourceNames[mode]);
      if (!planner) throw Error("先选择镜头规划模型");
      planning = true;
      render();
      try {
        const body = {
          workflow: mode,
          count,
          source_asset: source,
          planner,
          characters: state.cast,
          target_character: target,
          prompt: state.prompt.trim(),
          output: state.output,
          workspace_id: getCanvas().doc.id,
        };
        plan = await sendOnce("plan", body);
        generationGroup = "";
        persist();
        await saveCanvas();
        render();
      } finally {
        planning = false;
        render();
      }
    }
    if (action === "shoot-retry") {
      const job = state.jobs.find((j) => j.task_id === el.dataset.job),
        failed = job.items.filter((i) => i.state === "failed");
      if (!failed.length) return true;
      dialog(
        "重试失败镜头",
        `<p>只重新生成 ${failed.length} 张失败图片, 成功结果保留.</p>`,
        button(
          "生成失败项",
          "shoot-retry-confirm",
          "refresh",
          `data-job="${job.task_id}" class="primary"`,
        ),
      );
    }
    if (action === "shoot-retry-confirm") {
      const job = state.jobs.find((j) => j.task_id === el.dataset.job);
      el.disabled = true;
      try {
        const result = await sendOnce("retry", { task_id: job.task_id });
        generationGroup = result.task_id;
        close();
        toast("失败镜头已重新加入任务");
        persist();
      } finally {
        el.disabled = false;
      }
    }
    return true;
  }
  function change(e) {
    if (e.target.id === "shoot-count")
      count = Math.max(
        1,
        Math.min(12, Math.trunc(Number(e.target.value) || 1)),
      );
    if (e.target.id === "shoot-planner") planner = e.target.value;
    if (e.target.id === "shoot-target") target = e.target.value;
    if (e.target.id === "shoot-group") {
      generationGroup = e.target.value;
      plan = null;
      renderBoard();
    }
    if (e.target.dataset.shotCheck !== undefined) {
      plan.shots[Number(e.target.dataset.shotCheck)].selected =
        e.target.checked;
      renderBoard();
    }
    if (["shoot-count", "shoot-target", "output"].includes(e.target.id))
      requestAnimationFrame(render);
    persist();
  }
  function onInput(e) {
    if (e.target.id === "prompt" && plan) requestAnimationFrame(renderBoard);
    if (e.target.dataset.shotPrompt !== undefined) {
      plan.shots[Number(e.target.dataset.shotPrompt)].prompt = e.target.value;
      persist();
    }
  }
  async function poll() {
    if (plan?.state === "planning") {
      const next = await query("plan", { id: plan.id });
      if (next && next.state !== plan.state) {
        plan = next;
        persist();
        if (state.page === "workspace") render();
      }
    }
    if (
      state.page === "workspace" &&
      view === "board" &&
      !document.querySelector(".planned-shot textarea:focus")
    )
      renderBoard();
  }
  function submitted(id) {
    generationGroup = id;
    persist();
    renderBoard();
  }
  return {
    renderWorkspace,
    renderBoard,
    restore,
    capture,
    action,
    change,
    onInput,
    payload,
    poll,
    submitted,
    request: sendOnce,
    selectMode,
    get mode() {
      return mode;
    },
    get submitting() {
      return submitting;
    },
    get view() {
      return view;
    },
  };
}
