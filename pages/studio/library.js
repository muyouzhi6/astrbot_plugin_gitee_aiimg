import { api, query, loadImages } from "./api.js";
import { esc, icon, button, ib, dialog, close, toast } from "./ui.js";

export function createLibrary({
  state,
  render,
  getAssets,
  viewAsset,
  refresh,
  saveCanvas,
}) {
  const selected = new Set();
  let selecting = false;
  function assetActions(a) {
    return `${button(a.favorite ? "取消收藏" : "收藏防删", "lib-favorite", "star", `data-id="${a.id}" data-value="${!a.favorite}" class="${a.favorite ? "favorited" : ""}"`)}${ib("删除图片", "lib-delete", "trash", `data-id="${a.id}" ${a.favorite ? "disabled" : ""}`)}`;
  }
  function renderLibrary(el) {
    const storage = state.storage || {};
    el.className = "standard-page library-page";
    el.innerHTML = `<div class="library-toolbar"><div class="segments">${[
      ["history", "生成历史"],
      ["", "全部图片"],
      ["favorite", "收藏夹"],
      ["reference", "参考图"],
    ]
      .map(([id, label]) =>
        button(
          label,
          "gallery-kind",
          "",
          `data-kind="${id}" class="${state.kind === id ? "selected" : ""}"`,
        ),
      )
      .join(
        "",
      )}</div><div>${button(selecting ? "完成选择" : "批量管理", "lib-select", selecting ? "check" : "gallery")}${ib("存储设置", "lib-storage", "settings")}</div></div>
      <div class="library-count"><span>${storage.count ?? state.assets.length} 张图片 · ${storage.favorites || 0} 张收藏</span>${storage.max_count ? `<span>上限 ${storage.max_count} 张${storage.count > storage.max_count ? " · 受保护或新图片暂超上限" : ""}</span>` : ""}</div>
      ${selecting ? `<div class="selection-bar"><span>已选 ${selected.size} 张</span>${button("全选本页", "lib-all")}${button("收藏所选", "lib-favorite-all", "star")}${button("删除所选", "lib-delete-all", "trash", selected.size ? "" : "disabled")}</div>` : ""}
      <div class="asset-grid">${state.assets.map((a) => `<article class="library-card ${selected.has(a.id) ? "selected" : ""}"><button class="asset-card" data-action="${selecting ? "lib-toggle" : "view-asset"}" data-id="${a.id}"><div><img data-asset="${a.id}" alt="${esc(a.title || a.name || a.model || "图片")}" loading="lazy"></div><span>${esc(a.title || a.model || a.name || "图片")}<small>${a.width} × ${a.height}</small></span></button>${selecting ? `<input type="checkbox" aria-label="选择图片" data-lib-check="${a.id}" ${selected.has(a.id) ? "checked" : ""}>` : ""}<div class="library-card-actions">${ib(a.favorite ? "取消收藏" : "收藏防删", "lib-favorite", "star", `data-id="${a.id}" data-value="${!a.favorite}" class="icon-button ${a.favorite ? "favorited" : ""}"`)}${ib("删除图片", "lib-delete", "trash", `data-id="${a.id}" ${a.favorite ? "disabled" : ""}`)}</div></article>`).join("")}</div>${state.more ? button("加载更多", "more-assets", "", 'class="load-more"') : ""}${!state.assets.length ? '<p class="library-empty">这里还没有图片.</p>' : ""}`;
    el.querySelectorAll("[data-lib-check]").forEach(
      (input) =>
        (input.onchange = () => {
          toggle(input.dataset.libCheck);
          render();
        }),
    );
    loadImages(el);
  }
  function toggle(id) {
    if (selected.has(id)) selected.delete(id);
    else selected.add(id);
  }
  async function reload() {
    await getAssets();
    state.storage = await query("storage", {});
    render();
  }
  function deletion(ids) {
    dialog(
      "删除图片",
      `<p>删除所选 ${ids.length} 张图片及原图文件? 画布中的对应图片会一并移除.</p><p class="field-help">收藏、人物身份参考和工作流引用会跳过. 此操作不可撤销.</p>`,
      button(
        "确认删除",
        "lib-confirm-delete",
        "trash",
        `data-ids="${esc(JSON.stringify(ids))}"`,
      ),
    );
  }
  async function action(action, el) {
    if (!action.startsWith("lib-")) return false;
    if (action === "lib-select") {
      selecting = !selecting;
      selected.clear();
      render();
    }
    if (action === "lib-toggle") {
      toggle(el.dataset.id);
      render();
    }
    if (action === "lib-all") {
      state.assets.forEach((a) => selected.add(a.id));
      render();
    }
    if (action === "lib-favorite" || action === "lib-favorite-all") {
      const ids = action === "lib-favorite" ? [el.dataset.id] : [...selected];
      if (!ids.length) throw Error("先选择图片");
      const value =
        action === "lib-favorite-all" || el.dataset.value === "true";
      await api("favorite", { ids, value });
      if (document.querySelector(".asset-detail")) await viewAsset(ids[0]);
      else await reload();
      toast(value ? "已收藏, 删除与自动清理都会跳过" : "已取消收藏");
    }
    if (action === "lib-delete") deletion([el.dataset.id]);
    if (action === "lib-delete-all") deletion([...selected]);
    if (action === "lib-confirm-delete") {
      el.disabled = true;
      try {
        if (state.canvasDirty) await saveCanvas();
        const r = await api("delete-assets", {
          ids: JSON.parse(el.dataset.ids),
        });
        selected.clear();
        close();
        await refresh();
        await reload();
        if (r.skipped.length)
          dialog(
            "删除结果",
            `<p>已删除 ${r.deleted.length} 张, 保留 ${r.skipped.length} 张.</p><p class="field-help">${[...new Set(r.skipped.map((x) => x.reason))].map(esc).join(" · ")}</p>`,
          );
        else toast(`已删除 ${r.deleted.length} 张`);
      } finally {
        el.disabled = false;
      }
    }
    if (action === "lib-storage") {
      state.storage = await query("storage", {});
      const s = state.storage;
      dialog(
        "存储设置",
        `<div class="storage-summary"><strong>${s.count} 张</strong><span>${(s.bytes / 1024 / 1024).toFixed(1)} MB 原图 · ${s.protected} 张受保护</span></div><label class="field"><span>最大保留数量</span><input id="library-limit" type="number" min="0" max="100000" value="${s.max_count}"><small>0 为不限. 超过时清理最旧的普通图片, 收藏与引用中的图片保留. 新图片保留至少 5 分钟.</small></label><p class="field-help">降低上限会立即清理符合条件的图片.</p>`,
        button("保存并应用", "lib-save-storage", "check", 'class="primary"'),
      );
    }
    if (action === "lib-save-storage") {
      const max_count = Number(document.querySelector("#library-limit").value);
      el.disabled = true;
      try {
        if (state.canvasDirty) await saveCanvas();
        const r = await api("storage", {
          revision: state.storage.revision,
          max_count,
        });
        close();
        await refresh();
        await reload();
        toast(
          `设置已保存${r.deleted?.length ? `, 已清理 ${r.deleted.length} 张` : ""}`,
        );
      } finally {
        el.disabled = false;
      }
    }
    return true;
  }
  return { render: renderLibrary, action, assetActions };
}
