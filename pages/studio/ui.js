export const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
import { iconNodes } from "./lucide-icons.js";
export const icon = (name) =>
  `<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${(
    iconNodes[name] || iconNodes.info
  )
    .map(
      ([tag, attributes]) =>
        `<${tag} ${Object.entries(attributes)
          .filter(([key]) => key !== "key")
          .map(([key, value]) => `${key}="${esc(value)}"`)
          .join(" ")} />`,
    )
    .join("")}</svg>`;
export const button = (label, action, symbol = "", extra = "") =>
  `<button type="button" data-action="${action}" ${extra}>${symbol ? icon(symbol) : ""}<span>${esc(label)}</span></button>`;
export const ib = (label, action, symbol, extra = "") =>
  `<button type="button" class="icon-button" title="${esc(label)}" aria-label="${esc(label)}" data-action="${action}" ${extra}>${icon(symbol)}</button>`;
export function toast(message) {
  const el = document.querySelector("#toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 4500);
}
export function dialog(title, html, footer = "") {
  document.querySelector("#overlay").innerHTML =
    `<div class="scrim"><section class="sheet" role="dialog" aria-modal="true" aria-label="${esc(title)}"><header><h2>${esc(title)}</h2>${ib("关闭", "close", "close")}</header><div class="sheet-body">${html}</div>${footer ? `<footer>${footer}</footer>` : ""}</section></div>`;
  document
    .querySelector("#overlay input, #overlay textarea, #overlay button")
    ?.focus();
}
export const close = () => {
  document.querySelector("#overlay").innerHTML = "";
};
export function field(label, input, hint = "") {
  return `<label class="field"><span>${esc(label)}</span>${input}${hint ? `<small>${esc(hint)}</small>` : ""}</label>`;
}
export const input = (name, value = "", type = "text", extra = "") =>
  `<input name="${name}" type="${type}" value="${esc(value)}" ${extra}>`;
export const option = (value, label, selected) =>
  `<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(label)}</option>`;
export const empty = (text, action = "upload", label = "上传图片") =>
  `<div class="empty"><div class="empty-mark">${icon("canvas")}</div><p>${esc(text)}</p>${action ? button(label, action, "plus") : ""}</div>`;
export const time = (t) =>
  new Date(t * 1000).toLocaleString("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
