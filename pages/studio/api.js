let bridge;
export async function ready() {
  bridge = window.AstrBotPluginPage;
  if (!bridge) throw Error("请从 AstrBot 插件详情页打开工作台");
  await bridge.ready();
}
function unwrap(result) {
  if (result?.ok === false) throw Error(result.error || "操作未完成");
  if (result?.ok === true) return result.data;
  if (result?.data?.ok === true) return result.data.data;
  if (result?.data?.ok === false) throw Error(result.data.error);
  throw Error("服务响应异常, 请刷新页面");
}
export async function api(action, body) {
  try {
    return unwrap(
      await (body === undefined
        ? bridge.apiGet("studio/" + action)
        : bridge.apiPost("studio/" + action, body)),
    );
  } catch (e) {
    throw Error(e.message || "连接中断, 请稍后重试");
  }
}
export const query = (action, params) =>
  bridge.apiGet("studio/" + action, params).then(unwrap);
export const upload = (file) =>
  bridge.upload("studio/upload", file).then(unwrap);
export const download = (id) => bridge.download("studio/download", { id });
const thumbs = new Map();
export async function assetURL(id, thumbnail = true) {
  const k = id + thumbnail;
  if (!thumbs.has(k)) {
    if (thumbs.size >= 120) thumbs.delete(thumbs.keys().next().value);
    thumbs.set(
      k,
      query("asset", {
        id,
        thumbnail: thumbnail === "preview" ? "preview" : thumbnail ? "1" : "0",
      })
        .then((r) => r.data_url)
        .catch((e) => {
          thumbs.delete(k);
          throw e;
        }),
    );
  }
  return thumbs.get(k);
}
export async function loadImages(root = document) {
  const imgs = [...root.querySelectorAll("img[data-asset]")];
  let i = 0;
  await Promise.all(
    Array.from({ length: Math.min(imgs.length, 4) }, async () => {
      while (i < imgs.length) {
        const img = imgs[i++];
        try {
          const src = await assetURL(img.dataset.asset);
          if (img.isConnected) img.src = src;
        } catch {
          img.alt = "图片暂不可用";
        }
      }
    }),
  );
}
