import { assetURL } from "./api.js";
import { uid } from "./id.js";
import { esc, icon, ib, toast } from "./ui.js";

export class Canvas {
  constructor(workspace, changed) {
    this.doc = structuredClone(workspace);
    this.changed = changed;
    this.selected = null;
    this.zoom = 0.4;
    this.pan = { x: 80, y: 70 };
    this.undoStack = [];
    this.redoStack = [];
    this.pointers = new Map();
    this.tool = "select";
  }
  snapshot() {
    this.undoStack.push(JSON.stringify(this.doc.layers));
    if (this.undoStack.length > 50) this.undoStack.shift();
    this.redoStack = [];
  }
  update() {
    this.changed(this.doc);
    this.paint();
  }
  undo(redo = false) {
    const source = redo ? this.redoStack : this.undoStack,
      target = redo ? this.undoStack : this.redoStack;
    if (!source.length) return;
    target.push(JSON.stringify(this.doc.layers));
    this.doc.layers = JSON.parse(source.pop());
    this.update();
  }
  async add(asset) {
    this.snapshot();
    const width = Math.min(asset.width, 900);
    this.doc.layers.push({
      id: uid(),
      asset_id: asset.id,
      x: this.doc.layers.length * 60,
      y: this.doc.layers.length * 50,
      width,
      height: (width * asset.height) / asset.width,
      rotation: 0,
    });
    this.selected = this.doc.layers.at(-1).id;
    this.update();
  }
  mount(host) {
    this.host = host;
    host.innerHTML =
      '<div class="canvas-world"></div><div class="canvas-empty"><span>把灵感放进画布</span><small>上传图片, 或从画廊选择</small></div>';
    this.paint();
    host.onpointerdown = (e) => {
      if (e.target.closest("button")) return;
      host.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const node = e.target.closest("[data-layer]");
      if (node && this.tool !== "hand") this.selected = node.dataset.layer;
      else this.selected = null;
      this.start = {
        x: e.clientX,
        y: e.clientY,
        pan: { ...this.pan },
        layer: this.selected
          ? structuredClone(this.doc.layers.find((l) => l.id === this.selected))
          : null,
        resize: e.target.classList.contains("resize-handle"),
      };
      this.snapshot();
      this.paint();
    };
    host.onpointermove = (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      const old = [...this.pointers.values()];
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pointers.size === 2) {
        const now = [...this.pointers.values()];
        const dist = (p) => Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
        this.zoom = Math.max(
          0.05,
          Math.min(3, (this.zoom * dist(now)) / Math.max(1, dist(old))),
        );
        this.paint();
        return;
      }
      if (!this.start) return;
      const dx = e.clientX - this.start.x,
        dy = e.clientY - this.start.y;
      if (this.start.layer) {
        const l = this.doc.layers.find((l) => l.id === this.selected);
        if (!l) return;
        if (this.start.resize) {
          l.width = Math.max(40, this.start.layer.width + dx / this.zoom);
          l.height =
            (l.width * this.start.layer.height) / this.start.layer.width;
        } else {
          l.x = this.start.layer.x + dx / this.zoom;
          l.y = this.start.layer.y + dy / this.zoom;
        }
      } else this.pan = { x: this.start.pan.x + dx, y: this.start.pan.y + dy };
      this.paint();
    };
    host.onpointerup = (e) => {
      this.pointers.delete(e.pointerId);
      this.start = null;
      this.changed(this.doc);
    };
    host.onpointercancel = host.onpointerup;
    host.onwheel = (e) => {
      e.preventDefault();
      const rect = host.getBoundingClientRect(),
        x = e.clientX - rect.left,
        y = e.clientY - rect.top;
      const next = Math.max(
        0.05,
        Math.min(3, this.zoom * Math.exp(-e.deltaY * 0.001)),
      );
      this.pan = {
        x: x - ((x - this.pan.x) * next) / this.zoom,
        y: y - ((y - this.pan.y) * next) / this.zoom,
      };
      this.zoom = next;
      this.paint();
    };
  }
  paint() {
    if (!this.host?.isConnected) return;
    const world = this.host.querySelector(".canvas-world");
    world.style.transform = `translate(${this.pan.x}px,${this.pan.y}px) scale(${this.zoom})`;
    const existing = new Map(
      [...world.children].map((e) => [e.dataset.layer, e]),
    );
    for (const l of this.doc.layers) {
      let el = existing.get(l.id);
      if (!el) {
        el = document.createElement("div");
        el.dataset.layer = l.id;
        el.innerHTML = `<img alt="画布图片" draggable="false"><span class="resize-handle"></span>`;
        world.append(el);
        assetURL(l.asset_id)
          .then((url) => {
            el.querySelector("img").src = url;
          })
          .catch(() => toast("图片加载失败"));
      }
      existing.delete(l.id);
      el.className =
        "canvas-layer" + (this.selected === l.id ? " selected" : "");
      Object.assign(el.style, {
        left: l.x + "px",
        top: l.y + "px",
        width: l.width + "px",
        height: l.height + "px",
        transform: `rotate(${l.rotation}deg)`,
      });
      world.append(el);
    }
    existing.forEach((el) => el.remove());
    this.host.querySelector(".canvas-empty").hidden = !!this.doc.layers.length;
    const use = document.querySelector("#use-layer");
    if (use) use.disabled = !this.selected;
    const z = document.querySelector("#zoom-value");
    if (z) z.textContent = Math.round(this.zoom * 100) + "%";
  }
  mutate(action) {
    const l = this.doc.layers.find((l) => l.id === this.selected);
    if (!l) {
      toast("先选择一张图片");
      return;
    }
    this.snapshot();
    if (action === "remove")
      this.doc.layers = this.doc.layers.filter((x) => x !== l);
    if (action === "rotate") l.rotation = (l.rotation + 90) % 360;
    if (action === "front")
      this.doc.layers = [...this.doc.layers.filter((x) => x !== l), l];
    if (action === "copy")
      this.doc.layers.push({
        ...l,
        id: uid(),
        x: l.x + 40,
        y: l.y + 40,
      });
    this.update();
  }
  fit() {
    if (!this.host?.isConnected) return;
    const ls = this.doc.layers;
    if (!ls.length) {
      this.pan = { x: 60, y: 60 };
      this.zoom = 0.4;
      this.paint();
      return;
    }
    const x = Math.min(...ls.map((l) => l.x)),
      y = Math.min(...ls.map((l) => l.y));
    const w = Math.max(...ls.map((l) => l.x + l.width)) - x,
      h = Math.max(...ls.map((l) => l.y + l.height)) - y;
    this.zoom = Math.min(
      (this.host.clientWidth - 80) / Math.max(1, w),
      (this.host.clientHeight - 80) / Math.max(1, h),
      1,
    );
    this.pan = { x: 40 - x * this.zoom, y: 40 - y * this.zoom };
    this.paint();
  }
  async exportBlob() {
    if (!this.doc.layers.length) throw Error("画布还是空的");
    const ls = this.doc.layers;
    const corners = ls.flatMap((l) => {
      const r = (l.rotation * Math.PI) / 180,
        cx = l.x + l.width / 2,
        cy = l.y + l.height / 2;
      return [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ].map(([x, y]) => ({
        x:
          cx +
          ((x * l.width) / 2) * Math.cos(r) -
          ((y * l.height) / 2) * Math.sin(r),
        y:
          cy +
          ((x * l.width) / 2) * Math.sin(r) +
          ((y * l.height) / 2) * Math.cos(r),
      }));
    });
    const minX = Math.min(...corners.map((p) => p.x)),
      minY = Math.min(...corners.map((p) => p.y)),
      w = Math.ceil(Math.max(...corners.map((p) => p.x)) - minX),
      h = Math.ceil(Math.max(...corners.map((p) => p.y)) - minY);
    if (w * h > 40_000_000) throw Error("画布导出超过 4000 万像素, 请缩小图层");
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    for (const l of ls) {
      const img = new Image();
      img.src = await assetURL(l.asset_id, false);
      await img.decode();
      ctx.save();
      ctx.translate(l.x + l.width / 2 - minX, l.y + l.height / 2 - minY);
      ctx.rotate((l.rotation * Math.PI) / 180);
      ctx.drawImage(img, -l.width / 2, -l.height / 2, l.width, l.height);
      ctx.restore();
    }
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  }
  toolbar() {
    return `<div class="canvas-tools">${ib("上传图片", "upload", "upload")}${ib("从画廊添加", "canvas-library", "gallery")}${ib("平移画布", "hand", "hand")}${ib("撤销", "undo", "undo")}${ib("重做", "redo", "redo")}<i></i>${ib("裁剪", "crop", "crop")}${ib("复制图层", "layer-copy", "copy")}${ib("旋转", "layer-rotate", "refresh")}${ib("移到最前", "layer-front", "up")}${ib("移除图层", "layer-remove", "trash")}<i></i>${ib("适应画布", "fit", "zoom")}<span id="zoom-value">40%</span></div>`;
  }
}
