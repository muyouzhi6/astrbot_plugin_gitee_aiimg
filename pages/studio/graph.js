import { api, query, loadImages } from "./api.js";
import { uid } from "./id.js";
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

const names = {
  text: "提示词",
  image: "参考图片",
  person: "出镜人物",
  plan: "镜头规划",
  generate: "生成图片",
  output: "结果输出",
};
const symbols = {
  text: "edit",
  image: "canvas",
  person: "people",
  plan: "tasks",
  generate: "play",
  output: "gallery",
};
const ports = { text: "提示词", image: "图片", people: "人物", plan: "镜头" };
const statusNames = {
  running: "执行中",
  pending: "等待",
  completed: "完成",
  failed: "失败",
  cancelled: "已停止",
  interrupted: "已中断",
  skipped: "已跳过",
};

export function createGraphEditor({ state, render, picker, viewAsset }) {
  let doc = null,
    selected = "",
    linking = "",
    dirty = false,
    run = null,
    zoom = 0.85,
    pan = { x: 25, y: 35 },
    saving = false;
  let runKey = "",
    autoView = true,
    activeDrag = null;
  function restore() {
    if (!doc) doc = structuredClone(state.graphs?.[0] || null);
    else if (!dirty) {
      const found = state.graphs.find((g) => g.id === doc.id);
      if (found) doc = structuredClone(found);
    }
    if (doc && !run)
      run =
        [...(state.graphRuns || [])]
          .reverse()
          .find((r) => r.graph_id === doc.id) || null;
    runKey = doc?.pending_run || "";
    if (run && run.state !== "running" && run.id === runKey) runKey = "";
  }
  function node(type, x, y) {
    return {
      id: uid(),
      type,
      x,
      y,
      ...(type === "generate"
        ? { count: 1, output: "3:4 4K", provider: "" }
        : {}),
      ...(type === "plan"
        ? { count: 4, output: "3:4 4K", workflow: "variants", planner: "" }
        : {}),
      ...(type === "person" ? { characters: [] } : {}),
      ...(type === "text" ? { text: "" } : {}),
    };
  }
  function template(kind) {
    const text = node("text", 40, 70),
      image = node("image", 40, 340),
      people = node("person", 40, 590),
      plan = node("plan", 360, 160),
      generate = node("generate", 680, 160),
      output = node("output", 1000, 160);
    text.text =
      kind === "simple"
        ? "一只陶瓷杯, 窗边自然光, 极简产品摄影"
        : "保留主体与风格, 设计不同机位的镜头";
    if (kind === "simple") {
      generate.x = 370;
      output.x = 700;
      return {
        name: "自由生图",
        nodes: [text, generate, output],
        edges: [
          { from: text.id, to: generate.id, port: "text" },
          { from: generate.id, to: output.id, port: "image" },
        ],
      };
    }
    plan.workflow = kind === "recreate" ? "recreate" : "variants";
    const nodes = [text, image, plan, generate, output],
      edges = [
        { from: text.id, to: plan.id, port: "text" },
        { from: image.id, to: plan.id, port: "image" },
        { from: plan.id, to: generate.id, port: "plan" },
        { from: generate.id, to: output.id, port: "image" },
      ];
    if (kind === "recreate") {
      nodes.push(people);
      edges.push({ from: people.id, to: plan.id, port: "people" });
    }
    return {
      name: kind === "recreate" ? "人物仿拍" : "成片变体",
      nodes,
      edges,
    };
  }
  function mark() {
    dirty = true;
    runKey = "";
    if (doc) delete doc.pending_run;
    document.querySelector("#graph-save")?.removeAttribute("disabled");
    const label = document.querySelector("#graph-save span");
    if (label) label.textContent = "保存修改";
    const status = document.querySelector(".node-save-status");
    if (status) status.textContent = "有未保存修改";
    steps();
  }
  function issue(n) {
    const incoming = doc.edges.filter((e) => e.to === n.id),
      has = (p) => incoming.some((e) => e.port === p);
    if (n.type !== "output" && !doc.edges.some((e) => e.from === n.id))
      return "连接到下一步";
    if (n.type === "text" && !n.text?.trim()) return "填写画面描述";
    if (n.type === "image" && !n.asset_id) return "选择参考图片";
    if (n.type === "person" && !n.characters?.length) return "选择出镜人物";
    if (n.type === "plan") {
      if (!has("image")) return "连接参考图片";
      if (!n.planner) return "选择规划模型";
      if (["recreate", "outfit"].includes(n.workflow) && !has("people"))
        return "连接出镜人物";
    }
    if (n.type === "generate") {
      if (has("plan") && incoming.length > 1)
        return "保留镜头输入, 移除重复输入";
      if (!has("plan") && !has("text")) return "连接提示词或镜头";
      if (
        n.provider &&
        !state.config.providers.some((p) => p.id === n.provider)
      )
        return "重新选择服务商";
    }
    if (n.type === "output" && !has("image")) return "连接生成图片";
    return "";
  }
  function steps() {
    const el = document.querySelector(".graph-steps");
    if (!el || !doc) return;
    el.innerHTML = doc.nodes
      .map(
        (n, i) =>
          `<button type="button" data-action="flow-select" data-id="${n.id}" class="${selected === n.id ? "active" : ""}" aria-pressed="${selected === n.id}"><span class="step-number">${i + 1}</span><span><strong>${esc(n.label || names[n.type])}</strong><small>${esc(issue(n) || "可编辑")}</small></span>${icon(issue(n) ? "edit" : "check")}</button>`,
      )
      .join("");
    const label = document.querySelector(".graph-readiness");
    if (label) {
      const count = doc.nodes.filter(issue).length;
      label.textContent = count ? `${count} 项待设置` : "设置就绪";
    }
  }
  function selectNode(id, scroll = false) {
    selected = id;
    paint();
    inspector();
    steps();
    if (scroll && matchMedia("(max-width: 900px)").matches)
      document
        .querySelector(".graph-inspector")
        ?.scrollIntoView({ block: "start", behavior: "smooth" });
  }
  function renderGraph(el) {
    el.className = "graph-page";
    if (!doc) {
      el.innerHTML = `<div class="graph-welcome"><span class="eyebrow">可编排的创作</span><h2>把想法连成工作流.</h2><p>从模板开始, 拖动节点, 连接图片、提示词与模型.</p><div>${button("自由生图", "flow-new", "plus", 'data-template="simple"')}${button("成片变体", "flow-new", "copy", 'data-template="variants"')}${button("人物仿拍", "flow-new", "people", 'data-template="recreate"')}</div>${button("导入工作流", "flow-import", "upload")}</div>`;
      return;
    }
    if (!doc.nodes.some((n) => n.id === selected))
      selected = (doc.nodes.find(issue) || doc.nodes[0])?.id || "";
    el.innerHTML = `<div class="graph-toolbar"><div><select id="graph-select" aria-label="选择工作流">${state.graphs.map((g) => option(g.id, g.name, doc.id)).join("")}${!doc.id ? option("", doc.name, "") : ""}</select>${ib("新建工作流", "flow-new", "plus")}${ib("重命名工作流", "flow-rename", "edit")}</div><div>${button("导入", "flow-import", "upload")}${button("导出", "flow-export", "download")}${button(dirty ? "保存修改" : "已保存", "flow-save", "check", `id="graph-save" ${dirty ? "" : "disabled"}`)}${run?.state === "running" ? button("停止", "flow-stop", "close") : button("运行工作流", "flow-run", "play", 'class="primary"')}</div></div><div class="graph-setup-heading"><span>点击步骤, 编辑内容</span><span class="graph-readiness"></span></div><div class="graph-steps" aria-label="工作流步骤"></div>
      <div class="graph-body"><div class="graph-stage-wrap"><div class="node-palette">${Object.keys(
        names,
      )
        .map((type) =>
          button(
            names[type],
            "flow-add",
            symbols[type],
            `data-type="${type}" draggable="true"`,
          ),
        )
        .join(
          "",
        )}</div><div class="graph-stage" tabindex="0" aria-label="节点工作流画布"><div class="graph-world"><svg class="graph-wires" aria-hidden="true"></svg><div class="graph-nodes"></div></div><div class="graph-hint">点击编辑 · 拖动标题移动节点 · 输出接输入</div></div><div class="graph-controls">${ib("缩小", "flow-zoom-out", "back")}<span id="graph-zoom">${Math.round(zoom * 100)}%</span>${ib("放大", "flow-zoom-in", "plus")}${button("适应画布", "flow-fit", "zoom")}${button("取消连线", "flow-unlink", "close", `class="unlink-control" ${linking ? "" : "hidden"}`)}</div></div><aside class="graph-inspector"></aside></div><div class="graph-run-summary"></div>`;
    steps();
    paint();
    inspector();
    summary();
    bind();
    document
      .querySelector(".graph-toolbar>div")
      ?.insertAdjacentHTML(
        "beforeend",
        ib("删除工作流", "flow-delete", "trash"),
      );
    if (autoView) {
      autoView = false;
      requestAnimationFrame(fit);
    }
  }
  function coordinates(n, port) {
    const keys = Object.keys(state.nodeTypes[n.type]?.inputs || {});
    return {
      x: n.x + (port ? 0 : 224),
      y: n.y + 90 + (port ? keys.indexOf(port) * 32 : 0),
    };
  }
  function runMatches() {
    if (!run?.graph) return false;
    const signature = (graph) =>
      JSON.stringify({
        nodes: graph.nodes.map(({ x, y, label, ...rest }) => rest),
        edges: graph.edges,
      });
    return signature(doc) === signature(run.graph);
  }
  function paint() {
    const root = document.querySelector(".graph-stage");
    if (!root || !doc) return;
    root.querySelector(".graph-world").style.transform =
      `translate(${pan.x}px,${pan.y}px) scale(${zoom})`;
    root.querySelector(".graph-nodes").innerHTML = doc.nodes
      .map((n) => {
        const result = runMatches() ? run?.nodes?.[n.id] : null;
        const inputs = state.nodeTypes[n.type]?.inputs || {},
          output = state.nodeTypes[n.type]?.output;
        const detail =
          n.type === "text"
            ? n.text || "填写提示词"
            : n.type === "plan"
              ? `${{ variants: "变体", recreate: "仿拍", outfit: "换装" }[n.workflow]} · ${n.count} 张`
              : n.type === "generate"
                ? state.config.providers.find((p) => p.id === n.provider)
                    ?.label ||
                  n.provider ||
                  "按回退链路"
                : n.type === "person"
                  ? `${n.characters?.length || 0} 位人物`
                  : "";
        return `<article class="flow-node ${selected === n.id ? "selected" : ""} ${result?.state || ""}" data-node="${n.id}" style="left:${n.x}px;top:${n.y}px"><header data-drag-node="${n.id}">${icon(symbols[n.type])}<strong>${esc(n.label || names[n.type])}</strong><span class="node-state">${statusNames[result?.state] || ""}</span></header><div class="node-detail">${esc(detail)}</div><div class="node-ports">${Object.entries(
          inputs,
        )
          .map(
            ([key, type]) =>
              `<button class="node-input ${linking && state.nodeTypes[doc.nodes.find((n) => n.id === linking)?.type]?.output === type ? "compatible" : ""}" data-action="flow-connect" data-node="${n.id}" data-port="${key}" aria-label="${names[n.type]}的${ports[key]}输入"><i></i>${ports[key]}</button>`,
          )
          .join(
            "",
          )}${output ? `<button class="node-output" data-action="flow-link" data-node="${n.id}" aria-label="${names[n.type]}输出"><span>${{ text: "文字", images: "图片", people: "人物", plan: "镜头" }[output]}</span><i></i></button>` : ""}</div>${n.type === "image" && n.asset_id ? `<button class="node-preview" data-action="flow-select" data-id="${n.id}"><img data-asset="${n.asset_id}" alt="节点参考图"></button>` : ""}${button(issue(n) || "编辑节点", "flow-select", "edit", `data-id="${n.id}" class="node-edit" aria-label="编辑${esc(n.label || names[n.type])}"`)}</article>`;
      })
      .join("");
    root.querySelector("svg.graph-wires").innerHTML = doc.edges
      .map((e) => {
        const a = coordinates(doc.nodes.find((n) => n.id === e.from)),
          b = coordinates(
            doc.nodes.find((n) => n.id === e.to),
            e.port,
          ),
          bend = Math.max(65, Math.abs(b.x - a.x) * 0.45);
        return `<path d="M${a.x} ${a.y} C${a.x + bend} ${a.y},${b.x - bend} ${b.y},${b.x} ${b.y}"/>`;
      })
      .join("");
    loadImages(root);
    const z = document.querySelector("#graph-zoom");
    if (z) z.textContent = Math.round(zoom * 100) + "%";
    const hint = document.querySelector(".graph-hint");
    if (hint)
      hint.textContent = linking
        ? "点击同类型的输入圆点完成连线"
        : "点击编辑 · 拖动标题移动节点 · 输出接输入";
    const cancel = document.querySelector(".unlink-control");
    if (cancel) cancel.hidden = !linking;
  }
  function inspector() {
    const el = document.querySelector(".graph-inspector");
    if (!el) return;
    const n = doc.nodes.find((n) => n.id === selected);
    if (!n) {
      el.innerHTML = `<h3>节点设置</h3><p>选择节点以配置模型和输入.</p><div class="graph-mini-guide"><span>1. 添加节点</span><span>2. 连接输入与输出</span><span>3. 保存并运行</span></div>`;
      return;
    }
    const hasPlan = doc.edges.some((e) => e.to === n.id && e.port === "plan"),
      result = runMatches() ? run?.nodes?.[n.id] : null;
    let fields = "";
    if (n.type === "text")
      fields += field(
        "提示词",
        `<textarea name="text" rows="8">${esc(n.text || "")}</textarea>`,
      );
    if (n.type === "image")
      fields += `<div class="node-image-picker">${n.asset_id ? `<img data-asset="${n.asset_id}" alt="参考图片">` : ""}${button(n.asset_id ? "更换图片" : "选择图片", "flow-image", "gallery")}</div>`;
    if (n.type === "person")
      fields +=
        state.characters
          .map(
            (c) =>
              `<label class="node-person"><input type="checkbox" name="character" value="${c.id}" ${n.characters?.includes(c.id) ? "checked" : ""}>${esc(c.name)}</label>`,
          )
          .join("") || "<p>先在形象库添加人物.</p>";
    if (n.type === "plan")
      fields +=
        field(
          "拍摄方式",
          `<select name="workflow">${[
            ["variants", "成片变体"],
            ["recreate", "严格仿拍"],
            ["outfit", "换装拍摄"],
          ]
            .map(([id, label]) => option(id, label, n.workflow))
            .join("")}</select>`,
        ) +
        field(
          "规划模型",
          `<select name="planner">${option("", "选择视觉对话模型", n.planner)}${state.planners.map((p) => option(p.id, p.model + " · " + p.id, n.planner)).join("")}</select>`,
        ) +
        field(
          "参考对应人物",
          `<select name="target_character">${option("", "单人自动绑定", n.target_character)}${state.characters.map((c) => option(c.id, c.name, n.target_character)).join("")}</select>`,
        );
    if (n.type === "generate")
      fields += field(
        "生成服务商",
        `<select name="provider">${option("", "按回退链路", n.provider)}${state.config.providers
          .filter((p) => !p.__template_key.includes("video"))
          .map((p) => option(p.id, p.label || p.id, n.provider))
          .join("")}</select>`,
      );
    if (n.type === "plan" || (n.type === "generate" && !hasPlan))
      fields +=
        field(
          "图片数量",
          input("count", n.count || 1, "number", 'min="1" max="12"'),
        ) +
        field(
          "规格",
          `<select name="output">${["3:4 4K", "1:1 4K", "16:9 4K", "9:16 4K", "3:4 2K", "1024x1024"].map((x) => option(x, x, n.output)).join("")}</select>`,
        );
    else if (hasPlan)
      fields += '<p class="field-help">数量、人物与画幅使用上游镜头规划.</p>';
    el.innerHTML = `<header><div><small>步骤 ${doc.nodes.indexOf(n) + 1}</small><h3>${esc(n.label || names[n.type])}</h3></div>${ib("移除节点", "flow-remove", "trash")}</header><div class="node-fields">${fields}</div><div class="node-connections"><h4>输入来源</h4>${
      Object.entries(state.nodeTypes[n.type]?.inputs || {})
        .filter(([port, type]) => {
          const connected = doc.edges.some(
            (e) => e.to === n.id && e.port === port,
          );
          if (hasPlan && port !== "plan" && !connected) return false;
          return (
            connected ||
            port === "text" ||
            n.type === "output" ||
            doc.nodes.some(
              (s) => s.id !== n.id && state.nodeTypes[s.type]?.output === type,
            )
          );
        })
        .map(([port, type]) =>
          field(
            ports[port],
            `<select data-connect-port="${port}">${option("", "未连接", "")}${doc.nodes
              .filter(
                (s) =>
                  s.id !== n.id && state.nodeTypes[s.type]?.output === type,
              )
              .map((s) =>
                option(
                  s.id,
                  s.label || names[s.type],
                  doc.edges.find((e) => e.to === n.id && e.port === port)?.from,
                ),
              )
              .join("")}</select>`,
          ),
        )
        .join("") || ""
    }</div><details class="node-naming"><summary>节点名称</summary><div class="node-fields">${field("名称", input("label", n.label || names[n.type]))}</div></details>${result?.error ? `<p class="node-error">${esc(result.error)}</p>` : ""}${result?.assets?.length ? `<div class="node-results">${result.assets.map((id) => `<button data-action="view-asset" data-id="${id}"><img data-asset="${id}" alt="节点结果"></button>`).join("")}</div>` : ""}<div class="node-savebar"><small class="node-save-status">${dirty ? "有未保存修改" : "已保存"}</small>${button("保存", "flow-save", "check")}${button(doc.nodes.indexOf(n) === doc.nodes.length - 1 ? "回到第一步" : "下一步", "flow-next", "arrow")}</div>`;
    el.querySelectorAll(
      ".node-fields input,.node-fields select,.node-fields textarea",
    ).forEach(
      (input) =>
        (input.oninput = () => {
          if (input.name === "character")
            n.characters = [
              ...el.querySelectorAll('[name="character"]:checked'),
            ].map((x) => x.value);
          else
            n[input.name] =
              input.name === "count" ? Number(input.value) : input.value;
          mark();
          paint();
        }),
    );
    el.querySelectorAll("[data-connect-port]").forEach(
      (input) =>
        (input.onchange = () =>
          connect(input.value, n.id, input.dataset.connectPort)),
    );
    loadImages(el);
  }
  function connect(from, to, port) {
    const previous = doc.edges;
    doc.edges = doc.edges.filter((e) => !(e.to === to && e.port === port));
    if (from) {
      const walk = (id, seen = new Set()) => {
        if (id === from) return true;
        if (seen.has(id)) return false;
        seen.add(id);
        return doc.edges
          .filter((e) => e.from === id)
          .some((e) => walk(e.to, seen));
      };
      if (from === to || walk(to)) {
        doc.edges = previous;
        toast("这条连线会形成循环");
        return;
      }
      doc.edges.push({ from, to, port });
    }
    linking = "";
    mark();
    paint();
    inspector();
  }
  function summary() {
    const el = document.querySelector(".graph-run-summary");
    if (!el) return;
    const runs = [...(state.graphRuns || [])]
      .filter((r) => r.graph_id === doc.id)
      .reverse();
    el.innerHTML = `<div><h3>运行记录</h3><select id="graph-run-select" aria-label="查看运行记录">${!run ? '<option value="">尚未运行</option>' : ""}${runs.map((r) => option(r.id, `${time(r.created)} · ${statusNames[r.state] || r.state}`, run?.id)).join("")}</select></div>${run ? `<p>${statusNames[run.state]} · ${Object.values(run.nodes).filter((n) => n.state === "completed").length} / ${Object.keys(run.nodes).length} 个节点${!runMatches() ? " · 当前配置已有修改" : ""}</p><div class="run-images">${(run.assets || []).map((id) => `<button data-action="view-asset" data-id="${id}"><img data-asset="${id}" alt="本次运行结果"></button>`).join("")}</div>` : "<p>运行后逐节点显示状态与结果.</p>"}`;
    loadImages(el);
    el.querySelector("select").onchange = async (e) => {
      if (e.target.value) {
        run = await query("graph-run", { id: e.target.value });
        paint();
        inspector();
        summary();
      }
    };
  }
  function bind() {
    const stage = document.querySelector(".graph-stage");
    document.querySelector("#graph-select").onchange = async (e) => {
      try {
        if (dirty) await save();
        doc = structuredClone(
          state.graphs.find((g) => g.id === e.target.value),
        );
        selected = "";
        run =
          [...state.graphRuns].reverse().find((r) => r.graph_id === doc.id) ||
          null;
        runKey = "";
        autoView = true;
        render();
      } catch (error) {
        toast(error.message);
      }
    };
    document
      .querySelectorAll("[draggable=true]")
      .forEach(
        (b) =>
          (b.ondragstart = (e) =>
            e.dataTransfer.setData("text/plain", b.dataset.type)),
      );
    stage.ondragover = (e) => e.preventDefault();
    stage.ondrop = (e) => {
      e.preventDefault();
      const type = e.dataTransfer.getData("text/plain"),
        r = stage.getBoundingClientRect();
      if (names[type])
        add(
          type,
          (e.clientX - r.left - pan.x) / zoom,
          (e.clientY - r.top - pan.y) / zoom,
        );
    };
    stage.onpointerdown = (e) => {
      if (e.target.closest("button,input,textarea,select")) return;
      const id = e.target.closest("[data-node]")?.dataset.node;
      if (id) {
        selected = id;
        inspector();
        paint();
        steps();
      }
      const n = doc.nodes.find((n) => n.id === id);
      if (id && !e.target.closest("[data-drag-node]")) return;
      activeDrag = {
        id,
        x: e.clientX,
        y: e.clientY,
        origin: n ? { x: n.x, y: n.y } : { ...pan },
      };
      stage.setPointerCapture(e.pointerId);
    };
    stage.onpointermove = (e) => {
      if (!activeDrag) return;
      const dx = e.clientX - activeDrag.x,
        dy = e.clientY - activeDrag.y;
      if (activeDrag.id) {
        const n = doc.nodes.find((n) => n.id === activeDrag.id);
        n.x = activeDrag.origin.x + dx / zoom;
        n.y = activeDrag.origin.y + dy / zoom;
        mark();
      } else pan = { x: activeDrag.origin.x + dx, y: activeDrag.origin.y + dy };
      paint();
    };
    stage.onpointerup = () => {
      activeDrag = null;
      paint();
      steps();
    };
    stage.onpointercancel = stage.onpointerup;
    stage.onwheel = (e) => {
      e.preventDefault();
      const rect = stage.getBoundingClientRect(),
        x = e.clientX - rect.left,
        y = e.clientY - rect.top,
        next = Math.min(1.6, Math.max(0.2, zoom * Math.exp(-e.deltaY * 0.001)));
      pan = {
        x: x - ((x - pan.x) * next) / zoom,
        y: y - ((y - pan.y) * next) / zoom,
      };
      zoom = next;
      paint();
    };
  }
  function add(type, x = 60, y = 60) {
    const n = node(type, x, y);
    doc.nodes.push(n);
    selected = n.id;
    mark();
    render();
  }
  function fit() {
    const stage = document.querySelector(".graph-stage");
    if (!stage || !doc.nodes.length) return;
    const minX = Math.min(...doc.nodes.map((n) => n.x)),
      minY = Math.min(...doc.nodes.map((n) => n.y)),
      width = Math.max(...doc.nodes.map((n) => n.x + 240)) - minX,
      height = Math.max(...doc.nodes.map((n) => n.y + 250)) - minY;
    zoom = Math.max(
      0.2,
      Math.min(
        1,
        (stage.clientWidth - 50) / width,
        (stage.clientHeight - 50) / height,
      ),
    );
    pan = { x: 25 - minX * zoom, y: 25 - minY * zoom };
    paint();
  }
  async function save() {
    if (saving) throw Error("正在保存, 请稍候");
    saving = true;
    const current = doc,
      snapshot = structuredClone(doc);
    try {
      const saved = await api("graph", snapshot);
      const unchanged = JSON.stringify(current) === JSON.stringify(snapshot);
      if (doc === current) {
        if (unchanged) doc = structuredClone(saved);
        else {
          doc.id = saved.id;
          doc.revision = saved.revision;
        }
        dirty = !unchanged;
      }
      const i = state.graphs.findIndex((g) => g.id === saved.id);
      if (i < 0) state.graphs.push(structuredClone(saved));
      else state.graphs[i] = structuredClone(saved);
      return doc;
    } finally {
      saving = false;
    }
  }
  async function action(action, el) {
    if (!action.startsWith("flow-")) return false;
    if (action === "flow-new") {
      if (!el.dataset.template) {
        dialog(
          "选择起点",
          `${button("自由生图", "flow-new", "plus", 'data-template="simple"')}${button("成片变体", "flow-new", "copy", 'data-template="variants"')}${button("人物仿拍", "flow-new", "people", 'data-template="recreate"')}`,
        );
        return true;
      }
      if (dirty) await save();
      close();
      doc = template(el.dataset.template || "simple");
      selected = "";
      run = null;
      mark();
      autoView = true;
      await save();
      render();
    }
    if (action === "flow-add")
      add(
        el.dataset.type,
        80 + doc.nodes.length * 18,
        80 + doc.nodes.length * 16,
      );
    if (action === "flow-select") {
      selectNode(el.dataset.id, true);
    }
    if (action === "flow-next")
      selectNode(
        doc.nodes[
          (doc.nodes.findIndex((n) => n.id === selected) + 1) % doc.nodes.length
        ].id,
        true,
      );
    if (action === "flow-link") {
      linking = el.dataset.node;
      paint();
      document.querySelector(".graph-hint").textContent =
        "点击兼容的输入圆点完成连线";
    }
    if (action === "flow-connect") {
      if (!linking) {
        selected = el.dataset.node;
        inspector();
        return true;
      }
      const source = doc.nodes.find((n) => n.id === linking),
        dest = doc.nodes.find((n) => n.id === el.dataset.node);
      if (
        state.nodeTypes[source.type].output !==
        state.nodeTypes[dest.type].inputs[el.dataset.port]
      )
        throw Error("请选择相同类型的输入接口");
      connect(linking, el.dataset.node, el.dataset.port);
    }
    if (action === "flow-unlink") {
      linking = "";
      paint();
    }
    if (action === "flow-remove") {
      doc.nodes = doc.nodes.filter((n) => n.id !== selected);
      doc.edges = doc.edges.filter(
        (e) => e.from !== selected && e.to !== selected,
      );
      selected = "";
      mark();
      render();
    }
    if (action === "flow-image")
      await picker((a) => {
        doc.nodes.find((n) => n.id === selected).asset_id = a.id;
        close();
        mark();
        paint();
        inspector();
      }, "选择工作流参考");
    if (action === "flow-save") {
      await save();
      render();
      toast("工作流已保存");
    }
    if (action === "flow-rename")
      dialog(
        "工作流名称",
        input("graph-name", doc.name),
        button("保存", "flow-name-save", "check"),
      );
    if (action === "flow-name-save") {
      doc.name = document.querySelector('[name="graph-name"]').value;
      mark();
      await save();
      close();
      render();
    }
    if (action === "flow-run") {
      const pending = doc.nodes.filter(issue);
      if (pending.length) {
        selectNode(pending[0].id, true);
        toast(
          `${pending[0].label || names[pending[0].type]}: ${issue(pending[0])}`,
        );
        return true;
      }
      if (dirty) await save();
      if (dirty) throw Error("内容刚刚有改动, 请保存后再运行");
      dialog(
        "运行工作流",
        `<p>执行 ${doc.nodes.length} 个节点. 规划完成后会自动进入生成节点, 使用各节点选择的模型.</p><p class="field-help">开始后可以离开页面, 刷新不会重复执行.</p>`,
        button("开始运行", "flow-run-confirm", "play", 'class="primary"'),
      );
    }
    if (action === "flow-run-confirm") {
      el.disabled = true;
      runKey = runKey || uid();
      try {
        doc.pending_run = runKey;
        await save();
        run = await api("graph-run", {
          graph_id: doc.id,
          revision: doc.revision,
          request_id: runKey,
        });
        state.graphRuns.push(run);
        close();
        render();
      } finally {
        el.disabled = false;
      }
    }
    if (action === "flow-delete")
      dialog(
        "删除工作流",
        `<p>删除 ${esc(doc.name)}? 运行记录和生成图片会保留.</p>`,
        button("确认删除", "flow-delete-confirm", "trash"),
      );
    if (action === "flow-delete-confirm") {
      await api("graph", { id: doc.id, revision: doc.revision, delete: true });
      state.graphs = state.graphs.filter((g) => g.id !== doc.id);
      doc = structuredClone(state.graphs[0] || null);
      run = null;
      dirty = false;
      selected = "";
      close();
      render();
    }
    if (action === "flow-stop") {
      run = await api("graph-cancel", { id: run.id });
      render();
    }
    if (action === "flow-fit") fit();
    if (action === "flow-zoom-in" || action === "flow-zoom-out") {
      zoom = Math.max(
        0.2,
        Math.min(1.6, zoom + (action === "flow-zoom-in" ? 0.1 : -0.1)),
      );
      paint();
    }
    if (action === "flow-export") {
      const blob = new Blob(
          [
            JSON.stringify(
              { name: doc.name, nodes: doc.nodes, edges: doc.edges },
              null,
              2,
            ),
          ],
          { type: "application/json" },
        ),
        url = URL.createObjectURL(blob),
        a = document.createElement("a");
      a.href = url;
      a.download = doc.name + ".json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    if (action === "flow-import") {
      if (dirty) await save();
      const file = document.createElement("input");
      file.type = "file";
      file.accept = "application/json";
      file.onchange = async () => {
        try {
          if (file.files[0].size > 1024 * 1024) throw Error("工作流文件过大");
          const data = JSON.parse(await file.files[0].text());
          const saved = await api("graph", {
            name: data.name,
            nodes: data.nodes,
            edges: data.edges,
          });
          state.graphs.push(saved);
          doc = structuredClone(saved);
          dirty = false;
          selected = "";
          run = null;
          autoView = true;
          render();
        } catch (e) {
          toast(e.message);
        }
      };
      file.click();
    }
    return true;
  }
  async function poll() {
    if (run?.state === "running") {
      const next = await query("graph-run", { id: run.id });
      if (JSON.stringify(next) !== JSON.stringify(run)) {
        run = next;
        const i = state.graphRuns.findIndex((r) => r.id === run.id);
        if (i >= 0) state.graphRuns[i] = run;
        if (state.page === "flows" && !activeDrag) {
          paint();
          if (!document.querySelector(".graph-inspector :focus")) inspector();
          summary();
          if (run.state !== "running") {
            runKey = "";
            render();
          }
        }
      }
    }
  }
  return {
    render: renderGraph,
    action,
    restore,
    poll,
    get dirty() {
      return dirty;
    },
  };
}
