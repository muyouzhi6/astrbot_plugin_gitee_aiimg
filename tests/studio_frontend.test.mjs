import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import test from "node:test";
import { uid } from "../pages/studio/id.js";
import { Canvas } from "../pages/studio/canvas.js";
import { api, query, ready } from "../pages/studio/api.js";
import { createGraphEditor } from "../pages/studio/graph.js";
import { createWorkspace } from "../pages/studio/workspace.js";

test("connection failures stay actionable and never replay image requests", async (t) => {
  const previous = globalThis.window;
  t.after(() => {
    globalThis.window = previous;
  });
  let calls = 0;
  globalThis.window = {
    AstrBotPluginPage: {
      ready: async () => ({}),
      apiPost: async () => {
        calls++;
        throw Error("Request failed with status code 503");
      },
      apiGet: async () => {
        throw Error("Network Error");
      },
    },
  };
  await ready();
  await assert.rejects(api("generate", {}), /HTTP 503.*任务页/);
  assert.equal(calls, 1);
  await assert.rejects(query("state", {}), /连接中断/);
});

test("planning from free canvas opens the visible review and result view", async (t) => {
  const previous = globalThis.window;
  t.after(() => {
    globalThis.window = previous;
  });
  globalThis.window = {
    AstrBotPluginPage: {
      ready: async () => ({}),
      apiPost: async () => ({
        ok: true,
        data: { id: "plan", state: "planning" },
      }),
    },
  };
  await ready();
  const doc = {
    layers: [],
    shoot: {
      view: "canvas",
      mode: "variants",
      source: "photo",
      planner: "vision",
      count: 2,
    },
  };
  const state = { prompt: "a cup", plans: [] };
  const workspace = createWorkspace({
    state,
    getCanvas: () => ({ doc }),
    render() {},
    shell() {},
    changed() {},
    saveCanvas: async () => {},
    picker() {},
    viewAsset() {},
  });
  workspace.restore(doc);
  await workspace.action("shoot-plan", {});
  assert.equal(workspace.capture().view, "board");
  assert.equal(workspace.capture().plan_id, "plan");
});

test("HTTP dashboard IDs support creation, canvas copy, undo and redo", async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", {
    configurable: true,
    value: { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) },
  });
  t.after(() => Object.defineProperty(globalThis, "crypto", descriptor));
  const ids = Array.from({ length: 1000 }, uid);
  assert.equal(new Set(ids).size, 1000);
  assert.ok(
    ids.every((id) =>
      /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(
        id,
      ),
    ),
  );
  const canvas = new Canvas({ layers: [] }, () => {});
  await canvas.add({ id: "photo", width: 800, height: 1200 });
  canvas.mutate("copy");
  assert.equal(canvas.doc.layers.length, 2);
  assert.notEqual(canvas.doc.layers[0].id, canvas.doc.layers[1].id);
  assert.equal(canvas.doc.layers[1].asset_id, "photo");
  canvas.undo();
  assert.equal(canvas.doc.layers.length, 1);
  canvas.undo(true);
  assert.equal(canvas.doc.layers.length, 2);
});

test("workflow saving preserves edits made while the server responds", async (t) => {
  const oldDocument = globalThis.document,
    oldWindow = globalThis.window;
  const toast = { classList: { add() {}, remove() {} } };
  globalThis.document = {
    querySelector: (selector) => (selector === "#toast" ? toast : null),
  };
  t.mock.method(globalThis, "setTimeout", () => 0);
  t.after(() => {
    globalThis.document = oldDocument;
    globalThis.window = oldWindow;
  });
  let finish, submitted;
  globalThis.window = {
    AstrBotPluginPage: {
      ready: async () => ({}),
      apiPost: async (_, body) => {
        submitted = structuredClone(body);
        return new Promise((resolve) => {
          finish = () =>
            resolve({
              ok: true,
              data: { ...submitted, revision: submitted.revision + 1 },
            });
        });
      },
    },
  };
  await ready();
  const original = {
    id: "graph",
    name: "Draft",
    revision: 1,
    nodes: [{ id: "text", type: "text", text: "a cup", x: 0, y: 0 }],
    edges: [],
  };
  const state = { graphs: [structuredClone(original)], graphRuns: [] };
  const editor = createGraphEditor({
    state,
    render() {},
    picker() {},
    viewAsset() {},
  });
  editor.restore();
  const saving = editor.action("flow-save", {});
  await editor.action("flow-add", { dataset: { type: "image" } });
  finish();
  await saving;
  assert.equal(editor.dirty, true);
  assert.equal(state.graphs[0].nodes.length, 1);
  const next = editor.action("flow-save", {});
  assert.equal(submitted.nodes.length, 2);
  assert.equal(submitted.revision, 2);
  finish();
  await next;
  assert.equal(editor.dirty, false);
  assert.equal(state.graphs[0].nodes.length, 2);
  await editor.action("flow-add", { dataset: { type: "text" } });
  assert.equal(state.graphs[0].nodes.length, 2);
});
