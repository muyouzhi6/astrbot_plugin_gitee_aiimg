import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import vm from "node:vm";
import { ready, uploadReference } from "../pages/studio/api.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const python = spawnSync(process.env.PYTHON || resolve(root, ".venv/bin/python"), [
  "-c",
  "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('astrbot').origin).parent / 'dashboard')",
], { encoding: "utf8" });
assert.equal(python.status, 0, "Install AstrBot to test its actual page bridge");
const dashboard = python.stdout.trim();
const assets = resolve(dashboard, "dist/assets");
const bundle = readdirSync(assets).find((name) => /^Plugin(?:Page|View)Page-.*\.js$/.test(name));
assert.ok(bundle, "Locate installed AstrBot page host");
const normalizer = readFileSync(resolve(assets, bundle), "utf8")
  .match(/\w+=(\w+=>\{if\(typeof \w+!="string"\)throw new Error\("Plugin bridge endpoint must be a string\."\);.*?\.join\("\/"\)\})/s);
assert.ok(normalizer, "Locate the actual host endpoint validator");
const normalize = vm.runInNewContext(`(${normalizer[1]})`);

test("native page host rejects invalid plugin endpoints", () => {
  for (const endpoint of ["", "../escape", "studio/./state", "https://example.com", "studio/state?secret=1"]) {
    assert.throws(() => normalize(endpoint));
  }
  assert.equal(normalize("studio/reference-upload/asset_test"), "studio/reference-upload/asset_test");
});

for (const id of ["", "asset_test"]) {
  test(`native page bridge uploads ${id ? "replacement" : "new reference"}`, async (t) => {
    const previous = globalThis.window;
    t.after(() => { globalThis.window = previous; });
    let receive, sent;
    const origin = "http://localhost";
    const parent = {
      postMessage(message) {
        if (message.kind !== "request") return;
        try {
          normalize(message.endpoint);
          sent = message;
          receive({ source: parent, origin, data: {
            channel: "astrbot-plugin-page", kind: "response",
            requestId: message.requestId, ok: true,
            data: { ok: true, data: { id: "updated" } },
          } });
        } catch (error) {
          receive({ source: parent, origin, data: {
            channel: "astrbot-plugin-page", kind: "response",
            requestId: message.requestId, ok: false, error: error.message,
          } });
        }
      },
    };
    const window = { parent, location: { origin }, addEventListener(_, handler) { receive = handler; } };
    vm.runInNewContext(readFileSync(resolve(dashboard, "plugin_page_bridge.js"), "utf8"), { window });
    window.AstrBotPluginPage.__setInitialContext({ locale: "zh-CN" });
    globalThis.window = window;
    await ready();
    const file = new Blob(["reference fixture"], { type: "image/png" });
    assert.deepEqual(await uploadReference(file, id), { id: "updated" });
    assert.equal(sent.endpoint, "studio/reference-upload" + (id ? "/" + id : ""));
    assert.equal(sent.action, "files:upload");
    assert.equal(Buffer.from(sent.fileBuffer).toString(), "reference fixture");
  });
}
