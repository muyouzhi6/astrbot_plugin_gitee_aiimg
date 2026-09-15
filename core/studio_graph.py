"""Persisted typed node graphs executed through the existing studio queues."""

from __future__ import annotations

import asyncio
import copy
import math
import hashlib
import re
import time

from .studio_workflows import WORKFLOWS, image_count, request_key
from .studio_store import StudioConflict
from .studio_characters import resolve_characters


NODE_TYPES = {
    "text": {"label": "提示词", "inputs": {}, "output": "text"},
    "image": {"label": "参考图片", "inputs": {}, "output": "images"},
    "person": {"label": "出镜人物", "inputs": {}, "output": "people"},
    "plan": {
        "label": "镜头规划",
        "inputs": {"text": "text", "image": "images", "people": "people"},
        "output": "plan",
    },
    "generate": {
        "label": "生成图片",
        "inputs": {
            "text": "text",
            "image": "images",
            "people": "people",
            "plan": "plan",
        },
        "output": "images",
    },
    "output": {"label": "结果输出", "inputs": {"image": "images"}, "output": None},
}
TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted", "expired"}


def validate_graph(graph, *, executable=False):
    graph = copy.deepcopy(graph)
    if (
        not isinstance(graph.get("name"), str)
        or not 1 <= len(graph["name"].strip()) <= 80
    ):
        raise ValueError("请输入 1 至 80 字的工作流名称")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if (
        not isinstance(nodes, list)
        or not 0 <= len(nodes) <= 30
        or not isinstance(edges, list)
        or len(edges) > 60
    ):
        raise ValueError("工作流需有 1 至 30 个节点, 最多 60 条连线")
    by_id = {}
    for n in nodes:
        if (
            not isinstance(n, dict)
            or n.get("type") not in NODE_TYPES
            or not isinstance(n.get("id"), str)
            or not re.fullmatch(r"[\w-]{1,60}", n["id"])
            or n["id"] in by_id
        ):
            raise ValueError("节点类型或标识无效")
        for coordinate in ("x", "y"):
            value = n.get(coordinate, 0)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or abs(value) > 20000
            ):
                raise ValueError("节点位置无效")
        if n["type"] in {"plan", "generate"}:
            image_count(n.get("count", 1))
        if n["type"] == "plan" and n.get("workflow", "variants") not in WORKFLOWS:
            raise ValueError("规划节点模式无效")
        if len(str(n.get("text", ""))) > 18000:
            raise ValueError("节点提示词过长")
        by_id[n["id"]] = n
        if n["type"] == "text" and not isinstance(n.get("text", ""), str):
            raise ValueError("提示词必须为文字")
        if n["type"] == "person" and (
            not isinstance(n.get("characters", []), list)
            or len(n.get("characters", [])) > 4
            or any(not isinstance(x, str) for x in n.get("characters", []))
        ):
            raise ValueError("人物节点格式无效")
    incoming = {key: {} for key in by_id}
    outgoing = {key: [] for key in by_id}
    for e in edges:
        if (
            not isinstance(e, dict)
            or e.get("from") not in by_id
            or e.get("to") not in by_id
            or e["from"] == e["to"]
        ):
            raise ValueError("连线引用了不存在的节点")
        source, target = by_id[e["from"]], by_id[e["to"]]
        port = e.get("port")
        if (
            not NODE_TYPES[source["type"]]["output"]
            or NODE_TYPES[target["type"]]["inputs"].get(port)
            != NODE_TYPES[source["type"]]["output"]
        ):
            raise ValueError("连线类型不匹配")
        if port in incoming[target["id"]]:
            raise ValueError("每个输入接口只能连接一个节点")
        incoming[target["id"]][port] = source["id"]
        outgoing[source["id"]].append(target["id"])
    indegree = {key: len(set(ports.values())) for key, ports in incoming.items()}
    order = []
    while len(order) < len(nodes):
        ready = [key for key in by_id if key not in order and indegree[key] == 0]
        if not ready:
            raise ValueError("工作流不能形成循环连线")
        for key in ready:
            order.append(key)
            for dest in set(outgoing[key]):
                indegree[dest] -= 1
    if executable:
        outputs = [n["id"] for n in nodes if n["type"] == "output"]
        if not outputs:
            raise ValueError("请添加并连接结果输出节点")
        used = set()

        def visit(key):
            if key in used:
                return
            used.add(key)
            for parent in incoming[key].values():
                visit(parent)

        for key in outputs:
            visit(key)
        if len(used) != len(nodes):
            raise ValueError("有节点未连接到结果输出, 请连接或移除")
        if not any(n["type"] == "generate" for n in nodes):
            raise ValueError("请添加生成图片节点")
        for n in nodes:
            ports = incoming[n["id"]]
            if n["type"] == "text" and not str(n.get("text", "")).strip():
                raise ValueError("请填写提示词节点")
            if n["type"] == "image" and not n.get("asset_id"):
                raise ValueError("请为参考图片节点选图")
            if n["type"] == "person" and not n.get("characters"):
                raise ValueError("请选择出镜人物")
            if n["type"] == "plan":
                if "image" not in ports or not n.get("planner"):
                    raise ValueError("镜头规划节点需要参考图片和规划模型")
                if (
                    n.get("workflow") in {"recreate", "outfit"}
                    and "people" not in ports
                ):
                    raise ValueError("仿拍与换装需要连接出镜人物")
            if n["type"] == "generate":
                if "plan" in ports and len(ports) > 1:
                    raise ValueError("规划结果已包含参考和人物, 生成节点仅连接镜头输入")
                if "plan" not in ports and "text" not in ports:
                    raise ValueError("生成节点需要提示词或镜头规划")
            if n["type"] == "output" and "image" not in ports:
                raise ValueError("结果输出需要连接图片")
        if (
            sum(n.get("count", 1) for n in nodes if n["type"] in {"generate", "plan"})
            > 60
        ):
            raise ValueError("单次工作流的计划图片数量过多, 请拆分为多个工作流")
    return graph, order, incoming


class StudioGraph:
    def __init__(self, studio):
        self.studio = studio
        self.tasks = {}

    def recover(self):
        for run in self.studio.store.documents("graph_run"):
            if run["state"] == "running":
                run["state"] = "interrupted"
                for n in run["nodes"].values():
                    if n["state"] in {"running", "pending"}:
                        n["state"] = "interrupted"
                self.studio.store.save_document("graph_run", run)

    def save(self, body):
        graph, _, _ = validate_graph(body)
        for n in graph["nodes"]:
            if n["type"] == "image" and n.get("asset_id"):
                self.studio.store.asset_path(n["asset_id"])
        return self.studio.store.save_document("graph", graph)

    def public_run(self, run):
        if not run:
            return None
        result = copy.deepcopy(run)
        result["assets"] = [
            key for key in result.get("assets", []) if self.studio.store.available(key)
        ]
        for node in result["nodes"].values():
            if "assets" in node:
                node["assets"] = [
                    key for key in node["assets"] if self.studio.store.available(key)
                ]
        return result

    async def start(self, body):
        key = request_key(body.get("request_id"))
        store = self.studio.store
        prior = store.document("graph_run", key)
        if prior:
            return prior
        if not self.studio.manager or not self.studio.manager.accepting:
            raise ValueError("任务系统尚未就绪")
        if len(self.tasks) >= 2:
            raise ValueError("已有两个工作流执行中, 请稍候")
        graph = store.document("graph", body.get("graph_id", ""))
        if not graph or graph["revision"] != body.get("revision"):
            raise StudioConflict("工作流已变化, 请保存或刷新后执行")
        graph, order, ports = validate_graph(graph, executable=True)
        for n in graph["nodes"]:
            if n["type"] == "image":
                store.asset_path(n["asset_id"])
            if n["type"] == "person":
                for cid in n["characters"]:
                    if not store.document("character", cid):
                        raise ValueError("出镜人物已不存在")
            if n["type"] == "plan" and n["planner"] not in {
                p["id"] for p in self.studio.workflows.providers()
            }:
                raise ValueError("规划模型已不存在")
            if (
                n["type"] == "generate"
                and n.get("provider")
                and n["provider"] not in self.studio.plugin.registry.provider_ids()
            ):
                raise ValueError("生成服务商已不存在")
        portraits = {}
        for n in graph["nodes"]:
            if n["type"] == "person":
                life = await self.studio.plugin._get_life_context_without_llm()
                portraits[n["id"]] = await asyncio.to_thread(
                    resolve_characters,
                    store,
                    n["characters"],
                    n.get("outfits", [""] * len(n["characters"])),
                    sender="",
                    scope="studio",
                    bot_id="",
                    life_context=life,
                    administrator=True,
                )
        run = store.save_document(
            "graph_run",
            {
                "id": key,
                "graph_id": graph["id"],
                "graph": graph,
                "state": "running",
                "created": time.time(),
                "nodes": {n["id"]: {"state": "pending"} for n in graph["nodes"]},
                "assets": [],
            },
        )
        task = asyncio.create_task(self.execute(key, graph, order, ports, portraits))
        self.tasks[key] = task
        task.add_done_callback(lambda _: self.tasks.pop(key, None))
        return run

    def update(self, key, node=None, **fields):
        run = self.studio.store.document("graph_run", key)
        if node:
            run["nodes"][node].update(fields)
        else:
            run.update(fields)
        return self.studio.store.save_document("graph_run", run)

    async def execute(self, key, graph, order, ports, portraits):
        values = {}
        fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        current = None
        try:
            for index, nid in enumerate(order):
                current = nid
                node = next(n for n in graph["nodes"] if n["id"] == nid)
                self.update(key, nid, state="running")
                inputs = {port: values[parent] for port, parent in ports[nid].items()}
                kind = node["type"]
                if kind == "text":
                    value = node["text"]
                elif kind == "image":
                    value = [node["asset_id"]]
                elif kind == "person":
                    value = {
                        "characters": node["characters"],
                        "outfits": node.get("outfits", [""] * len(node["characters"])),
                        "portrait": portraits[nid],
                    }
                elif kind == "plan":
                    if len(inputs["image"]) != 1:
                        raise ValueError("镜头规划一次需要一张母片, 请连接单图节点")
                    payload = {
                        "request_id": f"{fingerprint}-plan-{index}",
                        "workflow": node.get("workflow", "variants"),
                        "source_asset": inputs["image"][0],
                        "characters": inputs.get("people", {}).get("characters", []),
                        "target_character": node.get("target_character", ""),
                        "prompt": inputs.get("text", ""),
                        "planner": node["planner"],
                        "count": node.get("count", 4),
                        "output": node.get("output", "3:4 4K"),
                        "workspace_id": "",
                    }
                    async with self.studio.lock:
                        plan = await self.studio.workflows.start(payload)
                    self.update(key, nid, plan_id=plan["id"])
                    while plan["state"] == "planning":
                        await asyncio.sleep(0.5)
                        plan = self.studio.store.document("plan", plan["id"])
                    if plan["state"] != "ready":
                        raise ValueError(plan.get("error") or "镜头规划未完成")
                    value = {
                        **payload,
                        "plan_id": plan["id"],
                        "shots": plan["shots"],
                        "outfits": inputs.get("people", {}).get("outfits", []),
                        "portrait": inputs.get("people", {}).get("portrait"),
                    }
                elif kind == "generate":
                    if "plan" in inputs:
                        payload = dict(inputs["plan"])
                    else:
                        payload = {
                            "workflow": "generate",
                            "prompt": inputs["text"],
                            "assets": inputs.get("image", []),
                            **inputs.get("people", {}),
                            "count": node.get("count", 1),
                            "output": node.get("output", "3:4 4K"),
                        }
                    payload.update(
                        provider=node.get("provider", ""),
                        workspace_id="",
                        request_id=f"{fingerprint}-image-{index}",
                    )
                    portrait = payload.pop("portrait", None)
                    async with self.studio.lock:
                        task = await self.studio.submit(payload, portrait=portrait)
                    self.update(key, nid, task_id=task["task_id"])
                    while True:
                        row = await self.studio.manager.get_task(task["task_id"])
                        if not row or row["state"] in TERMINAL:
                            break
                        await asyncio.sleep(0.5)
                    value = [
                        i["asset_id"]
                        for i in (row or {}).get("items", [])
                        if i.get("asset_id")
                    ]
                    self.update(key, nid, assets=value)
                    if not row or row["state"] != "completed":
                        raise ValueError(
                            "生成节点未全部成功, 已保留完成图片, 后续节点停止"
                        )
                else:
                    value = inputs["image"]
                    self.update(
                        key,
                        assets=list(
                            dict.fromkeys(
                                self.studio.store.document("graph_run", key)["assets"]
                                + value
                            )
                        ),
                    )
                values[nid] = value
                self.update(
                    key,
                    nid,
                    state="completed",
                    **(
                        {"assets": value}
                        if kind in {"image", "generate", "output"}
                        else {}
                    ),
                )
            self.update(key, state="completed")
        except asyncio.CancelledError:
            await self.stop_children(key)
            if current:
                self.update(key, current, state="cancelled")
            self.update(key, state="cancelled")
        except Exception as exc:
            from .background_tasks import BackgroundImageTaskManager

            if current:
                self.update(
                    key,
                    current,
                    state="failed",
                    error=BackgroundImageTaskManager.sanitize_error(exc),
                )
            self.update(key, state="failed")
        finally:
            run = self.studio.store.document("graph_run", key)
            for node in run["nodes"].values():
                if node["state"] == "pending":
                    node["state"] = "skipped"
            self.studio.store.save_document("graph_run", run)

    async def stop_children(self, key):
        run = self.studio.store.document("graph_run", key)
        for node in run["nodes"].values():
            if node.get("task_id"):
                await self.studio.manager.cancel_task(
                    node["task_id"], "工作流停止", suppress_future_injection=True
                )
            task = self.studio.workflows.tasks.get(node.get("plan_id"))
            if task:
                task.cancel()

    async def cancel(self, key):
        task = self.tasks.get(key)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.studio.store.document("graph_run", key)

    async def close(self):
        for key in list(self.tasks):
            await self.cancel(key)
