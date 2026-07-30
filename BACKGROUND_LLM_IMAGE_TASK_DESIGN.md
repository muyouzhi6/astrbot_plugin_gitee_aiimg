# LLM 生图后台任务设计与实施规格

## 1. 结论

该功能可以只修改 `astrbot_plugin_gitee_aiimg` 实现，不需要修改 AstrBot 源码，也不需要修改 ContextAware。

再次审查后的支持边界必须同时写清：首版只支持一个 AstrBot 进程中一个正在工作的 Gitee 插件实例；持久化账本使用插件数据目录中的 SQLite，不再使用多 key AstrBot 插件 KV 模拟事务；`delivery_state=confirmed` 只表示受支持 adapter 的发送调用已经成功返回，即 transport accepted，不等价于用户终端已读或平台提供了端到端 receipt。消息平台没有 idempotency key 时，发送调用成功返回前后的崩溃窗口只能收敛为 `unknown`，不能承诺 exactly-once。

实施路线不是移植 AstrNa 的群聊并发补丁，也不直接使用 AstrBot 当前的 `FunctionTool.is_background_task`。插件采用以下闭环：

1. LLM 调用 `aiimg_generate` 或 `aiimg_batch_generate` 后，插件只做参数校验、输入图片固化、任务容量预留和任务登记。
2. 插件完成有界参数解析和输入固化后，创建自己的 `asyncio.Task` 执行批量规划和耗时生图，并在调用 planner LLM 或图片 provider 前向当前 Agent 返回 `accepted` 结果。
3. 当前 Agent 根据人格自然告诉用户“任务已经开始，可以继续聊天”，随后释放 AstrBot 的会话锁。
4. 后续每次正常 LLM 请求，插件通过 `on_llm_request(priority=-20)` 注入单图或批量 parent/child 状态、已生成的详细提示词、耗时和发送状态。
5. 图片生成完成后，后台 worker 先发送图片，并把 adapter 成功返回或 `unknown` 写入事务账本，再向原平台提交一个内部合成事件。
6. 内部合成事件重新经过 AstrBot 正常 pipeline，因此会获得 LLM 阶段的 session lock，并执行 ContextAware 等插件的 `on_llm_request` Hook；`commit_event()` 只表示成功放入 Core event queue，不表示 pipeline 已启动或文本已经发出。
7. 正常同进程、同 conversation 的成功、失败和 `/stop` 取消由主 Agent 根据原人格和当前上下文自然回应；`/reset`、`/new`、重启恢复等不应重入旧上下文的路径使用自然措辞的确定性主动通知。
8. 当前接单事件进入发送前 decoration 时若没有非空自然文本，插件先补一条确定性接单短句，再由 `after_message_sent` 确认 transport accepted；终态 Agent 通知未被 synthetic event claim 时由 watchdog 兜底，已进入发送但结果不明时收敛为 unknown，禁止静默或盲目重复。

首版只覆盖插件元数据中声明支持的平台：

- `aiocqhttp`
- `weixin_oc`

其他平台必须先通过能力检测和真实消息测试，不能默认宣称支持。

## 2. 不采用的方案

### 2.1 不移植 AstrNa 群聊并发补丁

AstrNa 的目标是放宽群消息处理并发，不是管理一个跨数分钟的生图任务。它不能完整解决：

- 同一私聊会话被 Tool 长时间占用。
- 同一用户继续聊天时的任务状态查询。
- 图片发送成功后让主 Agent 再次按人格回应。
- 失败、取消、插件重载和进程重启后的状态收敛。
- ContextAware 临时场景在完成通知中的注入。

同时，AstrNa 依赖较多 Core 私有实现和运行时 patch，不符合“只在 Gitee 插件内做稳定功能”的目标。

### 2.2 不直接使用 `FunctionTool.is_background_task`

AstrBot `4.26.8` 和当前上游 `master` 的原生后台 Tool 路径存在以下限制：

- 非内置插件 Tool 会经过 `_PermissionGuardedTool` 包装，包装对象没有复制 `is_background_task`，后台标记会丢失。
- 原生完成唤醒直接调用 `build_main_agent()`，不会执行 pipeline 中的 `OnLLMRequestEvent`，ContextAware 的场景无法自动注入。
- 原生完成唤醒没有获取普通消息使用的 session lock，存在与用户新消息并发更新历史的风险。
- 原生后台任务未注册到 `active_event_registry`，`/stop`、`/reset`、`/new` 无法可靠控制任务。
- 插件拿不到 Core 生成的 task ID，难以建立详细状态账本和幂等通知。

为这些问题修改 AstrBot Core 虽然可行，但违反本项目的实施边界，因此本设计不采用该路径。

## 3. 当前阻塞点

当前 `aiimg_generate` 在 Tool handler 内直接等待全部工作完成：

```text
aiimg_generate
  -> 路由判断
  -> await draw.generate / edit.edit
  -> await _send_image_with_fallback
  -> 返回 CallToolResult
```

Tool 运行在 Internal Agent 的 UMO session lock 内。线上一次生图约需 `110` 至 `313` 秒，因此整个 Agent 和当前会话会被锁住数分钟。

当前 `aiimg_batch_generate` 更重：

```text
aiimg_batch_generate
  -> await _plan_batch_prompt_items（最多 3 次聊天 LLM 调用）
  -> await _run_batch_specs（多张图片并发生成）
  -> await _send_batch_results（逐张发送）
  -> 返回 CallToolResult
```

因此批量后台化必须从 planner 阶段开始，不能只把 `_run_batch_specs()` 套进后台。现有批量成功数按“生成成功”统计，`_send_batch_results_single()` 又没有把每张 `SendImageResult` 汇总回父任务；后台版本必须改成按“实际发送确认”统计，否则会出现 Tool 声称整批已发送、实际有图片发送失败的假成功。

现有代码虽然已经具备以下基础能力，但不是完整后台任务系统：

- `_image_inflight`、`_video_inflight` 只保存计数，无法查询具体任务。
- `_video_tasks` 只跟踪视频 task，没有 UMO、sender、conversation 和状态元数据。
- `_append_plugin_conversation_note()` 采用“读取完整历史后整体覆盖”的方式，后台并发时可能发生 lost update。
- `_save_last_image_task_meta()` 只在图片成功发送后记录最后一次任务，不包含 running、failed、cancelled 或 interrupted。
- `_last_image_by_user` 只按 sender ID 隔离，同一用户跨群可能串用最后图片。

后台生图不能在这些结构上简单套一层 `create_task()`，必须补齐状态、取消、通知和会话隔离。

## 4. 总体架构

### 4.1 新增模块

新增：

```text
core/background_tasks.py
```

该模块集中承载以下职责：

- `ImageTaskRecord` 数据结构。
- `PreparedImageJob` 进程内执行对象，只保存已固化输入、解析后的 output intent 和 provider 调用参数，不保存原始 event。
- `PreparedBatchJob` 批量 parent 执行对象，保存原始总要求、目标数量、共享输入 manifest 和规划参数。
- `BatchItemRecord` 子任务结构，逐张保存 effective prompt、比例、生成状态、发送状态、错误和 receipt。
- `TaskDeliveryTarget` 投递目标，只保存 platform、UMO、sender、group、self ID 和 source message ID。
- 内存任务表和 `asyncio.Lock`。
- SQLite 事务账本、owner lease、capacity reservation、receipt 和 outbox。
- 任务状态迁移校验。
- 全局 provider semaphore 和有界等待队列。
- 单并发 planner semaphore，以及批量 child item 的容量预留和公平调度。
- 后台 task 和取消监视器。
- 内部合成事件构造与提交。
- 完成通知 watchdog。
- 启动恢复和过期清理。

所有 `asyncio.create_task()`，包括 single/batch worker、planner、child runner、notification watchdog、reset barrier finalizer、owner heartbeat、GC 和 recovery drain，都必须通过 manager 的 `_managed_tasks` 集合创建和持有强引用。统一 done callback 必须移除引用、读取 `task.exception()` 并记录未处理异常；禁止散落 fire-and-forget task。每个 UMO 的 send gate、notification queue、barrier 和 scheduler deque 在无 active task、无 waiter、无 outbox 后立即删除，不能让访问过的群聊永久驻留内存。

这是一个有明确边界且会被多条执行路径复用的复杂模块，单独拆分比继续扩张四千余行的 `main.py` 更安全。

`main.py` 只负责：

- 初始化任务管理器。
- 注册 Hook。
- 将 `aiimg_generate` 的耗时部分交给任务管理器。
- 将 `aiimg_batch_generate` 的 planner、并发生成和逐张发送全部交给任务管理器。
- 调用现有 draw、edit、selfie 和发送逻辑。
- 在 `terminate()` 中关闭任务管理器。

后台 worker 禁止持有原始 `AstrMessageEvent`。原事件只在 Tool 快速阶段用于解析 sender、session、conversation、引用图和平台信息；Tool 返回后，worker 只能使用 `PreparedImageJob` 和 `TaskDeliveryTarget`。这既避免事件临时文件被清理，也避免把一个已经结束的 pipeline event 当作长期可复用对象。

### 4.2 执行时序

```text
用户消息
  -> 主 Agent 调用 aiimg_generate
  -> 插件分配 task_id，创建 preparing record 并固化输入
  -> 插件完成 prompt/route/output intent 解析
  -> 任务进入 queued 并创建后台 worker
  -> 后台 worker 启动
  -> Tool 返回 accepted + 任务摘要
  -> 主 Agent 按人格回复“我开始画了”
  -> 当前 Agent 结束，session lock 释放

用户继续聊天
  -> 正常 pipeline
  -> ContextAware on_llm_request(priority=-10)
  -> Gitee 状态注入 on_llm_request(priority=-20)
  -> Agent 可看到任务状态、详细提示词和实时耗时

后台 worker
  -> 等待全局 provider semaphore
  -> running
  -> sending
  -> adapter 发送调用成功返回
  -> completed(image_sent=true)
  -> 提交内部合成事件
  -> 正常 pipeline 获取 session lock
  -> ContextAware 注入当前场景
  -> Gitee 注入终态任务详情并禁用生图 Tool
  -> 主 Agent 按人格自然回应
  -> after_message_sent 校验 transport-accepted 标记后写 notification_state=sent
```

批量时序：

```text
主 Agent 调用 aiimg_batch_generate(count=N)
  -> 固化共享参考图并原子预留 N 个 child capacity
  -> 创建 batch parent(state=planning)
  -> Tool 返回 accepted，session lock 释放
  -> planner semaphore 内调用 _plan_batch_prompt_items
  -> 为 N 个 child 写入完整 effective prompt 和 output intent
  -> child 按全局 provider semaphore 并发生成
  -> 每个 child 独立记录 generated/failed
  -> 生成阶段收敛后，成功 child 按原索引顺序逐张发送
  -> 每张独立写 image receipt
  -> parent 汇总 completed/partial/failed/cancelled
  -> 只提交一次 parent 终态合成事件
  -> Agent 自然总结“计划 N 张，已发 X 张，失败 Y 张”
```

批量任务不会为每张图片都唤醒一次 Agent，避免群聊被连续 N 条“第几张完成了”刷屏。生成过程中的逐项状态通过普通对话 Hook 查询；整批终态只自然回应一次。

正常同 conversation 的失败和 `/stop` 取消走同一条 Agent 终态通知链；reset/new/restart 中断走确定性主动通知。所有路径都必须按账本如实区分 `image_sent=false`、`delivery_state=confirmed` 和 `delivery_state=unknown`；其中 `confirmed` 的用户可见文案可以说“已经发出来”，但技术日志和测试必须标注其真实语义是 adapter transport accepted。

Tool 的接单回复有独立 acknowledgment 状态。正常情况由当前主 Agent 自然接单；若进入 `on_decorating_result` 时最终 chain 没有非空 Plain，Hook 直接补一次自然短句，随后由 `after_message_sent` 确认。worker 可以立即开始调用 provider，但在进入图片发送阶段前会短暂等待接单状态解析，避免极快任务出现“图片先到，接单话后到”的倒序。若主 Agent 连 decoration 阶段都没有进入，当前 session lock 通常也尚未释放，这属于聊天 LLM/pipeline 故障，不应再用一个会与迟到 Agent 回复竞态的独立定时接单消息掩盖。

## 5. 任务状态机

```text
preparing
  -> queued
  -> running
  -> sending
  -> completed

preparing | queued | running | sending
  -> failed
  -> cancelled
  -> interrupted
```

批量 parent：

```text
preparing
  -> planning
  -> queued
  -> running
  -> sending
  -> completed | partial

preparing | planning | queued | running | sending
  -> failed
  -> cancelled
  -> interrupted
```

批量 child：

```text
planned -> queued -> running -> generated -> sending -> completed
planned | queued | running | generated | sending
  -> failed | cancelled | unknown
```

状态语义：

- `preparing`：已经创建账本记录，正在固化参考图、解析模式和构建 effective prompt；该阶段仍处于当前 Tool 调用内。
- `queued`：单图的 effective prompt，或 batch 已规划 child 的 effective prompts 已经写入账本，任务进入图片 provider 队列。单图 Tool 到达 queued 才返回 `status=accepted`；batch Tool 在输入固化、容量预留和 parent planning record 持久化后即可返回 accepted。
- `running`：已经调用图片 provider，等待生成结果。
- `sending`：图片已经生成，正在发送给消息平台。
- `completed`：受支持 adapter 的图片发送调用已经成功返回，必须满足 `image_sent=true` 和 `delivery_state=confirmed`；这不是用户终端已读证明。
- `failed`：生成或发送失败，记录经过脱敏和截断的错误。
- `cancelled`：用户通过 `/stop`、`/reset`、`/new` 或插件命令取消。
- `interrupted`：插件重载、AstrBot 重启或任务生命周期异常中断。
- `planning`：仅用于 batch parent，后台 planner 正在生成每张图片的差异化提示词和比例。
- `partial`：批量任务至少有一张确认发送，并且同进程内存在已明确 failed 或 cancelled 的 child；通过 `terminal_reason` 区分失败或取消。只要存在投递结果 unknown，parent 必须使用 `interrupted`，不能伪装成已知 partial。

禁止把 provider 没有提供的进度伪造成百分比。Bot 只能看到真实阶段和实时耗时，例如“已运行 182 秒，仍在等待 provider 返回”。

全局同时运行的图片 provider 调用由 semaphore 限制，队列也必须有上限。队列已满时 Tool 直接返回明确失败，不得继续创建 detached task。现有 `max_user_concurrency` 继续约束单个 scope 的 parent task 数量，任务槽位从 `preparing` 一直持有到终态，不能在 Tool 返回时提前释放。

批量任务在接单前按 requested_count 原子预留 child capacity，不能让一个 8 张 batch 在账面上只占一个队列槽。planner 使用独立的单并发 semaphore，图片 child 使用与单图任务共享的全局 provider semaphore。调度器按 parent task 做 work-conserving round-robin：只有一个 parent 时可以吃满允许并发；出现其他 single/batch parent 后，下一个空闲槽优先轮转给不同 parent。每个 batch 同时仍受 `min(mode.batch_concurrency, max_running)` 限制，防止一整批长期独占所有 provider 槽位。

requested_count 大于当前可预留容量时整批拒绝并让 Agent 如实说明“当前队列放不下这么多张”，不得静默缩减用户要求的数量，也不得先接单再只生成一部分。

## 6. 任务记录

每条任务至少保存：

```json
{
  "schema_version": 1,
  "revision": 7,
  "task_id": "img_01J...",
  "state": "running",
  "task_kind": "batch",
  "terminal_reason": "",
  "owner_instance_id": "plugin_01J...",
  "owner_epoch": 17,
  "umo": "platform_id:GroupMessage:session_id",
  "platform_id": "platform-instance-id",
  "platform_name": "aiocqhttp",
  "message_type": "GroupMessage",
  "session_id": "session_id",
  "group_id": "123456",
  "self_id": "987654",
  "sender_id": "111222",
  "sender_name": "用户昵称",
  "conversation_id": "conversation-uuid",
  "reset_epoch": 3,
  "source_message_id": "message-id",
  "request_fingerprint": "sha256(scope+source_message_id+normalized_args)",
  "mode": "selfie_ref",
  "requested_count": 4,
  "planned_count": 4,
  "generated_count": 2,
  "sent_count": 1,
  "failed_count": 0,
  "cancelled_count": 0,
  "unknown_count": 0,
  "backend_requested": "auto",
  "aspect_ratio": "3:4",
  "resolution": "2K",
  "user_prompt": "用户原始要求",
  "effective_prompt": "实际提交给图片 provider 的完整提示词",
  "current_attempt": 1,
  "attempts": [
    {
      "attempt": 1,
      "mode": "selfie_ref",
      "effective_prompt": "本次高层路由实际使用的提示词",
      "state": "running",
      "error_code": ""
    }
  ],
  "reference_source": "webui",
  "reference_count": 1,
  "extra_reference_count": 0,
  "image_generated": false,
  "image_sent": false,
  "delivery_state": "not_started",
  "send_attempt_id": "",
  "input_manifest": [
    {
      "relative_path": "background_tasks/img_01J.../inputs/0001.bin",
      "size": 123456,
      "sha256": "..."
    }
  ],
  "items": [
    {
      "item_id": "img_01J..._01",
      "index": 1,
      "state": "completed",
      "mode": "selfie_ref",
      "user_prompt": "第 1 张规划要求",
      "effective_prompt": "第 1 张实际提交给 provider 的完整提示词",
      "aspect_ratio": "3:4",
      "image_generated": true,
      "image_sent": true,
      "delivery_state": "confirmed",
      "error_code": "",
      "send_attempt_id": "send_01J..._01"
    }
  ],
  "send_reason": "",
  "error_code": "",
  "error_message": "",
  "cancel_reason": "",
  "created_at": 0,
  "started_at": 0,
  "updated_at": 0,
  "finished_at": 0,
  "notification_token": "notify_01J...",
  "notification_state": "pending",
  "notification_queued_at": 0,
  "notification_sent_at": 0,
  "ack_token": "ack_01J...",
  "ack_state": "pending",
  "ack_sent_at": 0
}
```

约束：

- `effective_prompt` 必须在调用 provider 前更新。
- auto-selfie 等高层路由发生 fallback 时，必须先追加 `attempts` 并更新当前 effective prompt，再调用下一条 provider 路径，不能让 Bot 看到已经过期的提示词。
- 单图 `completed` 必须同时满足 `image_generated=true` 和 `image_sent=true`。
- 图片生成成功但发送失败属于 `failed`，不能伪装成 completed。
- `delivery_state` 只允许 `not_started -> attempting -> confirmed|unknown`。进程在发送阶段崩溃时只能恢复为 `unknown`，禁止自动重发。
- `confirmed` 只表示 `aiocqhttp` 或 `weixin_oc` 的发送 await 成功返回；adapter 没有保留平台真实 message ID 时，不得在日志、测试或恢复逻辑中把它升级成端到端送达证明。
- 图片和人格化终态文本必须作为两个独立发送 attempt 记录。尤其是 `weixin_oc`，同一富媒体链可能出现前置文字与 media 部分成功，首版禁止把图片 caption 和图片合并成一个不可拆分 receipt。
- 错误文本必须去除 API key、Authorization header、完整 URL query 和超长响应体。
- 任务状态只能单向迁移，终态不可重新变回 running。
- `input_manifest` 只保存插件数据目录中的相对路径、大小和哈希，不把参考图 bytes 写进 SQLite record。
- 单图 task 的 `task_kind=single` 且 `items=[]`；批量 task 的 `task_kind=batch`，所有逐图事实必须写入 `items`，parent 的计数字段只能由 child 汇总生成。
- batch parent 的 `image_generated/image_sent` 只表示“至少有一张”，完整性判断必须使用 generated_count/sent_count；每个 child 还必须保存与单图相同的 attempts 列表。
- batch `completed` 要求 `sent_count=requested_count`；`partial` 要求 `sent_count>0` 且至少一个 child 已明确 failed/cancelled；`failed` 要求 `sent_count=0` 且不是用户主动取消；任何 child 为 unknown 时 parent 必须为 interrupted。
- planner 尚未完成时 `items=[]`，Bot 只能看到 parent 原始要求和 planning 状态，不能编造尚不存在的子提示词。
- `request_fingerprint` 在同一 scope 内保留 10 分钟去重窗口；相同 source message 和规范化参数再次触发时返回原 task，不重复预留容量或调用 provider。

## 7. SQLite 事务账本

AstrBot 插件 KV 的 `put_kv_data()` 只能逐 key upsert，没有 multi-key transaction、CAS 或可靠前缀扫描。task、catalog、active、reservation 和 outbox 如果分开写，进程在任意两步之间退出都会留下孤儿任务或永久占用容量。首版不再用插件 KV 承担后台任务账本，改用 Python 标准库 `sqlite3`，数据库固定落在 AstrBot 插件数据目录：

```text
<plugin_data>/background_tasks/background_tasks.sqlite3
```

路径使用公开 `StarTools.get_data_dir()` 取得并用 `pathlib.Path` 拼接，不能写死插件安装目录。Linux 上数据目录权限设为 `0700`、数据库和输入文件设为 `0600`；其他平台 best effort 保持仅当前用户可读。info 日志禁止输出完整 user/effective prompt、参考图内容或 URL query，只记录 task ID、长度、哈希和脱敏错误。数据库启用：

```text
PRAGMA journal_mode=WAL
PRAGMA synchronous=FULL
PRAGMA foreign_keys=ON
PRAGMA busy_timeout=5000
```

所有写事务由 manager 的单个 `asyncio.Lock` 串行，并通过 `asyncio.to_thread()` 执行短连接事务，避免 fsync 阻塞 event loop。每个事务使用 `BEGIN IMMEDIATE`；禁止在事务或 manager lock 内等待 provider、平台发送、conversation manager、文件下载或内部事件提交。

最小表结构：

```text
runtime_owner(
  singleton PRIMARY KEY,
  owner_instance_id,
  owner_epoch,
  heartbeat_at_ms,
  state
)

tasks(
  task_id PRIMARY KEY,
  scope_hash,
  state,
  task_kind,
  owner_epoch,
  request_fingerprint,
  record_json,
  created_at_ms,
  updated_at_ms,
  expires_at_ms
)

reservations(
  task_id PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
  total,
  remaining,
  released
)

receipts(
  send_attempt_id PRIMARY KEY,
  task_id REFERENCES tasks(task_id) ON DELETE CASCADE,
  item_id,
  kind,
  delivery_state,
  transport,
  response_digest,
  created_at_ms
)

notification_outbox(
  token PRIMARY KEY,
  task_id REFERENCES tasks(task_id) ON DELETE CASCADE,
  kind,
  state,
  attempt_id,
  payload_json,
  queued_at_ms,
  updated_at_ms
)

request_dedupe(
  request_fingerprint PRIMARY KEY,
  task_id,
  expires_at_ms
)
```

`tasks` 至少建立 `(scope_hash, updated_at_ms)`、`(state, updated_at_ms)` 和 `(expires_at_ms)` 索引。`scope_hash` 对 `(umo, self_id, sender_id, conversation_id)` 做稳定哈希。任务详情仍保存在 versioned `record_json` 中；可查询和需要原子约束的状态、时间、owner、scope、fingerprint 单独成列，禁止只靠反序列化 JSON 扫全表。

schema migration 必须按单调 `schema_version` 在一个 `BEGIN IMMEDIATE` 事务中执行；遇到高于当前代码支持的版本时 fail closed，不得降级覆盖。迁移前使用 SQLite backup API 生成单个轮换备份，最多保留 3 份；禁止在数据库打开且 WAL 未 checkpoint 时只复制主 `.sqlite3` 文件冒充完整备份。

接单事务必须一次完成：

1. 校验 live owner lease、scope parent 限额、全局 capacity 和 request fingerprint。
2. 插入 task、reservation 和 request_dedupe。
3. 将 single 从 `preparing` 推进 `queued`，或将 batch 推进 `planning`。
4. 提交成功后才创建进程内 worker 并返回 `accepted`；事务失败则不创建 task。

capacity 以 reservation 为唯一事实源。single 的 `total=remaining=1`；batch 的 `total=remaining=requested_count`。child 从非终态进入终态时，在同一事务内仅当旧状态仍非终态才把 `remaining` 减一；planner 失败或 parent 在 child 建立前取消时一次性释放全部 remaining；parent 终态必须断言 `remaining=0`。不得同时在普通异常分支和 `finally` 各减一次，`finally` 只调用按旧状态判断的幂等收敛事务。

进程内公平队列采用最小实现：每个 parent 一个 ready child deque，另有一个 parent ring。每释放一个 provider 槽，只从 ring 下一个 parent 取一个 child；该 parent 仍有 ready child 时放回 ring 尾部。single 被视为只有一个 child 的 parent。取消或终态 parent 立即从 ring 移除。只有一个 parent 时允许持续占用空闲槽，因此保持 work-conserving；多个 parent 时不会让一个大 batch 把后续空闲槽全部吃完。

owner lease 只用于防止同一 SQLite 文件被两个插件实例同时执行：

- `initialize()` 在事务中取得 `runtime_owner`，生成新的 `owner_epoch` 并每 10 秒 heartbeat。
- 发现 45 秒内仍有 live owner 时，后台模式 fail closed，原同步路径继续可用，禁止第二个 manager 接管。
- worker 的每次状态更新都带 `WHERE owner_epoch=?`，发送图片、提交 synthetic event 和 drain outbox 前重新校验 live lease。
- graceful terminate 先把 owner 设为 draining，收敛 worker 后释放 lease；crash 后新实例只能在 lease 过期后接管。
- 多个 AstrBot 实例若使用不同 plugin data 目录，彼此看不到同一 lease，首版明确不支持这种 active-active 部署。京东云必须只运行一个 AstrBot 实例和一个 Gitee 插件 owner。

lease 检查与真实网络发送之间仍有不可原子化的小窗口，因此即使同库双 owner 被 fencing，也不能宣称平台发送 exactly-once。该窗口统一按 `delivery_state=unknown` 处理，不自动重发。

事件依赖输入落在：

```text
<plugin_data>/background_tasks/<task_id>/inputs/
```

写文件使用 `.part` 临时文件加原子替换。任务进入终态后按保留策略清理输入；生成结果如果仍被“最后图片/重发图片”引用，则按现有图片缓存策略保留。

保留策略：

- active 任务最长保留 2 小时；超时后写 cancellation tombstone、取消 worker、在 shield transaction 中标记 failed/interrupted 并释放 reservation，不能只改数据库状态后留下 provider task 继续跑。
- completed、failed、cancelled、interrupted 保留 24 小时。
- 注入 LLM 的完整终态任务只保留 30 分钟。
- 每个 scope 最多保留 16 条记录。
- terminal task 总数最多 512；outbox 最多 512 项；达到硬上限且无法安全 GC 时拒绝新任务并记录告警。
- 单个 user/effective prompt 最多 32768 字符，单个 task 的 `record_json` 最多 512 KiB，单个输入文件最多 20 MiB、每个 task 输入总量最多 64 MiB；超过时在接单阶段拒绝。
- 时间戳统一使用 UTC Unix milliseconds；运行中耗时使用 monotonic clock，GC 对超过当前时间 5 分钟的 future timestamp 做 clamp 并告警。
- GC 在单个事务中删除过期 task、reservation、receipt、outbox 和 dedupe，再双向清理对应输入目录；文件删除失败记录重试计数，不能让异常终止整个 GC。
- recovery outbox 最长保留 24 小时并限制退避次数；超过期限后标记 expired/unknown、保留审计摘要并停止自动发送，禁止永久重试和无限增长。
- 启动时执行 `PRAGMA quick_check`；失败时禁用后台模式并保留数据库供人工排查，不自动删除或重建账本。完整 `integrity_check` 只用于离线诊断。

## 8. Tool 快速返回

`aiimg_generate` 的同步阶段只允许执行：

- 参数标准化和配置检查。
- 去重和并发额度检查。
- 确定 UMO、sender、conversation 和 reset epoch。
- 解析 mode、output intent、aspect ratio 和 resolution。
- 固化本轮消息中的输入图片为插件数据目录文件。
- 构建 text/edit 场景的 effective prompt。
- 自拍模式读取参考图并构建完整 effective prompt。
- 创建 preparing record，持久化输入 manifest，将状态推进到 queued，再创建后台 task。

禁止在快速阶段调用真正的生成 provider。

输入图片必须在 Tool 返回前固化到插件数据目录。AstrBot pipeline 结束时会清理事件临时文件，如果后台 worker 继续引用原始临时路径，可能在生成开始前就失效。大图只保留文件，不在任务表中长期持有 bytes；每个输入文件必须有大小上限、总量上限和哈希。

自拍链路当前把“读取参考图、构建 final prompt、解析 chain override、调用 edit provider”放在同一个函数中。实施时必须把 provider 调用前的部分拆成一次 `PreparedImageJob` 构建，不能把现有 `_generate_selfie_image_with_meta()` 整体丢进后台，否则 worker 仍然依赖原事件，且 Tool 返回时并不知道真实 effective prompt。

`aiimg_batch_generate` 的同步阶段只允许执行：

- 校验 count、mode、backend、output intent 和功能开关。
- 固化 edit/selfie 共用的输入图片和参考图。
- 获取原 conversation、scope、epoch 和 delivery target。
- 原子预留 requested_count 个 child capacity。
- 创建 `task_kind=batch`、`state=planning` 的 parent record 和后台 worker。

禁止在 Tool 内调用 `_plan_batch_prompt_items()`、`_run_batch_specs()` 或 `_send_batch_results()`。planner 必须在后台使用该 UMO 对应的 provider，而不是无 UMO 的默认 provider；最多 3 次规划重试仍保留，但受 cancellation 和 timeout 控制。规划成功后，每个 child 的完整 effective prompt、比例和 output intent 写入账本，然后才进入图片 provider 队列。

Tool 返回内容使用结构化文本，至少包含：

```json
{
  "status": "accepted",
  "task_id": "img_01J...",
  "mode": "selfie_ref",
  "user_prompt": "...",
  "effective_prompt": "...",
  "message": "The image task is running in the background. Respond naturally as yourself and tell the user they can continue chatting. Do not imply completion."
}
```

批量 Tool 返回：

```json
{
  "status": "accepted",
  "task_id": "batch_01J...",
  "task_kind": "batch",
  "state": "planning",
  "requested_count": 4,
  "mode": "selfie_ref",
  "user_prompt": "...",
  "message": "The batch image task is planning and running in the background. Tell the user how many images were accepted and that they can continue chatting. Child prompts will become available after planning."
}
```

当前主 Agent 已经知道自己的 Tool call 参数，再获得 accepted 结果后可以按人格自然回应。这里不发送固定“任务已创建”模板。

返回 accepted 前在原事件 extra 中写入 task_id 和 ack token。正常 Agent 回复由 `on_decorating_result` 确保存在非空 Plain，再由 `after_message_sent` 确认 adapter await 成功返回。后台功能开启时，原有 `_begin_user_job()` 的计数所有权必须转交任务管理器，并在终态释放，不能继续依赖 Tool handler 的 `finally` 提前执行 `_end_user_job()`。

## 9. 必要 Hook

### 9.1 `on_llm_request(priority=-20)`

职责：

- 向普通对话注入当前 sender/scope 的 active task。
- 注入最近的 terminal task，使 Agent 知道图片是否已经发出。
- 计算动态 elapsed seconds。
- 对内部完成事件注入指定 task 的完整详情。
- 对内部完成事件移除 `aiimg_generate`、`gitee_draw_image`、`gitee_edit_image`、`aiimg_batch_generate` 和 `send_message_to_user`，强制 Agent 用一次普通最终文本回应，防止递归生图或绕过 respond stage 的发送确认。

ContextAware 当前使用 `priority=-10`。AstrBot Hook 按 priority 从高到低执行，因此 Gitee 使用 `-20`，顺序为：

```text
其他普通 Hook
  -> ContextAware 场景
  -> Gitee 生图任务状态
```

注入使用临时 `TextPart`：

```xml
<background_image_tasks>
  <task>
    <state>running</state>
    <current_attempt>1</current_attempt>
    <requester>用户昵称</requester>
    <elapsed_seconds>182</elapsed_seconds>
    <user_prompt>...</user_prompt>
    <effective_prompt>...</effective_prompt>
    <image_generated>false</image_generated>
    <image_sent>false</image_sent>
    <delivery_state>not_started</delivery_state>
    <notification_state>pending</notification_state>
  </task>
</background_image_tasks>
```

注入限制：

- 当前 sender 的最新 active task 可包含完整 effective prompt。
- batch parent 在 `planning` 时注入 requested_count、原始 user_prompt 和规划耗时；规划完成后注入 child index、child state、effective_prompt、image_generated/image_sent 和错误摘要。
- 自动状态块被长度上限截断时写 `prompts_truncated=true` 和 task_id，明确要求 Agent 在用户追问全部提示词时调用 `aiimg_task_status` 分页查询，不能凭摘要补写。
- 同群其他用户的任务只注入 requester、state 和 elapsed，避免上下文膨胀。
- 最多注入 3 条 parent task；每个 batch 最多展开 8 个 child 摘要，超过时只给出计数和最近变化的 child。
- 总长度上限 6000 字符。
- TextPart 必须调用 `mark_as_temp()`，避免每轮重复持久化整个状态块。
- user/effective prompt 必须做 XML 转义或改用严格 JSON 序列化，不能让提示词中的伪闭合标签破坏状态块结构。
- reset/new 取消且被标记为 `suppress_future_injection=true` 的终态只用于当次确定性通知，不再注入清空后的新上下文。

### 9.2 `aiimg_task_status` 只读 Tool

批量任务最多 8 至 32 个 child，不能每轮都把所有完整提示词塞进 6000 字符的自动状态块。新增一个无副作用、立即返回的只读 LLM Tool：

```text
aiimg_task_status(task_id="", include_prompts=true, offset=0, limit=8)
```

- task_id 为空时只查询当前 sender/scope 的最新 active 或 terminal task。
- batch 返回 parent 汇总、每个 child 的状态、完整 effective prompt、比例、image_sent 和错误摘要。
- limit 限制为 1 至 8；超过 8 张时通过 offset 分页，并返回 total/next_offset，避免一个状态查询塞爆上下文。
- 禁止通过 task_id 查询其他 sender 的完整提示词；群里其他人的任务仍只能返回公开摘要。
- 该 Tool 不调用任何 provider，不创建任务，不改变状态，也不受生图并发额度限制。
- 当用户明确询问“每张图的详细提示词”而自动注入已截断时，Agent 应调用该 Tool 获取完整事实。

### 9.3 平台消息 Hook `priority=10`

增加一个所有平台消息监听器，用于：

- 在其他普通消息插件和内置命令前识别 `/stop`、`/reset`、`/new`，以及已经由 cmdmask 写入 extra 的目标命令。
- `/stop` 直接向插件 task registry 写 cancellation tombstone，取消当前 UMO/scope 的 single 或 batch parent，并把取消数量写入命令 event extra；不伪造 `AstrMessageEvent` 注册到 Core `active_event_registry`。
- 为 `/reset`、`/new` 建立短生命周期 send barrier，但不在权限和命令结果确认前立即取消任务。
- 对内部合成事件校验 task_id、notification token 和 delivery attempt，重复事件在进入 LLM 前停止。

该 Hook 不得覆盖或吞掉 AstrBot 内置命令结果。

`/reset` 和 `/new` 不能在这个前置 Hook 中直接增加 epoch。群聊 reset 可能因权限不足被内置命令拒绝；如果插件先取消任务，就会出现“reset 没成功，图却被停了”的误伤。正确做法是先建立 send barrier，等同一命令事件进入发送前 decoration 阶段后，根据 Core 已设置的 `_clean_group_context_session=true` 成功标记推进 epoch 和取消任务。barrier 期间 worker 不得进入 sending；命令被拒绝、执行异常或 marker 缺失时解除阻塞，原任务继续运行。

### 9.4 `on_decorating_result`

该公开 Hook 位于平台发送前，用于处理已经由内置命令或 Agent 生成的最终 result：

- `/stop` event extra 中取消数量大于 0 时，把结果改成一致的自然说明，例如“已请求停止当前会话中的 2 个后台生图任务，其他 Agent 任务也按 AstrBot 的停止逻辑处理”，避免内置 registry 不认识插件 task 时同时回复“没有运行任务”。
- `/reset`、`/new` 只有在 `_clean_group_context_session=true` 时才在 SQLite 事务中推进 reset epoch、写 cancellation tombstone 并解除 send barrier；marker 缺失时只解除 barrier。
- 内部 notification event 在发送前记录最终非空 Plain 文本摘要，并将 `notification_state=queued` CAS 为 `agent_sending`，但此时不得标记 sent。
- 原始 Tool event 在发送前记录最终 ack Plain 文本摘要，供发送后校验；streaming result 无法稳定取得最终 chain 时后台模式同步降级。

Core 当前正常 Internal Agent 的 `/reset` 和 `/new` 成功路径都会设置 `_clean_group_context_session`；权限拒绝、无 conversation 和 provider 缺失路径不会设置。首版后台模式只支持 Internal Agent，因此不依赖第三方 runner 的不同命令分支。

### 9.5 `after_message_sent`

内部合成事件带有：

```text
event.extra["gitee_bgimg_task_id"]
event.extra["gitee_bgimg_notification_token"]
event.extra["gitee_bgimg_internal_event"] = true
```

`after_message_sent` 即使 `event.send()` 抛异常也可能被 Core 调用，因此它只在满足以下全部条件时把 notification 标为 sent：

- task_id 和 token 与账本一致。
- `event._has_send_oper is True`。当前 `aiocqhttp` 和 `weixin_oc` 都只在 adapter 发送 await 成功返回后调用基类 `send()` 设置该字段；“Hook 被调用”本身不是传输成功证明。
- `event.get_result()` 的最终 chain 中存在非空 `Plain` 文本。仅有 At/Reply、空结果或此前发送过的 Tool 使用提示，不能冒充人格化终态回复。
- 最终 Plain 摘要与 `on_decorating_result` 记录的待发送摘要一致，避免其他插件在两个 Hook 之间替换结果后误确认。
- notification 尚未进入 sent/fallback_sent 终态。

同一 Hook 还负责：

- 原始 Tool 事件携带 ack token，且 `_has_send_oper is True`、最终 chain 含非空 `Plain` 时，写 `ack_state=sent`。
- 原始 Tool event 的 deterministic ack 如果由 decoration Hook 补入，也走相同 digest 和 `_has_send_oper` 校验后写 `ack_state=fallback_sent`。
- 内部 notification event 已进入 `agent_sending` 但 `_has_send_oper=false` 时写 `notification_state=unknown`，不再由 watchdog 并发 fallback；仍停留 queued 的事件才允许 watchdog claim。

`_has_send_oper` 是当前 AstrBot 的内部字段，因此必须在启动能力检测中 feature-detect。它只能代表一次成功返回的发送操作，不能证明用户终端已收到。字段缺失时后台模式必须同步降级，不能把 `after_message_sent` 当成成功信号硬猜。

### 9.6 `initialize()`

初始化时：

- 创建任务管理器。
- 打开 SQLite，执行 schema migration、integrity check，并取得 owner lease。
- 查询所有非终态 task 和未发送 outbox。
- 将上次进程残留的 preparing/planning/queued/running，以及 batch 中尚未发送的 generated child 标记为 interrupted。
- 将残留 sending single/child 标记为 interrupted，并把 `delivery_state` 设为 unknown；禁止自动重发图片。
- 将重启恢复的 interrupted/unknown 通知放入 recovery outbox；此阶段只持久化，不假设平台已经可发送。
- 清理过期记录。
- 执行平台和 AstrBot 接口能力检测。

### 9.7 `on_astrbot_loaded()` 与 recovery drain

AstrBot 完成启动后 drain recovery outbox：

- 重新按 platform ID 定位 adapter，平台尚不可用时做有限退避重试。
- 对 interrupted/unknown 使用确定性主动消息，不提交 ContextAware 合成事件。
- 只有主动发送成功后才把 recovery notice 标为 sent；失败项保留到下一次 drain 或首次正常平台消息触发时再试。
- 热重载场景中平台通常已经存在，`initialize()` 可以额外启动一个短延迟 drain；该 drain 与 `on_astrbot_loaded()` 使用同一 outbox token 和事务 claim，避免同一进程内重复发送。平台在成功返回前断线时仍只能记为 unknown，不能承诺跨崩溃 exactly-once。

### 9.8 `terminate()`

终止时：

- 设置 manager closing 标志，拒绝新任务。
- 停止 scheduler、owner heartbeat 和新的 outbox claim。
- 对所有 worker 写 cancellation tombstone 后调用 `task.cancel()`，并以 `gather(return_exceptions=True)` 等待有限时间收敛。
- worker 必须显式捕获 `asyncio.CancelledError`，在 `asyncio.shield()` 保护下执行一次幂等 terminal transaction，写 interrupted/cancelled、释放 reservation、归还 inflight，再重新抛出 CancelledError。
- 超时仍未收敛的 task 由 manager 在独立事务中标记 interrupted；不得吞掉未处理 task exception。
- 清空 `_managed_tasks`、scheduler ring、per-UMO gate/queue、barrier 和 watchdog map，并断言内存 active/inflight 计数为零。
- 不在 AstrBot 完全关闭过程中强行发送消息；通知由下一次 initialize 恢复。
- 所有 worker 和 outbox drain 结束后才把 owner lease 标记 released，然后关闭 `ProviderRegistry`、draw/edit/http session，避免 worker 使用已关闭 backend 或旧 platform client。

## 10. 内部合成事件

### 10.1 目的

完成后不能直接调用裸 `llm_generate()`，因为那样不会自动获得：

- 正常 UMO session lock。
- ContextAware 的 `on_llm_request` 场景。
- 其他兼容插件的 LLM Hook。
- 正常 Agent 人格、fallback provider 和历史保存。
- 标准响应发送和 `after_message_sent`。

因此后台 worker 通过当前 platform instance 的 `create_event()` 和 `commit_event()` 提交内部事件，让它重新走完整 pipeline。`commit_event()` 当前只执行 event queue 的 `put_nowait`，没有返回 event、pipeline task 或 transport receipt；插件必须在提交前持久化 outbox，在 `after_message_sent` 之前都不能把 notification 视为已发送。

### 10.2 构造要求

内部事件必须保留原任务的：

- platform instance ID。
- message type。
- UMO/session ID。
- group ID。
- bot self ID。
- 原 requester sender ID 和 sender name。
- 原 conversation ID。

事件的 `message_str` 设为空字符串。详细任务数据由 Gitee 的 `on_llm_request` 从账本临时注入，避免 ContextAware 把整段结构化数据误记成真实用户发言。真正给 Agent 的生命周期指令放在绑定的 `ProviderRequest.prompt/system_prompt` 中，不放进 adapter message chain。

事件 message chain 不放 `Plain` 或 `Image`。为满足群聊和需要 wake prefix 的私聊唤醒门槛，放置一个只供 WakingCheck 使用的 `At(self_id)` 组件，并同时设置：

```text
event.is_wake = true
event.is_at_or_wake_command = true
```

WakingCheck 会从 At 组件确认唤醒；手工 flag 只是防御性设置，不能作为唯一唤醒依据。

事件 extra 中设置已经绑定原 conversation 的 `ProviderRequest`。Internal Agent 会因为存在 provider_request 而允许空文本组件事件继续执行，并在标准 session lock 内调用 `OnLLMRequestEvent`。

合成 `AstrBotMessage` 必须显式填充 `type`、`self_id`、`session_id`、`message_id`、`group`、`sender`、`message`、`message_str` 和 `raw_message`。每次图片发送或 notification 提交前都通过公开的 `Context.get_platform_inst(platform_id)` 重新定位当前 adapter，禁止把 adapter、bot client 或 platform event 保存进长期 job。重新定位后检查 instance ID、adapter name 和基础 readiness，再调用 adapter `create_event()`；图片 delivery event 不提交 pipeline，notification event 才调用 `commit_event()`。

`StarTools.create_event()` 当前不返回创建出的 event，无法让插件绑定 notification extra 和后续确认，因此本设计不使用该 convenience API。实现只调用公开 context 定位平台，再直接使用 adapter 的 `create_event()`/`commit_event()`；这些能力任何一项变化都触发同步降级。

### 10.3 ProviderRequest

ProviderRequest 使用：

- `conversation`：通知提交前按任务保存的原 cid 从 conversation manager 重新读取的新对象。禁止长期保存任务创建时的 conversation 对象，因为它的 history 可能已经过期。
- `prompt`：简短内部生命周期说明，例如 `[Internal lifecycle: resolve the completed background image task from plugin state.]`，不得包含用户详细提示词。
- `system_prompt`：要求 Agent 必须根据终态自然回应，不得静默，不得重新调用生图 Tool，不得暴露内部 JSON。
- `extra_user_content_parts`：由 Hook 注入详细 task record。

只能使用当前 `ProviderRequest` dataclass 已声明的字段，不假设存在任意 metadata。禁用本插件生图 Tool 时，必须复制当前 `func_tool` 集合并移除目标 Tool，不能修改全局 tool registry。

成功提示的系统约束：

```text
The image was already sent successfully. Always send one short natural follow-up in your current persona. Do not send the image again and do not claim it is still running.
```

失败、取消和中断提示使用对应状态，均明确要求必须回应。

batch 只在 parent 所有 child 都进入 terminal 且发送阶段收敛后提交一次合成事件。系统约束必须给出 requested/sent/failed/cancelled/unknown 数量：

```text
The batch task is terminal. Images with image_sent=true were already sent in original index order. Send one concise natural summary in your current persona. State partial success honestly. Do not resend images and do not start replacement tasks unless the user explicitly asks.
```

如果 parent 为 `partial`，Agent 必须说明实际发出几张以及剩余项失败、取消或未知，不能把“生成成功”说成“用户已经收到”。

AstrBot 正常历史保存会把本次 `ProviderRequest.prompt` 作为一条 user-role 输入保存，再保存 Agent 的 assistant 回复。这个简短 internal lifecycle marker 是为了让完成回应进入正常对话历史，也是“Bot 之后知道任务已经结束”的一部分；它不是用户真实发言，因此必须保持通用、明确标注 internal，绝不能把完整 task JSON 塞进去。若要求 AstrBot history 中连这一条合成 user-role marker 都不存在，就需要 Core 提供 no-save 输入能力，超出了“只改插件”的边界。

提交前必须重新确认当前 conversation ID 仍等于任务的原 cid，且 reset epoch、owner 和 task tombstone 均未变化。任一不一致时不再提交 Agent 合成事件，改走对应的确定性主动通知或直接结束，避免旧任务写入新 conversation。

Core session lock 只覆盖 Internal Agent 的 build/run/history 保存阶段，不覆盖 RespondStage 的平台发送，也不覆盖后台图片 worker。插件因此还需要每个 UMO 一个 notification dispatch queue：同一 UMO 同时只允许一个 token 进入 synthetic pipeline，后续 token 按 terminal timestamp 排队；这保证多个并发任务的人格化终态不会同时抢答，但不阻塞用户正常消息进入 Core session lock。

### 10.4 ContextAware 兼容

ContextAware 的 `on_message` 只有在 message chain 包含 Plain、Image 或 voice 内容时才记录用户消息。正常同进程、同 conversation 的完成事件只包含 At，因此不会在 `on_message` 阶段伪造一条“用户说了任务完成”的群聊记录。

但 ContextAware 当前还有一个懒初始化分支：如果该 UMO 的 ContextAware session 不存在，它会在 `on_llm_request` 中主动提取当前事件，At-only 事件可能被记录为一条内部占位消息。它还会在 `on_message` 阶段识别到 `/reset` 或 `/new` 后立即清理自己的 session，早于 Core 的权限和命令结果判断；因此一次无权限 reset 也可能让 ContextAware session 暂时为空。

为规避这一点，提交 synthetic event 前通过公开 `Context.get_registered_star("astrbot_plugin_context_aware")` 做 optional gate：

- 插件不存在、未激活，当前 platform 不在其 `support_platforms`，或当前是私聊且其公开 config `only_group_chat=true` 时，不需要 ContextAware gate。
- 插件已激活、当前 platform 受其支持且其配置会处理当前消息类型时，feature-detect `metadata.star_cls.has_session(umo)`。
- 只有 `has_session(umo)=true` 才允许提交 Agent 合成事件。
- API 缺失、调用异常或返回 false 时，使用确定性主动通知，不触发 ContextAware 懒初始化；不得导入它的私有模块、读写 `_sessions` 或修改其代码。

因此本设计只对“原任务仍处于同一进程、同一 epoch、同一 conversation，且受支持平台上的 ContextAware session 仍存在”的正常终态使用 Agent 合成事件；`/reset`、`/new`、无权限 reset 导致 ContextAware session 被提前清空、进程重启恢复、conversation 漂移和 API 不确定路径均使用确定性主动通知。

ContextAware 的 `on_llm_request(priority=-10)` 仍会读取该 UMO 已有的真实群聊快照并注入场景。Gitee 随后以 `priority=-20` 注入任务终态。

ContextAware 的 `on_llm_response` 会把主 Agent 的自然完成回复记录为 Bot 发言。内部事件必须使用原 sender ID 和 sender name，否则 ContextAware 会把回复对象错误记录为 Scheduler 或群 ID。

图片由后台 worker 直接发送，不会自动进入 ContextAware 的消息记录，因此 task record 中的 `image_sent=true/false` 是必需事实源。

ContextAware 在 `on_llm_response` 阶段记录 Bot 文本，早于平台实际发送。若随后平台发送失败，ContextAware 可能短暂保留一条用户实际没收到的 Agent 回复；插件不能在不修改 ContextAware 的前提下回滚这条私有记录。Gitee 账本中的 `notification_state` 和 `delivery_state` 必须作为权威事实，并在后续 LLM 请求中明确注入实际 fallback 状态。这是插件-only 边界下需要保留并监控的兼容性残余风险。

## 11. 取消与会话命令

### 11.1 `/stop`

`active_event_registry` 是 Core 进程内私有集合，只为真实 pipeline event 自动注册，既不持久化也没有后台 task ID 接口。首版不向其中手工塞入长期 control event，避免插件热重载后的悬空 event 和版本漂移。

插件使用两条公开路径：

- 平台消息 Hook 识别 `/stop`，立即在 SQLite 中为当前 UMO/scope 的未终态 parent 写 cancellation tombstone，并取消对应 planner/worker task。
- `on_decorating_result` 根据 event extra 中的取消数量修正最终命令回复，避免内置 `/stop` 只统计 Core event 后误报“没有运行任务”。
- 提供 `/停止生图 [task_id|全部]` 插件命令作为不依赖内置命令文本和 cmdmask 的稳定入口，并校验 sender/scope 权限。

如果未来 AstrBot 改变 `/stop` 命令路由或 decoration Hook，能力检测失败时仍保留 `/停止生图`，同时后台模式记录 warning；不得退回伪注册 active event。

批量 parent 只保存一个 cancellation tombstone。`/stop` 取消 parent 后，planner、所有 running child 和尚未调度的 child 一并取消；已经 `delivery_state=confirmed` 的图片不能撤回，parent 以 `partial + terminal_reason=cancelled` 收敛，并如实说明已经发出多少张、剩余多少张已停止。

### 11.2 `/reset` 和 `/new`

首版采用明确、可预测的策略：取消当前 UMO 下尚未完成的生图任务，禁止任务在清空或新建会话后突然发送幽灵图片。

处理顺序：

1. 高优先级消息 Hook 识别 reset/new，记录命令 event ID、原 conversation ID，并取得该 UMO 的 send gate 后建立 barrier；正在发送的图片先完成当前 attempt，之后不允许新的图片进入 sending。
2. 内置 reset/new 正常执行权限检查和会话操作。
3. `on_decorating_result` 通过 `_clean_group_context_session=true` 确认命令操作已经成功；回复文本是否最终发出不影响 reset/new 的真实状态。
4. 只有确认成功后，reset epoch 才加一并持久化，同时设置任务取消信号。
5. decoration Hook 在事务提交后释放 send gate；worker 取得 gate 后再次比较 epoch、owner 和 tombstone。
6. epoch 不一致时禁止图片发送，任务进入 cancelled。
7. reset/new 对应的任务终态不再提交 Agent 合成事件，而是主动发送一条自然、确定性的取消说明，避免向已经清空或新建的 ContextAware session 注入合成消息。

命令被拒绝、执行失败、pipeline 提前终止或 barrier 超时无法确认 marker 时，只解除 barrier，原任务继续运行。barrier 必须有独立 finalizer，确保异常路径释放 send gate。这样堵住“命令成功后才开始发送”的幽灵图片窗口，也不会因一次无权限 reset 误取消任务。

reset/new 是用户主动要求清空或切换上下文，因此这两个路径只保证用户收到取消说明，不再要求新上下文记住旧任务的详细提示词；强行注入反而违背 reset/new 语义。正常完成、失败和 `/stop` 取消仍会进入原 conversation 与 ContextAware。

即使 provider 不支持真正取消，上游请求晚些返回，也必须因为 epoch/tombstone 检查而丢弃结果，不能再发图。

reset/new barrier 必须在 batch 每张图片发送前重新检查，而不是只在整批开始发送前检查。这样命令在第 2 张和第 3 张之间成功时，已经发出的图片保留事实记录，剩余图片立即停止，禁止继续往新会话里灌图。

残余窗口必须写入验收结论：如果图片平台 await 已经开始，reset/new 无法让远端 API 回滚；send gate 只能保证命令操作与“尚未开始的发送”互斥。此时按实际返回记 confirmed 或 unknown，不得伪装成完全可撤销。

### 11.3 插件重载和 AstrBot 重启

- 热重载：terminate 将任务标记 interrupted，新实例 initialize 后恢复通知。
- 正常重启：启动时根据 SQLite 非终态记录标记 interrupted 并发送确定性通知，不重入 ContextAware pipeline。
- 进程被强制 kill：无法在退出前写状态；启动时将残留 active 记录标记 interrupted。
- 不尝试恢复不可重放的 provider 请求，避免重复扣费和重复发图。
- 如果残留状态为 sending 且没有 confirmed receipt，通知必须说明“发送结果未知，为避免重复不会自动重发”，不能笼统声称用户一定没收到。

## 12. 图片发送与通知顺序

成功路径必须严格执行：

```text
provider 返回图片
  -> image_generated=true
  -> state=sending
  -> SQLite 事务写 delivery_state=attempting 和 send_attempt_id
  -> 取得 UMO send gate，重新定位当前 adapter
  -> 重新校验 owner lease/epoch/cancel/send barrier
  -> await image-only send attempt
  -> adapter await 成功返回
  -> SQLite 单事务写 receipt、image_sent=true、delivery_state=confirmed
  -> image_sent=true
  -> state=completed
  -> 同事务持久化 completed tombstone、释放 reservation、写 notification outbox
  -> 提交自然完成通知事件
```

不能先通知“画好了”再发送图片。

平台在发起网络请求前明确拒绝，或返回明确失败 payload：

```text
image_generated=true
image_sent=false
state=failed
error_code=image_send_failed
```

Agent 应自然说明“图生成出来了，但发送失败”，不得声称用户已经收到。

平台调用发生 timeout、connection reset，或进程在 await 返回前后退出时，不能确定远端是否已经接受：

```text
image_generated=true
image_sent=false
state=interrupted
delivery_state=unknown
error_code=image_send_unknown
```

该状态禁止自动重发图片。Bot 只能说明“发送结果不确定，为避免重复没有自动重发”，用户明确要求后可通过新的任务或现有“重发图片”能力再次发送。

图片发送使用根据 `TaskDeliveryTarget` 和当前 adapter 新建但不提交 pipeline 的平台 event，从而复用现有文件、bytes 和压缩能力。禁止继续保存和调用原始用户 event，也禁止保存旧 adapter/client 引用跨越热重载。现有 `_send_image_with_fallback()` 必须拆分 failure taxonomy：只有本地编码、文件类型或远端明确拒绝且确认未接受时才能切换 fallback；timeout/断线一律返回 unknown，不得继续第 2 至第 5 次跨网络重试。

`weixin_oc` 的 text+media 发送会拆成两个 API 调用，并可能只成功一部分。首版图片 attempt 必须发送 image-only chain；人格化文本由后续 notification 独立发送和记账，禁止把前置 caption 与图片合成一个布尔结果。

batch 的图片 provider 阶段并发执行，但发送阶段在所有 child 生成状态收敛后按原始 index 顺序逐张发送，跳过生成失败项。这样不会出现第 4 张、第 1 张、第 3 张乱序到达。每张发送前都重新校验 owner、epoch、parent cancellation 和 send barrier；一张明确发送失败不阻止后续成功图片尝试发送。任何 child 为 unknown 时 parent 必须为 interrupted，并在最终摘要中分别给出 confirmed、failed、cancelled、unknown 数量。

receipt 与 task tombstone 写入同一个 SQLite 事务，不再维护第二套 receipt JSON 文件。网络发送本身不可能被包含在本地事务中：发送前先持久化 attempting，成功返回后再提交 confirmed receipt；进程若在两次事务之间退出，恢复状态只能是 `delivery_state=unknown`。这避免本地双账不一致，但无法消除平台无 idempotency key 的外部窗口。

## 13. 幂等与重复通知

图片平台通常不提供通用 idempotency key，因此只能实现插件侧尽力幂等：

- 每个任务只有一个 `notification_token`。
- batch parent 只有一个 notification token，但每个 child 有独立 send_attempt_id 和 image receipt。
- 终态和 notification outbox 在同一事务中持久化，再提交内部事件。
- notification 状态只允许 `pending -> queued -> agent_sending|fallback_sending -> sent|fallback_sent|unknown`。
- `after_message_sent` 必须校验 task_id 和 token。
- `after_message_sent` 还必须校验 `_has_send_oper is True`，并在 SQLite 单事务中写 notification receipt 和 sent 状态。
- 相同 token 的第二个内部事件在进入 Agent 前停止。
- 图片发送开始前写 `state=sending`，成功后立即写 completed tombstone。
- worker 在 `image_generated=false` 时才允许 provider 链内部重试；一旦已有生成结果，绝不能为了发送失败重新调用 provider 或重新创建 task ID。平台 fallback 只允许发生在可以证明远端未接受的本地/明确拒绝错误，timeout 和断线直接 unknown。
- `aiocqhttp` 当前丢弃 OneBot send 返回值，`weixin_oc` 使用本地随机 message ID；receipt 因此保存 adapter、attempt ID、response digest 和时间，不伪造平台真实 message ID。

notification outbox 使用事务 claim：synthetic event 在 `on_decorating_result` 中只能把 `queued` CAS 为 `agent_sending`；watchdog 只能把仍为 `queued` 的 token CAS 为 `fallback_sending`。任一方 claim 后，另一方必须停止发送。若平台发送成功后进程在 SQLite commit 前崩溃，恢复时只能标记 unknown，不能自动重放 Agent 通知。

残余风险：消息平台可能已经接受图片或文本，但网络在确认返回前断开。首版遇到这种错误直接记 unknown 而不是自动 fallback，可优先避免重复；但无法同时保证“不重复”和“绝不漏发”。没有平台级 idempotency API 时只能二选一，本设计选择不自动重复发送。

## 14. 通知 watchdog

接单确认由原事件的 `on_decorating_result` 补齐非空 Plain，并由 `after_message_sent` 收敛，不使用独立定时 watchdog，避免固定接单话和迟到的主 Agent 回复重复。若原 Agent 连 decoration 阶段都未进入，通常说明该次 LLM/pipeline 本身仍卡住，此时插件不能在不制造竞态的前提下保证额外接单消息。终态内部事件提交后启动 watchdog：

- 默认等待 90 秒。
- 如果 `notification_state=sent|fallback_sent|unknown`，结束。
- 90 秒后只有仍为 `queued` 的 token 才能事务 claim 为 `fallback_sending`；已进入 `agent_sending` 的 token 不得并发 fallback，而是在平台超时后由原发送路径收敛为 sent 或 unknown。
- fallback claim 成功后重新定位当前 adapter，并使用 `Context.send_message(umo, MessageChain)` 发送确定性短文本。
- fallback 文本包含成功/失败/取消事实，但不包含内部 task ID、堆栈和敏感错误。
- fallback await 成功返回后写 `notification_state=fallback_sent`；timeout/断线写 unknown，禁止自动再发。

示例兜底：

- 成功：`刚才那张图已经画好并发出来了。`
- 失败：`刚才那次生图没成功，任务已经结束了。`
- 取消：`刚才那次生图已经停下来了。`
- 中断：`刚才的生图任务因为服务重载中断了，没有继续发送。`
- 批量完成：`刚才那组图已经处理完了，计划 4 张，实际发出了 4 张。`
- 批量部分成功：`刚才那组图处理完了，4 张里发出了 3 张，另外 1 张没成功。`

正常路径仍由主 Agent 按人格生成，以上文本只用于防止静默。

watchdog 的 fallback 发送成功后必须把实际发送文本和 transport=`fallback` 写入账本，供后续 `on_llm_request` 注入。进程重启时如果 notification 处于 `agent_sending|fallback_sending` 且没有 receipt，不自动重放 Agent 事件；将状态标为 unknown。是否再发送“上次通知结果不确定”的恢复说明也必须经过独立 recovery token，且文案不能重复宣称成功或失败。

## 15. 能力检测和同步降级

后台模式启用前检查：

- 当前 Agent runner 是 AstrBot Internal Agent；第三方 runner 未通过同等 pipeline/Tool 行为验证时同步降级。
- AstrBot 版本和 `ProviderRequest` 可用性。
- 当前 platform adapter 能否按 platform instance ID 定位。
- adapter 是否实现 `create_event()` 和 `commit_event()`。
- `on_decorating_result` 和 `after_message_sent` Hook 是否可用，且非 streaming result 能读取最终 chain。
- event 能否保存 extra 和绑定 provider_request。
- event 是否存在 `_has_send_oper`，且受支持 adapter 的 `send()` 成功路径会在真实平台 await 后设置它。
- `after_message_sent` 时是否仍能读取最终 `MessageEventResult` chain，用于确认存在非空 Plain 回复。
- 当前事件能否获得 self ID、sender、message type 和 UMO。
- 当前请求能否获得稳定的 conversation ID；没有 cid 时不启动后台任务。
- batch planner 能否按 UMO 取得聊天 provider，并支持 timeout/cancellation。
- SQLite 数据库能否创建、执行 transaction、取得唯一 owner lease 并通过 quick check。
- 插件数据目录是否可原子写入 input manifest，且不位于不支持 SQLite WAL/locking 的网络文件系统。
- `weixin_oc` 当前 context token 是否可用；缺失时该次请求同步降级或明确拒绝后台主动投递。
- ContextAware 若已启用、支持当前 platform 且配置会处理当前消息类型，能否通过公开 metadata config 和 instance 调用 `has_session(umo)`；不能确认 session 存在时只禁用 synthetic completion，不影响后台生图本身。

任一关键能力缺失时：

- 记录一次明确 warning。
- 当前请求退回原同步 `aiimg_generate` 行为。
- 不创建半残的后台 task。
- 不修改 AstrBot 或运行时 monkey-patch Core。

后台模式启用后每 5 分钟输出一条不含 prompt 的 health summary：owner epoch、managed task 数、active parent、reservation remaining、running provider、ready child、notification queued/sending/unknown、oldest active age、DB/WAL bytes、GC failure count 和最近一次 heartbeat。以下情况立即 warning 并在连续 3 个周期不恢复时 fail closed 停止接单：owner heartbeat 落后、reservation 与 active child 不一致、outbox 达到 80% hard cap、WAL 持续增长且 checkpoint 失败、managed task 数超过理论上限或存在未消费 task exception。

配置新增一个功能组：

```text
features.background_llm_image.enabled
features.background_llm_image.max_running
features.background_llm_image.max_queued
```

灰度版本 `enabled` 默认 `false`，京东云验证通过后显式开启。`max_running` 默认 2，`max_queued` 默认 16，分别限制真实图片 provider 并发和预留 child 总量；这两个值是稳定性必需的背压，不是装饰性配置。批量数量继续服从现有 `features.batch.max_count`，planner 并发首版固定为 1。超时、TTL 和注入上限首版使用代码常量，避免配置项堆成仪表盘。

## 16. 群聊和多用户隔离

后台任务的所有缓存、并发额度、最后图片和 follow-up 元数据统一使用 scope key：

```text
(umo, self_id, sender_id, conversation_id)
```

必须同步修正 `_last_image_by_user` 只按 sender ID 的现状，防止同一用户在 A 群生成后到 B 群执行 `/重发图片` 时串图。

群聊注入策略：

- 当前 sender 的任务显示详细 prompt。
- 其他 sender 的任务只显示昵称、状态和耗时。
- 完成通知使用原 requester 的 sender ID/name，使 ContextAware 正确记录 Bot 在回复谁。
- 同一 UMO 的任务通知先经过插件 notification dispatch queue，再进入 Core LLM session lock，禁止多个终态事件并发覆盖会话历史。

## 17. 首版实施范围

包含：

- `aiimg_generate`。
- `aiimg_batch_generate`，包括 planner、child 并发生成、按序发送和 parent 终态通知。
- 兼容入口 `gitee_draw_image` 和 `gitee_edit_image`，因为它们最终调用 `aiimg_generate`。
- text、edit、selfie_ref 三种路由。
- 单图、多个独立单图 parent，以及一个 batch parent 下的多 child 并发。
- preparing、planning、queued、running、generated、sending、completed、partial、failed、cancelled、interrupted 状态，以及 Tool 层 `accepted` 回执。
- 普通对话状态注入。
- 正常同会话终态的 Agent 自然通知，以及 reset/new/restart 的确定性自然措辞通知。
- `/stop`、`/reset`、`/new`。
- 插件热重载和进程重启恢复。
- `aiocqhttp` 和 `weixin_oc`。

暂不包含：

- 视频后台链路重构。
- 直接命令 `/文生图`、`/改图`、`/自拍` 的后台化。
- 直接命令 `/批量` 的后台化；本期后台批量特指 LLM Tool `aiimg_batch_generate`。
- provider 不提供的精确百分比进度。
- AstrBot Core 通用后台任务框架。

视频和直接命令应在单图与 LLM 批量路径通过京东云灰度后复用同一任务管理器扩展。

## 18. 文件级实施清单

只修改 `astrbot_plugin_gitee_aiimg`：

```text
main.py
  - 初始化 BackgroundImageTaskManager
  - 重构 aiimg_generate 为快速预处理 + 后台执行
  - 重构 aiimg_batch_generate 为快速接单 + 后台 planner + child 调度
  - 新增 aiimg_task_status 只读 Tool
  - 新增 on_astrbot_loaded / on_llm_request / message / on_decorating_result / after_message_sent Hook
  - 调整 terminate 顺序
  - 将 last image key 改为 scope key

core/background_tasks.py
  - 新增 single/batch parent、BatchItemRecord、PreparedImageJob、PreparedBatchJob、输入 spool、状态机、child 容量预留、公平队列、SQLite、owner lease、取消、合成事件、receipt、outbox 和 watchdog

core/batch_executor.py
  - 保留直接命令兼容；LLM 后台 batch 不再使用独立 semaphore 的一次性 gather，改由任务管理器共享全局 provider 调度和逐 item 状态回写

_conf_schema.json
  - 新增 features.background_llm_image.enabled / max_running / max_queued

metadata.yaml
  - 版本号升级

README.md
  - 增加后台生图行为、状态查询和取消说明

CHANGELOG.md
  - 记录行为变化和兼容边界

tests/test_background_task_manager.py
  - 状态机、SQLite transaction、lease、恢复、取消和幂等测试

tests/test_background_pipeline_event.py
  - 内部事件、Hook 顺序、ContextAware 兼容和 fallback 测试
```

禁止修改：

```text
/Users/lifeilong/Projects/astrbot-dev/AstrBot/**
/Users/lifeilong/Projects/astrbot_plugin_context_aware/**
京东云 AstrBot Core 文件和容器镜像
```

## 19. 必要测试

### 19.1 单元测试

1. Tool 在 provider 尚未完成时已经返回 accepted。
2. text/edit/selfie 的 effective prompt 在 provider 调用前进入账本。
3. 输入图片在原事件清理后仍可使用。
4. worker 不持有原始 AstrMessageEvent，只使用 PreparedImageJob 和 TaskDeliveryTarget。
5. 状态迁移拒绝终态回退；single 到 queued 才能 accepted，batch 必须完成输入固化、容量预留和 planning record 持久化才能 accepted。
6. SQLite schema migration、索引、quick check、TTL/GC 和损坏数据库 fail-closed 正确。
7. provider semaphore 和 max_queued 能阻止无界后台任务。
8. completed 必须要求 image_sent=true 和 delivery_state=confirmed。
9. 失败错误经过脱敏和截断。
10. 同一 sender 在两个群的 task、last image 和并发额度互不串扰。
11. `/stop` 取消 provider task，且不再发送图片。
12. 无权限 `/reset` 不取消任务；成功 `/reset`、`/new` 会解析 barrier、推进 epoch 并丢弃晚到结果。
13. terminate 和 `CancelledError` 后所有 task 收敛，reservation remaining 与 inflight 归零，旧 owner epoch 不能继续写状态或发送。
14. 重启恢复把 preparing/planning/queued/running 和未发送 generated child 标为 interrupted，把 sending child 标为 delivery unknown；recovery token 在本地只 claim 一次，平台歧义不宣称 exactly-once。
15. 相同 notification token 在 SQLite CAS 下不会被 Agent 与 watchdog 同时 claim。
16. `after_message_sent` 在 `_has_send_oper=false` 或最终 chain 没有非空 Plain 时不得误标 sent。
17. decoration Hook 对空 ack chain 只补一次 deterministic Plain；after hook 只在 digest 与 `_has_send_oper` 同时匹配时确认。
18. 图片 send attempt 在 confirmed transaction 前崩溃时恢复为 unknown，禁止重新调用 provider 或自动重发图片。
19. auto-selfie fallback 时 attempts 和当前 effective prompt 与真实调用一致。
20. batch Tool 在 planner 尚未调用时已经返回 accepted，初始状态为 planning。
21. batch requested_count 在接单前原子预留 child capacity，容量不足时整批拒绝且不泄漏额度。
22. planner 使用任务 UMO 对应的 provider，规划失败时 parent failed 且不启动任何图片 provider。
23. planner 完成后每个 child 都有独立 effective prompt、比例和状态。
24. single 与 batch child 共享 max_running，batch 不会绕过全局并发限制。
25. parent 级 round-robin 和 per-task concurrency 同时生效，单图任务能够在批量运行期间获得下一个可用 provider 槽。
26. batch 生成并发但按 index 顺序发送，单个发送失败不阻止后续成功 child。
27. batch parent completed/partial/failed 只按实际 confirmed send 汇总，不按生成成功数冒充已发送数。
28. `/stop`、`/reset`、`/new` 能取消 planner、running child 和未调度 child；发送中取消只停止剩余图片。
29. 每个 batch child receipt 和 attempting tombstone 能阻止重启后自动重复发图；parent notification 只能被一个本地 sender claim。
30. `aiimg_task_status` 分页返回当前 sender/scope 的完整 child prompts，limit 上限生效且不能越权读取他人提示词。
31. live owner lease 存在时第二个 manager fail closed；lease 过期接管后旧 owner epoch 的 UPDATE 全部失败。
32. batch planner 失败、child 取消、重复 terminal callback 和 `CancelledError` 都不会 double-release 或泄漏 reservation。
33. aiocqhttp/weixin_oc timeout 或 connection reset 一律进入 unknown 且不触发跨 attempt retry；明确拒绝才允许 safe fallback。
34. weixin_oc 图片发送使用 image-only chain，图片和终态文本拥有独立 receipt，不会把 partial media 当整条成功。
35. task record、input、terminal rows 和 outbox 达到 byte/count hard cap 时拒绝新任务，GC 删除失败可重试且不泄漏内存 task/watchdog/barrier。

### 19.2 Pipeline 集成测试

1. 群聊内部事件通过 At(self_id) 进入正常 Agent pipeline。
2. 私聊内部事件进入正常 Agent pipeline。
3. 内部事件存在 provider_request 时允许空 Plain 内容。
4. 内部事件在 session lock 内执行。
5. ContextAware `priority=-10` 与 Gitee `priority=-20` 的内容同时存在。
6. ContextAware `on_message` 忽略只含 At 的内部事件。
7. ContextAware 把自然完成回复记录为 Bot 回复，并关联原 requester。
8. 完成事件中 Gitee 生图 Tool 被移除，不能递归调用。
9. 同一 UMO 用户在任务完成瞬间发送消息，历史不丢失、不覆盖。
10. completion event 仍为 queued 时 watchdog 能 CAS claim fallback；已经 agent_sending 的 event 不会并发 fallback。
11. ContextAware session 已存在时，`on_llm_request` 不触发空 session 懒初始化，也不新增 synthetic 用户消息。
12. reset/new/restart、无权限 reset 已清空 ContextAware session、`has_session=false` 和 API 异常路径不提交合成 Agent 事件，不触发 ContextAware 空 session 懒初始化。
13. completion Agent 生成文本但平台发送 timeout/断线时，账本进入 notification unknown，ContextAware 残余 Bot 记录由后续状态注入纠正，不自动重复文本。
14. AstrBot conversation history 只保存通用 internal lifecycle marker 和自然 Agent 回复，不保存完整 task JSON。
15. batch parent 只产生一次终态 Agent 回复，不为每个 child 重复唤醒 Agent。
16. batch partial 状态在 ContextAware 和 Gitee 注入中同时可见，Agent 准确说明计划数、发送数和失败数。

### 19.3 京东云真实 QQ 验收

每个场景使用唯一测试文本和日志时间窗：

1. 私聊启动一张预计超过 2 分钟的图，确认 Bot 立即自然接单。
2. 任务运行时连续发送两轮普通对话，确认均可正常回复。
3. 询问“现在画到哪了”和“详细提示词是什么”，确认回答与账本一致。
4. 图片完成后确认先收到图片，再收到人格化完成回应。
5. 制造 provider 失败，确认 Bot 主动说明失败。
6. 运行中执行 `/stop`，确认 provider task 取消、无晚到图片，并收到取消说明。
7. 普通群成员执行一个会被权限拒绝的 `/reset`，确认任务没有被误取消。
8. 同一无权限 `/reset` 已让 ContextAware session 清空时，任务完成走确定性主动通知，不创建 synthetic 占位消息；用户再发一条真实消息恢复 session 后，后续任务可重新使用人格化 completion。
9. 管理员运行中执行 `/reset`，确认无幽灵图片污染重置后的会话。
10. 运行中执行 `/new`，确认无旧任务写入新 conversation。
11. 运行中热重载插件，确认重载后主动说明 interrupted。
12. 运行中重启 AstrBot，确认启动恢复通知且不重复发图；sending 阶段恢复不得谎报一定成功或失败。
13. 群聊由两个用户同时提交任务，确认任务和提示词不串人。
14. 任务完成瞬间发送新消息，确认 conversation history 两边都保留。
15. ContextAware 日志显示正常完成事件有场景注入，reset/new/restart 没有 synthetic 用户消息污染。
16. 人为让 Agent completion 在入队前失败，确认 watchdog fallback；让平台发送在 await 中 timeout，确认 ContextAware 残余记录被 Gitee 权威状态覆盖且 notification=unknown、不盲目重发。
17. 容器日志无未处理 task exception、session lock 死锁、无界队列和重复通知。
18. 发起 4 张批量自拍，确认 Tool 在 planner 前接单，聊天不中断，随后 4 个 child 按 max_running 并发生成。
19. 批量运行中询问“每张分别是什么提示词”，确认 planning 阶段不编造，规划完成后能返回全部 child effective prompt。
20. 同时提交一个 8 张 batch 和一个单图任务，确认 batch 不长期独占所有 provider 槽，单图可以获得执行机会。
21. 制造 4 张中 1 张生成失败、1 张发送失败，确认成功图片按原顺序发出，parent 最终说明实际收到数量而不是生成数量。
22. 批量发送到一半执行 `/stop`、`/reset`、`/new`，确认已发图片不重复，剩余图片不再发送，终态数量准确。
23. 运行中热重载 platform adapter，确认 worker 不使用旧 event/client，无法重新定位时任务进入 unknown/interrupted。
24. 模拟第二个插件 owner 启动，确认后台模式 fail closed 而原同步生图仍可用。
25. 连续 72 小时灰度，至少覆盖 20 个单图、5 个 batch、2 次 provider 失败、2 次平台断线、1 次插件热重载和 1 次 AstrBot 重启；期间 DB quick check 正常、WAL 可回收、active reservation 最终归零、outbox 无持续堆积、无未处理 task exception。

## 20. 部署步骤

1. 在本地插件仓库实现并运行全部测试。
2. 执行 Python 编译、Ruff 和现有测试集。
3. 保持后台功能默认关闭，先部署代码。
4. 停止后台接单并等待 active task 收敛后，使用 SQLite backup API 备份账本，再备份京东云插件目录、配置和插件数据目录；不能只复制正在使用的主 DB 文件而漏掉 WAL。
5. 只替换 Gitee 插件文件，不改 AstrBot Core。
6. 热重载插件，验证原同步行为未回归。
7. 显式开启 `features.background_llm_image.enabled`。
8. 按真实 QQ 验收清单逐项测试。
9. 先按一个私聊和一个测试群灰度，完成全部真实验收后再扩大范围。
10. 连续观察至少 72 小时，记录 task 总数、active/reservation、outbox、unknown、DB/WAL 大小、provider latency、event loop lag 和未处理异常。
11. 功能开关关闭时 manager 先停止接单并收敛已有 worker；出现异常可退回同步路径，不需要回滚 AstrBot，但不得直接遗弃 active task。

## 21. 设计审查

### 21.1 已发现的高风险点及处理

| 风险 | 严重度 | 处理 |
| --- | --- | --- |
| 直接使用原生 background Tool 会丢插件后台标记 | P0 | 不使用该路径 |
| 完成通知绕过 on_llm_request，ContextAware 缺席 | P0 | 使用正常 pipeline 合成事件 |
| 后台任务与用户消息并发覆盖历史 | P0 | 完成通知必须进入正常 session lock |
| 原事件临时图片在 Tool 返回后被清理 | P0 | 返回 accepted 前固化输入图片 |
| AstrBot 插件 KV 多 key 写入无法原子维护 task/reservation/outbox | P0 | 使用插件数据目录 SQLite 和单事务状态迁移 |
| 两个插件实例同时执行导致重复 provider 调用或重复发图 | P0 | SQLite owner lease + owner_epoch fencing；不同数据目录 active-active 明确不支持 |
| `/stop`、`/reset`、`/new` 管不到 detached task | P0 | 插件 task registry + cancellation tombstone + public message/decoration Hook + send gate |
| 图片发出但 Bot 不说话 | P0 | Agent 通知 + 90 秒 watchdog fallback |
| Tool 已接单但当前 Agent 没发接单话 | P0 | ack token + decoration 阶段补 Plain + after hook 确认 |
| 失败、取消、重载静默 | P0 | 所有终态进入同一通知状态机 |
| CancelledError 跳过终态写入和 capacity release | P0 | shield 幂等 terminal transaction + gather(return_exceptions=True) |
| 无权限 reset 被插件提前当成成功，误取消任务 | P0 | 前置 send barrier，decoration 阶段只认 `_clean_group_context_session` marker |
| provider 任务无界堆积拖垮京东云 | P0 | max_running semaphore + max_queued 有界队列 |
| batch planner 仍在 Tool 内调用聊天 LLM | P0 | batch 从 planning 阶段整体后台化，Tool 在 planner 前返回 |
| batch parent 只占一个槽却并发启动 N 张 | P0 | 按 requested_count 预留 child capacity，single/batch 共享 provider semaphore |
| 批量“生成成功”被误报成“发送成功” | P0 | 每个 child 独立 SendImageResult、delivery_state 和 receipt，parent 只汇总 confirmed send |
| synthetic event 被 ContextAware 当用户消息 | P1 | message chain 只放 At，不放 Plain/Image |
| ContextAware 空 session 在 on_llm_request 懒初始化 synthetic 消息 | P1 | reset/new/restart/conversation 漂移不使用合成 Agent 事件 |
| 无权限 reset 提前清空 ContextAware session | P1 | synthetic 前调用公开 `has_session(umo)`；false/异常时确定性主动通知 |
| synthetic event 回复对象记录错误 | P1 | 保留原 sender ID/name |
| 旧任务污染新 conversation | P1 | 绑定原 cid，并在 reset/new 时取消和校验 epoch |
| 相同用户跨群串图或互相限流 | P1 | 统一 scope key |
| 完成通知递归调用生图 Tool | P1 | Hook 中移除本插件生成 Tool |
| `commit_event()` 入队被误当成通知已发送 | P1 | 终态+outbox 先事务落盘，queued 只代表入队，after hook 才确认 transport accepted |
| Agent 与 watchdog 同时发送通知 | P1 | notification token + SQLite CAS claim + 单向状态迁移 |
| after_message_sent 或 Tool 使用提示被误当作自然回复成功 | P1 | 必须同时校验 `_has_send_oper`、最终 Plain chain 和 receipt |
| 发送阶段崩溃后无法判断图片是否到达 | P1 | delivery_state=unknown，禁止自动重发并如实通知 |
| 后台 worker 持有已结束的原 event | P1 | PreparedImageJob + TaskDeliveryTarget，新建 delivery event |
| platform 热重载后 worker 使用旧 client | P1 | 每次发送前重新定位当前 adapter，旧引用不进入 job |
| weixin_oc text+media 部分发送 | P1 | 图片使用 image-only attempt，终态文本独立 receipt |
| 一个大 batch 长期独占全部 provider 槽 | P1 | parent 级 work-conserving round-robin + per-task child 上限 |
| batch 部分成功却只回复成功或失败二元结果 | P1 | parent partial 状态 + sent/failed/cancelled/unknown 聚合计数 |
| batch 多张完成时连续唤醒 Agent 刷屏 | P1 | child 只回写账本，parent 收敛后仅唤醒一次 |
| 平台提交后断线导致送达结果未知 | P2 | timeout/断线直接 unknown 且不自动重试；接受可能漏发或用户手动重发的残余风险 |
| 不支持的平台运行后台模式 | P2 | capability probe，不通过则同步降级 |
| ContextAware 已记录 Agent 文本但平台发送失败 | P2 | Gitee 账本作为权威事实并在后续请求注入实际状态；接受插件-only 无法回滚其私有记录的残余风险 |

### 21.2 审查结论

本轮源码与设计复审确认：单图和 LLM batch 从 planner 开始后台化、运行期间继续聊天、状态/详细提示词进入 Bot 上下文、图片完成后人格化回应、失败/取消/重启有说明，这些目标都可以在只修改 `astrbot_plugin_gitee_aiimg` 的前提下实现，不需要修改 AstrBot Core 或 ContextAware。

要达到可持续运行而不是演示可用，以下条件全部是发布阻断项：

- Tool 返回前固化所有事件依赖输入。
- 使用 SQLite 事务账本、owner lease 和 owner_epoch，不依赖 Core task ID 或多 key 插件 KV。
- worker 不保存原事件，使用 prepared job、delivery target 和新建平台 event。
- provider 有全局 semaphore 和有界队列，任务槽位持有到终态。
- batch planner 在后台执行，child capacity 按图片数量预留，single 与 batch 共享 provider 调度。
- batch 每张图片独立记录生成、发送和 receipt，parent 支持 completed/partial/failed/cancelled/interrupted。
- 完成通知先写 outbox，再通过支持平台的正常 event queue 进入 pipeline；入队、Agent 生成和 transport accepted 分开记账。
- synthetic event 使用 At 唤醒并保留原 sender/session。
- 任务状态通过 `on_llm_request(priority=-20)` 注入。
- `/stop`、`/reset`、`/new` 和重载均有取消或中断收敛，reset/new 只在 Core success marker 出现后推进 epoch。
- `CancelledError`、planner 失败、partial batch、platform timeout、热重载和 DB 写失败均有明确终态与 reservation release。
- 图片发送与自然通知有明确顺序、transport accepted 校验、receipt、接单 finalizer 和终态 watchdog。
- 默认关闭、能力检测失败同步降级。
- 京东云只运行一个 active AstrBot/Gitee owner，并完成真实 QQ 私聊、群聊、重启、断线、并发 batch 和至少 72 小时灰度观察。

插件-only 无法彻底消除的残余风险：

- QQ/微信当前发送 API 没有被 adapter 暴露为可复用 idempotency key 和端到端 receipt；timeout/崩溃窗口只能在“可能重复”和“可能漏发”之间取舍，本设计选择 unknown 后不自动重发。
- ContextAware 在 `on_llm_response` 阶段先记录 Bot 文本，早于平台发送；后续发送失败时 Gitee 只能注入权威状态纠正，不能在不修改 ContextAware 的前提下事务回滚其私有记录。
- `_has_send_oper`、adapter event 构造和 Core pipeline Hook 都存在版本漂移可能；必须 feature-detect、版本锁定和 fail closed，不能静默继续后台模式。
- provider 请求不可安全跨进程恢复；AstrBot 或插件重启时未完成生图会标记 interrupted，不自动续跑或重复扣费。
- 不同 plugin data 目录的多实例无法共享 SQLite lease，active-active 部署不在首版支持范围内。

截至 `2026-07-31`，`v5.1.2` 已在插件仓库完成 SQLite single-owner 任务管理器、single/batch parent-child、容量与 notification outbox 双重背压、公平调度、输入固化、投递 receipt、通知单调状态机、同 UMO 终态通知串行、ContextAware gate、reset/new send gate、pipeline synthetic completion、重启确定性通知和 5 分钟健康快照；全量自动化回归为 `177 passed`，命令为 `PYTHONPATH=.:tests uvx --from pytest --with pytest-asyncio --with pillow --with httpx --with aiohttp --with aiofiles --with openai --with curl-cffi pytest -q`，Python 编译与固定 Ruff `0.15.22` 检查通过。后台功能仍默认关闭；京东云此前已取得 `v5.1.0 / v5.1.1` 的真实 QQ 单图、失败、取消、双单图并发、2 图 batch、完整提示词注入和 ContextAware 群聊并行证据，但 `v5.1.2` 仍必须重新完成最终版真实 QQ 验收与连续 `72` 小时灰度，不能把旧版本证据偷换成最终版本长期稳定结论。

准确结论是：当前代码具备进入京东云灰度的工程基础，但不能承诺绝对零重复、零漏发或零版本兼容风险。QQ/微信缺少端到端 receipt 与幂等发送键的残余窗口仍按 `unknown` 后不自动重发处理；只有完成第 19.3 节真实验收与第 20 节连续灰度后，才能确认当前部署环境中的长期稳定性。若后续删除 capacity transaction、逐项状态、输入固化、owner fencing、reset/new send gate、unknown 语义、通知 CAS 或 shutdown 收敛中的任一项，都会退化成看似异步、实际容易丢状态、误取消、虚报成功或静默的半成品。
