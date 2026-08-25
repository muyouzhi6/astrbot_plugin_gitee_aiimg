# 更新日志

## [v5.1.23] - 2026-08-25

### 修复

- 视频 provider 链路扩展到实际下载和发送阶段：首个 provider 创建成功但媒体传输失败时，会继续尝试下一个配置的 provider。
- 图生视频在 ContextAware 清理临时图片且事件没有原始 CQ 字段时，按当前消息 ID 回查 OneBot 原图或 file_id。
- provider 异常日志补充异常类型和 repr，避免网络异常文本为空时无法诊断。

### 测试

- 新增 provider 下载/发送失败后的链路兜底回归测试。
- 新增当前消息 `get_msg` 回源失效临时图片的回归测试。

## [v5.1.22] - 2026-08-25

### 修复

- 兼容 3365 网关将 `Range: bytes=0-0` 错误返回为 `200` 的情况，改用完整首分块探测后并行下载鉴权视频，避免大文件单连接中断。
- 视频本地发送失败日志补充异常类型和 repr，便于定位空错误文本的网络异常。

### 测试

- 新增零字节 Range 探测返回 `200` 时切换完整分块 `206` 的回归测试。

## [v5.1.21] - 2026-08-25

### 修复

- 图生视频在 OneBot 原始消息以 CQ 字符串或 JSON 字符串提供时，能够回源原始图片，避免 ContextAware 临时文件清理后丢失参考图。
- 视频传输异常日志保留异常类型，避免 `httpx` 空错误文本导致无法诊断。

### 测试

- 新增 CQ 字符串与 JSON 字符串原始图片回源回归测试。

## [v5.1.20] - 2026-08-25

### 修复

- raw OneBot 图片回源成功时不再输出临时路径失败 warning，仅在所有图片候选均不可读时记录 warning。

## [v5.1.19] - 2026-08-25

### 修复

- 视频图生任务在 ContextAware 替换的临时图片路径失效时，回源原始 OneBot 图片 URL 或 `file_id`，避免因临时文件提前清理而跳过参考图。

### 测试

- 新增 raw OneBot 图片回源与文件 ID 解析回归测试。

## [v5.1.18] - 2026-08-25

### 修复

- 新增 3365 New API xAI 兼容视频模板，使用 `/v1/videos/generations`，不再把 3365 视频模型误配为 SD2.0 `/v1/video/generations`。
- 兼容美年达视频网关的合法时长约束；未配置时自动将 `seconds=5` 修正为 `4`，避免创建请求直接返回 400。

## [v5.1.16] - 2026-08-25

### 修复

- 视频任务在 AstrBot 清理消息临时文件前先快照参考图，避免图片转码找不到 `context-aware-compressed-*.jpg`。
- Grok 视频分辨率自动归一化为 API 接受的 `480p`、`720p`、`1080p` 格式。
- 视频 MP4 下载支持保留 `.part` 文件并用 HTTP `Range` 断点续传，降低鉴权 CDN 断流导致整段失败的概率。
- 鉴权视频内容下载保留同源 `Authorization`，跨域重定向时自动移除凭据。

### 测试

- 新增 Grok 分辨率、视频断点续传和消息临时文件快照回归测试。
- 真实 JDCloud `grok视频` 链路验证：任务成功、MP4 完整下载并由 NapCat 发出群视频。

## [v5.1.14] - 2026-08-24

### 新增

- 将原 `sora2_video` 模板泛化为通用 OpenAI Videos `/v1/videos` 异步视频服务商，新增 `openai_video` 别名并继续兼容 `x666_sora2` 旧配置。
- 支持 New API 风格的 `metadata.url` 结果，以及任务完成后通过鉴权 `/v1/videos/{task_id}/content` 下载视频。
- 鉴权视频会先下载为本地文件再发送，跨域重定向时自动移除 `Authorization`，避免把 API Key 转发给第三方 CDN。
- 修正 `grok_video`：按 xAI 官方异步 `/v1/videos/generations` 协议发送 `prompt`、`image.url`、`duration`、`aspect_ratio`、`resolution`，并轮询 `request_id`；不再错误调用 `/v1/chat/completions`。
- 下载器拒绝 JSON/HTML 错误页、空文件和无法识别的视频字节，避免把错误响应伪装成 MP4 发送。

### 测试

- 新增 New API 元数据、鉴权内容下载、OpenAI Videos provider 别名和跨域重定向凭据隔离回归测试。

## [v5.1.13] - 2026-08-17

### 修复

- 普通 LLM 对话不再自动注入后台生图任务的原始 `user_prompt`、`effective_prompt` 和批量子项提示词，避免历史生图内容触发 Provider 内容安全拦截或撑大上下文。
- 后台任务仍保留完整提示词在 SQLite 账本中；普通对话只接收有界状态摘要，完整提示词继续通过 `aiimg_task_status` 按需查询。

### 测试

- 新增回归测试，确保普通对话与终态通知上下文不泄漏原始生图提示词，并验证任务状态摘要保持有界。

## [v5.1.12] - 2026-08-17

### 新增

- 自拍生成会直接读取 `astrbot_plugin_life_scheduler` 已缓存的今日日程和穿搭，并注入最终图片提示词，无缓存时不会触发日程 LLM 生成。
- 生活日程插件未安装、未启用或版本过旧时，自拍链路自动降级，不影响原有图片生成。

### 测试

- 新增生活上下文读取、无插件降级和旧接口不回退生成的自拍提示词回归测试。

## [v5.1.11] - 2026-08-16

### 修复

- 修复 Gemini native HTTP 连接池仍限制为全局 `10`、单主机 `5`，导致批量并发提升到 `20-30` 后其余请求等待连接槽并在 30 秒时集中失败的问题。
- Gemini native 现在允许最多 `30` 个同主机连接，与插件后台 Provider 并发上限保持一致；30 秒超时仅约束真实 socket 建连，连接池等待由总请求超时控制。
- 超时错误不再把任意 `asyncio.TimeoutError` 错报为已经超过完整 Provider 总超时，日志会分别给出总请求和 socket 建连阈值。

### 测试

- 新增 Gemini native 连接池并发、总超时、socket 建连超时和用户可见超时文案回归测试。

## [v5.1.10] - 2026-08-16

### 修复

- 修复 LLM 工具说明仍建议 `2-8` 张，导致用户明确要求 `30` 张时模型擅自把 `count` 缩减为 `8` 的问题。
- `aiimg_batch_generate` 现在会识别当前用户消息中最后一个明确的“数字 + 张”，当模型工具参数与用户原话不一致时以用户数量为准，并继续受 `features.batch.max_count` 的安全上限约束。
- 工具说明和 WebUI 现在明确单批数量允许配置为 `1-32`，用户指定数量时模型不得自行减少、拆分或拒绝。

### 测试

- 新增用户要求 `30` 张但模型传入 `8` 时最终仍按 `30` 接单的回归测试，并校验旧的 `2-8` 工具提示不再存在。

## [v5.1.9] - 2026-08-16

### 改进

- 同步文生图、改图和自拍批量任务的可配置并发上限由 `8` 提升到 `30`。
- LLM 后台图片任务的全局 Provider 并发上限同步提升到 `30`，并在 WebUI 中为相关配置提供 `1-30` 的明确范围。
- 默认并发继续保持 `2`，现有配置不会在升级后被自动放大。

### 测试

- 新增同步批量和后台任务并发上下界回归测试，覆盖 `1` 和 `30` 的运行时裁剪行为。

## [v5.1.8] - 2026-08-14

### 修复

- 默认自拍提示词不再强制专业人像摄影、中性日光白平衡、固定机位或双手不持物，改为优先遵循用户指定的视角、动作、光线、构图与照片风格。
- 拍摄设备约束改为按语境处理：普通手持自拍时不额外生成手机；明确要求对镜自拍、手机入镜或展示设备时允许自然出现，同时避免复制或无故增加其它拍摄设备。
- 用户要求现在明确标记为最高优先级，最终设备规则不得改变用户指定的视角；他拍、定时拍摄、人物手势和手持日常物品不再受到全局限制。

### 测试

- 更新默认和自定义自拍提示词回归测试，覆盖普通手持自拍、明确对镜自拍、恋人视角、手持日常物品、设备去重以及用户要求优先级。

## [v5.1.7] - 2026-08-11

### 修复

- 重写默认自拍提示词，增加清晰锐利、真实肤色、中性白平衡和细节质感要求，避免“生活感”描述诱导暖黄、柔焦和低对比度画面。
- 保留画面外固定机位、双手不持物以及排除手机、镜子和拍摄设备的构图约束，并压缩否定描述，减少提示词语义互相干扰。

### 测试

- 更新默认和自定义自拍前缀回归测试，覆盖画质锚点、白平衡和设备排除约束。

## [v5.1.6] - 2026-08-07

### 修复

- 所有 provider 模板的通用请求超时默认值统一为 `600` 秒，并同步各运行时 fallback；已有 provider 的显式超时、URL、Key、模型和其它配置保持不变。
- `gemini_native` 新增 `max_retries` 配置，默认重试 `2` 次，可设为 `0` 禁用；请求超时、网络错误、限流和服务端错误会按上限重试并轮换 API Key，普通客户端错误不会重试。
- 默认自拍提示词改用“高质量人像摄影作品”和“生活感人像构图”，并固定追加定时固定机位、双手不持物、禁止对镜拍摄及排除手机、镜子和拍摄设备的约束，不再依赖上游人格提示能否完整传入图片 Provider。

### 测试

- 新增 provider 默认值、旧配置保留、Gemini 超时/服务端错误重试与 Key 轮换、不可重试错误和默认/自定义自拍前缀回归测试；完整回归为 `190 passed`，`compileall`、Ruff、格式和配置 JSON 检查通过。

## [v5.1.5] - 2026-08-02

### 修复

- 后台图片终态不再使用固定中文模板兜底；正常同会话仍由 Agent 按当前人格自然回应，模型失败、超时、上下文切换、重启恢复或合成事件入队失败时改为静默结束通知。
- 保留 notification outbox 的事务 claim、90 秒 watchdog 和迟到 Agent 抑制；watchdog 只把未完成通知收敛为 `failed`，不再绕过人格直接向用户发送统计话术。
- Agent completion 没有产生非空自然文本时立即停止该合成事件并收敛通知，避免 decoration 阶段重新补入机械播报。

### 测试

- 新增空 Agent completion、重启恢复和 watchdog 静默收敛回归测试；正常人格化 completion 与通知 claim 竞争测试继续保留，完整回归为 `180 passed`。

## [v5.1.4] - 2026-07-31

### 修复

- JPEG、WebP 和 PNG 的 Pillow 解码与编码现在运行在独立的有界线程池中，不再占用 AstrBot 主 event loop；4K 无损图片保存期间，其他群聊、私聊、owner heartbeat 和后台任务状态更新可以继续调度。
- 图片编码线程池最多使用 `2` 个 worker，并与 SQLite 使用的默认 executor 隔离；批量 Provider 同时返回多张图片时不会再串行冻结全部会话，也不会因 heartbeat 饿死触发 `Background owner lease is not valid`。
- 保持原有 JPEG、WebP、WebP lossless、PNG 和 auto 输出语义、元数据传递及像素一致性，不通过降低图片质量或放宽 owner lease 掩盖阻塞问题。

### 测试

- 新增 event loop 响应性与编码并发上限回归测试；完整测试为 `179 passed`，`compileall`、Ruff、格式和配置 JSON 检查通过。
- 本地双图 `3840x2160`、`WebP lossless effort=100` 实测最大 event loop lag 为 `0.0214s`，两张输出均约 `6.9 MiB`。

## [v5.1.2] - 2026-07-31

### 修复

- 通知 outbox 改为单调状态机，已确认 `sent / unknown / failed / expired` 的终态不再允许被迟到 callback 回退；通知回写同时校验当前 task owner epoch，发送 receipt 也会拒绝跨任务复用同一 attempt ID。
- 同一 UMO 的多个后台任务完成通知现在按 turn 串行，前一条必须先收敛为已发送、失败或未知，下一条才进入 synthetic pipeline，避免并发完成时抢占 ContextAware 会话历史或乱序回应。
- 进程重启恢复强制使用确定性主动通知，不再尝试重入旧 conversation；ContextAware 的公开 `has_session()` 同时兼容同步与异步实现。
- 同步发送降级路径遇到 timeout、connection reset 等歧义错误时立即记录失败并停止跨通道重试，避免平台已接收但响应丢失后重复发图。
- 后台入口新增 adapter transport 能力探测；未完成 notification outbox 增加独立硬上限，达到上限后停止后台接单并自动保留同步路径。

### 可观测性

- 新增每 5 分钟后台健康摘要与 SQLite passive WAL checkpoint，记录 owner epoch、managed task、active/reservation、provider/ready、outbox、最老任务、DB/WAL 和 heartbeat；健康检查连续 3 次失败会 fail closed 停止后台接单。
- 启动日志现在明确区分 `active`、`waiting_for_owner`、`startup_failed` 和 `disabled`，不会把等待旧 owner lease 的短暂阶段误报为配置关闭。

### 测试

- 新增通知终态防回退、owner fencing、receipt 归属、同 UMO 串行、等待超时清理、outbox 背压、健康快照、重启确定性通知、异步 ContextAware gate 和歧义发送不重试测试。
- 完整回归更新为 `177 passed`；Python 编译与固定 Ruff `0.15.22` 检查通过。

## [v5.1.1] - 2026-07-31

### 修复

- 修复 `/stop` 或进程重启恰好发生在图片发送阶段时，单图或批量子任务可能保留为 `attempting` 的问题；取消操作现在使用同一个 SQLite 原子事务读取最新 item/receipt，已确认发送的图片不会被旧状态覆盖，其余歧义发送收敛为 `unknown` 并继续禁止自动重发。
- 修复 Agent 终态回应与 90 秒确定性通知 watchdog 同时接管时可能重复回应的竞争窗口；Agent 回复进入 transport 前会重新确认 outbox claim，失去 claim 的旧回复会停止发送。
- 补全 `grok2api_video` 的 API Key 配置校验，避免错误配置通过启动检查后才在视频生成阶段失败。

### 测试

- 新增后台 batch 两层并发上限、连续 batch 公平调度、乱序生成按 index 发送、部分失败、发送中取消、发送中重启恢复、单图发送超时、防重复终态回应和 AstrBot loaded 恢复派发测试。
- Python `3.10 / 3.11 / 3.12 / 3.13` 完整回归均为 `168 passed`；`compileall`、Ruff 和 AstrBot `v4.16.0` 真实源码 import / Hook smoke 均通过。

## [v5.1.0] - 2026-07-31

### 修复

- 修复即梦 Provider 初始化时传入 `timeout` 会触发 `TypeError`，以及后端读取未定义 `timeout` 的问题；超时现在限制在 `1-3600` 秒。
- 即梦 Provider 现在会正确接收配置中的 `output_format`，不再静默回退为默认 JPEG。
- 清理全仓 Ruff 默认规则发现的未使用 import 和未使用变量，并统一 Ruff 格式。

### 稳定性与兼容性

- 新增 GitHub Actions 矩阵，覆盖 Ubuntu Python `3.10 / 3.11 / 3.12 / 3.13`、Windows Python `3.12` 和 macOS Python `3.12`；实际 AstrBot 运行环境仍遵循 Core 的 Python `>=3.12` 要求。
- CI 固定执行 `compileall`、`ruff check`、`ruff format --check` 和完整 pytest，避免只在部署机器上碰运气。
- 明确 LLM 后台任务与同步 `/批量` 命令的通知差异，以及单进程、本地可写 SQLite、平台、streaming 和跨系统 spool 恢复边界。
- 非优雅重启后旧 owner lease 尚未过期时，AstrBot 可先正常启动；插件会后台重试并在 lease 过期后自动接管。

### 测试

- 新增即梦 `timeout` clamp 与 `output_format` 透传回归测试。
- 增加 AstrBot 最低支持版本 `v4.16.0` 的插件 import / API smoke 验证流程。

## [v5.0.4] - 2026-07-31

### 修复

- 修复 AstrBot 或容器非优雅重启后，旧 owner lease 尚未过期会让后台生图在本次运行中永久禁用的问题。
- owner 冲突时不阻塞 AstrBot 启动，插件会按固定低频在后台重试；lease 过期后自动接管任务账本，并继续执行 `interrupted` 恢复通知。
- owner retry task 纳入插件关闭流程，热重载或停机时会先取消等待任务，避免残留协程。

### 测试

- 新增旧 owner 首次占用、随后释放时自动恢复后台模式的初始化测试。
- 全量回归测试更新为 `154 passed`。

## [v5.0.3] - 2026-07-31

### 修复

- 后台接单与终态回应改为跟踪本次最终 `event.send()` 的成功返回，不再要求下游插件处理后的文本与装饰阶段 digest 完全一致；TTS、文本清洗或语音替换不会再把已发送通知误记为 `unknown`。
- 保留 digest、发送标志和短 hash 诊断日志，不记录回应正文或图片提示词，便于定位平台发送与 outbox 回写问题。

### 测试

- 新增下游 TTS 改写、分段发送部分失败、真实发送异常、Provider 可控失败以及成功 `/new` 后旧任务取消与新会话隔离测试。
- 全量回归测试更新为 `153 passed`。

## [v5.0.2] - 2026-07-31

### 修复

- 提高后台任务 `after_message_sent` 确认钩子的优先级，避免其它高优先级清理插件在 stopped event 上提前终止钩子传播，导致已发送的接单或终态回应只能记为 `unknown`。
- `/stop`、`/reset` 和 `/new` 的后台任务收口现可在清理型插件终止传播前完成持久化。

## [v5.0.1] - 2026-07-31

### 修复

- 修复后台任务存在时将普通 `dict` 追加到 AstrBot `extra_user_content_parts`，导致用户追问任务进度时因缺少 `model_dump_for_context()` 直接中断 Agent 的问题。
- 任务状态与完整提示词现通过 `TextPart.mark_as_temp()` 注入，仅参与当前 LLM 请求，不持久化到会话历史。

### 测试

- 新增动态任务内容必须为 AstrBot `ContentPart` 兼容对象的回归断言。

## [v5.0.0] - 2026-07-30

### 新增

- 新增可选的 LLM 后台生图，`aiimg_generate` 与 `aiimg_batch_generate` 在完成输入固化和容量预留后立即返回，planner、Provider 调用和图片发送均在后台执行。
- 新增 SQLite 事务任务账本、单实例 owner lease、owner epoch fencing、WAL、容量 reservation、发送 receipt、通知 outbox 和重启恢复。
- 新增单图与 batch child 共享的全局有界并发调度；batch 使用 parent round-robin，避免大批量任务长期独占 Provider 槽。
- 新增 `aiimg_task_status` 只读 Tool，可分页查询任务状态、每张图的完整 effective prompt、比例和发送结果。
- 新增临时 LLM 状态注入，Bot 在后续聊天中可以看到任务正在规划、排队、生成或发送，以及图片是否已经发出。
- 新增完成、部分成功、失败、取消和重启中断的主动回应；正常会话通过 synthetic event 重入 AstrBot pipeline，继续兼容 ContextAware 场景注入。

### 稳定性

- 后台 worker 不持有已经结束的 `AstrMessageEvent`，输入图片在 Tool 返回前固化到插件数据目录，发送前重新定位当前 platform adapter。
- 图片使用独立 image-only attempt；timeout、connection reset 和崩溃歧义收敛为 `delivery_state=unknown`，禁止自动重发。
- `/stop` 可取消当前用户的后台任务；成功的 `/reset`、`/new` 使用发送闸门取消旧任务，权限失败不会误取消。
- ContextAware session 不存在、conversation 漂移或重启恢复时不创建 synthetic 用户内容，改用确定性主动通知。
- 后台功能默认关闭；只支持单 AstrBot 进程以及 `aiocqhttp` / `weixin_oc`，流式回复开启时自动回退原同步路径。

### 测试

- 新增任务账本与 pipeline 测试，覆盖 owner lease、容量原子预留、终态保护、公平调度、取消、重启恢复、通知 CAS、输入固化、Tool 提前返回、状态注入、ContextAware gate 和发送确认。
- 全量回归测试 `148 passed`，命令为 `PYTHONPATH=.:tests uvx --from pytest --with pytest-asyncio --with pillow --with httpx --with aiohttp --with aiofiles --with openai --with curl-cffi pytest -q`。

## [v4.3.16] - 2026-07-30

### 修复

- 修复新版 AstrBot `Context` 不存在 `base_config` 时插件静默回退到 `/` 前缀, 导致已配置 `.` 仍会被 `/` 命令触发的问题; 现通过 `Context.get_config()` 读取实际 `wake_prefix`.
- 修复批量出图、动态改图预设和视频的 `RegexFilter` 绕过 AstrBot 唤醒门禁, 导致群聊中未对 Bot 发出的消息也能执行任务的问题.
- 修复动态预设使用无边界子串匹配, 导致其他 Bot 回复的 `图片 + /表情包仓库` 被误识别为 `/表情包` 改图预设的问题.
- 改图、自拍、自拍参考、批量和视频 fallback 现仅接受原始消息链文本段开头的“实际配置前缀 + 完整命令 token”, 不再从普通句子中搜索并执行命令.

### 测试

- 新增 `.` 前缀下对 `/`、`，`、裸关键词、句中嵌入命令和预设名前缀词的误触发回归测试; 全量测试 `124 passed`.

## [v4.3.15] - 2026-07-30

### 新增

- Provider `output_format` 新增 `webp` 和 `webp_lossless`; `webp_lossless` 逐像素无损, 适合把 Gemini 4K PNG 压到 QQ 20MiB 图片门槛以内.
- `png` 现在会执行无损优化重编码; `auto` 继续完整保留上游原始字节, 兼容旧配置.
- 新增 `image_encoding` 高级配置, 可调整 JPEG 质量和色度采样、WebP 质量/压缩强度/编码方法以及 PNG 压缩等级.

### 优化

- JPEG 默认改为 `quality=95`、`4:4:4` 色度采样、渐进式优化, 显著改善彩色文字、细线和高频纹理, 避免旧默认采样造成的细节损失.
- 真实 Meinianda `gemini-3.1-flash-image` `3:4 4K` 样本由 `21.362MiB` PNG 转为 `16.856MiB` Lossless WebP, 解码后像素完全一致; PNG 无损优化仅降至 `21.091MiB`.

### 修复

- 修复 OpenAI Images 返回 URL 时错误地把 `output_format` 传给 `str()`, 导致 URL 图片保存路径抛出 `TypeError` 的问题.

### 测试

- 新增无损像素一致性、JPEG `4:4:4`、WebP 编码、配置 schema 和 OpenAI Images URL 保存回归测试.

## [v4.3.14] - 2026-07-29

### 修复

- 修复 `gpt-image-2` 在 OpenAI Images 和 OpenAI Chat 模板中丢失自适应比例, 最终始终请求或降级为 `1:1` 方图的问题.
- 修复群聊中裸 `绘图` / `aiimg` / `改图` / `aiedit` / `自拍` / `自拍参考` 文本可能被 RegexFilter 或 WakePro 误判为插件命令的问题; 现在必须在原始消息链中包含 AstrBot 当前配置的 `wake_prefix`.

### 优化

- 新增 `gpt-image-2` 专用尺寸映射, 例如 `16:9 4K -> 3840x2160`, `3:4 2K -> 1536x2048`; `4K` 尺寸满足 Meinianda 最长边和总像素限制.
- OpenAI Chat 模板会为 `gpt-image-2` 在请求顶层注入精确 `size`, OpenAI Images 模板直接使用同一映射, 其他模型保持原有输出参数行为.

### 测试

- 全量回归测试 `115 passed`.
- Meinianda 实测 Images API 的 `1280x720` / `720x1280` / `3840x2160` 和 Chat API 的 `1280x720` 均返回精确目标像素.

## [v4.3.13] - 2026-07-29

### 修复

- 修复所有 OpenAI 兼容后端（openai_chat / grok_chat / gemini_openai_chat）的 `resolve_output_intent` 返回 `aspect_ratio` key 但 `generate`/`edit` 签名缺少该参数，导致 TypeError 崩溃的问题。
- 修复仅传 `aspect_ratio` 而无 `size`/`resolution` 时，`_apply_gemini_image_config` 提前 return 导致比例注入被跳过的问题。
- 修复 gemini_flow2api 后端 `resolve_output_intent` 将精确像素尺寸（如 `2048x1152`）错误地作为分辨率标签传入的问题，改为取近似分辨率级别。

### 新增

- 全部图片生成后端支持 `output_format` 配置（`jpeg` / `png` / `auto`），默认 `jpeg`；Gemini 4K PNG 自动转为高质量 JPEG，显著降低文件体积（20MB → 3-5MB），解决 NapCat/OneBot 平台发送大图问题。
- 全部后端（包括 openai_chat、flow2api、grok、grok2api、openai_full_url、modelscope、vertex_ai、gitee_async、jimeng）新增 `resolve_output_intent`，LLM 指定的 `aspect_ratio`/`resolution`/`exact_size` 现可正确透传到所有后端。

### 优化

- `_apply_gemini_image_config` 新增 `aspectRatio` 注入逻辑，覆盖顶层 body、`image_config`、`generationConfig.imageConfig`，支持比例参数传递给 Meinianda/兼容网关。

## [v4.3.11] - 2026-07-26

### 修复

- 修复 LLM 未传 `aspect_ratio` 时, 自拍请求把空比例交给 Meinianda 并回退为 `1:1` 的问题。
- 修复批量 LLM planner 只规划提示词、不规划逐图比例, 导致整组图片无法自主使用不同比例的问题。

### 优化

- 批量 planner 为每张图输出独立 `aspect_ratio`; 用户未固定比例时至少使用两种适合构图的比例。
- 新增 `features.selfie.default_aspect_ratio`, 默认 `3:4`; 用户或 planner 的显式比例仍具有更高优先级。
- 强化单图 LLM tool 描述, 要求模型在用户未指定时按人像或横向构图主动选择比例。

### 测试

- 新增逐图比例解析、整组固定比例、混合比例校验、执行层合并和自拍默认比例回归测试。

## [v4.3.10] - 2026-07-26

### 修复

- 修复 LLM tool 猜测的正方形 `output` 覆盖 prompt 明确比例的问题。
- 修复批量 LLM tool 在提示词规划后可能丢失整组比例和分辨率的问题。

### 优化

- 单图和批量 LLM tool 新增独立 `aspect_ratio` / `resolution` 参数，并兼容旧 `output`。
- 增加 Meinianda Gemini Native 配置说明和最终输出参数日志。

### 测试

- 使用 Meinianda 临时 Key 实测 Gemini Native `16:9 1K` 输出为 `1376x768`。
- 新增 LLM 参数冲突和 `auto` 处理回归测试；全量测试 `101 passed`。

## [v4.3.9] - 2026-07-26

### 修复

- 修复 Gemini Native 在 LLM tool 自然提示词中无法识别比例和分辨率, 导致请求始终沿用 `1:1` 默认值的问题。

### 优化

- 自动识别提示词中的常规比例、`1K` / `2K` / `4K` 和精确尺寸, 并按用户级意图传递给对应 provider。
- 用户只指定比例或分辨率时, 默认精确尺寸会拆分为可补齐的另一维度, 不再覆盖用户参数。

### 测试

- 新增自然提示词解析、路由优先级和 Gemini Native `imageConfig` payload 测试。
- 全量测试 `99 passed`。

## [v4.3.8] - 2026-07-26

### 新增

- 新增统一输出意图解析, 支持精确尺寸 `2048x1152`、分辨率 `4K`、比例 `16:9` 以及组合形式 `16:9 4K`。
- 命令、批量命令和 LLM 工具统一支持末尾输出参数, 只解析提示词末尾最多两个控制 token, 不误吞正文中的比例或分辨率描述。
- 普通单图改图在没有显式比例时自动继承输入图比例; 多图改图和自拍参考照不执行该推断。

### 优化

- 输出参数按“用户参数 > 当前 provider 链路覆盖 > 功能默认值 > 单图输入比例 > provider 默认值”合并。
- 每次 fallback 到下一个 provider 时重新解析输出参数, 避免不同 backend 之间复用不兼容 kwargs。
- Gemini Native 使用 `aspectRatio + imageSize`; Vertex AI Anonymous 传递 `aspectRatio`, 并在模型支持时传递 `imageSize`; 声明固定尺寸集合的 OpenAI Images backend 会把比例与分辨率映射为最合适的像素尺寸。
- 补充 Pillow 运行依赖声明, 与现有图片读取和比例检测逻辑保持一致。

### 测试

- 补充输出解析、provider fallback、严格 kwargs、单图比例推断、自拍禁用推断、Gemini payload 和批量命令解析测试。
- 全量测试 `96 passed`。

## [v4.3.7] - 2026-07-19

### 修复

- 修复魔搭 Images API 已停止支持同步生图后持续返回 400 的问题。
- 为 ModelScope provider 增加异步任务提交、状态轮询、失败详情和超时控制。

### 测试

- 补充 ModelScope 异步请求 Header、任务轮询成功、失败、超时和 provider 路由测试。

## [v4.3.6] - 2026-07-02

### 修复

- 修复 Gemini native 改图 / 自拍链路同时发送 `x-goog-api-key` 与错误 `Authorization: Bearer <api_key>` 导致 401 的问题。

## [v4.3.5] - 2026-06-09

### 修复

- 自拍失败时不再发送任何聊天文本提示, 仅保留原有表情反馈与日志记录。

## [v4.3.4] - 2026-06-08

### 修复

- 修复 AstrBot 4.25.x 下 `/自拍` 可能因 wake prefix 被框架剥离后未进入兜底 handler, 导致无响应的问题。
- 修复 `图片 + /自拍参考 设置` 这类图片在前、命令在后的消息可能不触发参考照设置的问题。

### 测试

- 补充自拍命令兜底测试, 覆盖裸 wake 命令、带前缀命令、已激活 command handler 去重、未唤醒裸文本忽略以及图片前置 `/自拍参考 设置`。

## [v4.3.3] - 2026-05-29

### 修复

- 修复 QQ / OneBot 场景下, 改图预设通过 @ 他人头像作为参考图时可能失效的问题。
- 兼容标准 `At` 消息段, OneBot raw `at` 段和 `[CQ:at,qq=...]` 原始消息。
- 兼容 `@用户 /预设` 这类命令不在首个文本段的触发顺序。

### 测试

- 补充 @ 头像参考图解析测试, 覆盖 raw OneBot at 段, raw CQ at 字符串, 以及忽略 @自己 / @全体。

## [v4.3.2] - 2026-05-25

### 稳定性

- 修复个人微信 `weixin_oc` 发送前优化副本长期残留的问题。
- 优化后的 `weixin_send_*.jpg` 临时副本会在发送流程结束后自动清理。
- 生成新副本前会按数量与时间清理历史残留，避免高频生成图场景下 `Temp` 目录持续堆积。
- 优化副本文件名加入唯一后缀，避免同一源图并发发送时复用同一个临时文件。

### 测试

- 补充 `weixin_oc` 发送临时副本清理与残留清理测试。

## [v4.3.1] - 2026-05-24

### 新增

- 新增个人微信 `weixin_oc` 图片发送前优化配置：
  - `send.weixin_compress_images`
  - `send.weixin_image_max_side`
  - `send.weixin_image_max_size_kb`
  - `send.weixin_api_timeout_seconds`
- 发送图片前会识别当前事件平台；仅当平台为 `weixin_oc` 时，才生成高质量 JPEG 发送副本并调整适配器 API/CDN 上传超时。

### 稳定性

- 降低个人微信发送 4K / 大体积生成图时触发 `upload_to_cdn TimeoutError` 的概率。
- QQ / OneBot 原有图片发送、文件兜底、compact bytes 兜底逻辑保持不变。

### 文档与元数据

- README 补充个人微信发送前处理配置、平台限制和超时排障说明。
- `metadata.yaml` 增加 `weixin_oc` 支持平台提示。

## [v4.3.0] - 2026-04-26

### 新增

- 新增文生图预设能力，支持通过 `features.draw.presets` 配置预设，并使用 `/文生图 预设名 补充提示词` 调用。
- 新增统一批量命令入口，支持：
  - `/批量n aiimg ...`
  - `/批量n 文生图 ...`
  - `/批量n aiedit ...`
  - `/批量n 自拍 ...`
  - `/批量n 改图预设名 ...`
- 新增 `LLM` 批量工具 `aiimg_batch_generate`，支持先规划多条不重复提示词，再一次性批量生成整组图片。
- 批量结果统一改为单张直接发送。
- 批量发送不再额外附带“标题 / 提示词 / 状态 / 失败提示”这类通知文本，只保留原插件自己的表情反馈。

### 配置增强

- 新增 `features.draw.batch_concurrency`，单独控制文生图批量并发。
- 新增 `features.edit.batch_concurrency`，单独控制改图 / 自拍批量并发。
- 新增 `features.batch.max_count`，控制单次批量最大数量，默认 `8`。
- 新增 provider 级 `generate_request_mode` / `edit_request_mode`，支持 `auto`、`stream`、`non_stream`。

### LLM 工具与批量规划

- `aiimg_generate` 继续支持 `auto` / `text` / `edit` / `selfie_ref` 路由。
- `aiimg_batch_generate` 默认批量数量为 `4`，建议范围 `2-8`，最终会被 `features.batch.max_count` 限制。
- 批量规划器会要求 `LLM` 输出 `title`、`prompt`、`variation_focus`，并对数量与去重结果做校验。
- 批量规划更适合同场景、同穿搭、不同姿势 / 角度 / 表情的成组出图需求。

### 兼容性与稳定性

- 修复 `ProviderRegistry` 中新默认 `auto` 覆盖旧 `enable_stream_*` 布尔配置的问题。
- 对齐 `ProviderRegistry` 与 `OpenAIChatImageBackend` 的 `request_mode` 兼容语义：
  - 显式 `stream` / `non_stream` 优先
  - `auto + enable_stream_*` 保留旧配置行为
  - 都没有时才回到默认 `auto`
- 统一 `validate()` warning 文案与运行时实际回退行为，避免“文档一套、执行一套”。
- 对单路径 provider 增加 `request_mode` 忽略提示，减少误解。

### 文档与元数据

- 重写 `README.md`，补充新功能的实际调用方式、批量命令、`LLM tool`、并发配置、平台限制与请求模式说明。
- 为插件补充 `metadata.yaml` 中的 `astrbot_version` 与 `support_platforms` 提示。
- 新增本 `CHANGELOG.md`，开始记录版本更新内容。

## [v4.2.26] - 历史基线版本

- 这是补充 `CHANGELOG` 之前的基线版本号。
- 更早的历史更新尚未回填到本文件。
# v5.1.17

- 增加 3365 SD2.0 视频模板，适配 `/v1/video/generations` 的文生/图生任务、状态轮询和鉴权视频下载。
