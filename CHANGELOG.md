# 更新日志

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
