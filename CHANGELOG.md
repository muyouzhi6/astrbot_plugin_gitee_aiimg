# 更新日志

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
