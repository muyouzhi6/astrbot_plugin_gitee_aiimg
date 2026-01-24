# 视频功能企划（Grok imagine 视频生成）

## 目标

在现有 `astrbot_plugin_gitee`（文生图 + 改图）基础上新增「参考图 + 提示词」的视频生成功能，对接 `grok-imagine-0.9`（`/v1/chat/completions`），并复用本插件现有的：

- 参考图获取：沿用 `core/utils.py:get_images_from_event`
- 任务状态展示：沿用 `core/emoji_feedback.py` 的表情标注（处理中/成功/失败）
- 预设提示词：逻辑与「改图预设」一致（`预设提示词 + 额外提示词` 拼接），但通过 `/视频` 前缀避免与改图预设命令冲突

同时补齐长耗时任务的稳定性能力：

- 失败自动重试（次数可配置）
- 自定义超时时间（最大 3600 秒）
- 单用户并发保护（避免刷屏/并发耗尽）

## 指令设计

### 1) 基础模式

`/视频 提示词`

- 从当前消息 / 引用消息 / @头像 / 发送者头像中获取一张参考图
- 调用 Grok API 生成视频

### 2) 预设模式（带前缀，避免冲突）

`/视频 预设名 额外提示词`

- `预设名` 来自配置 `video.presets`
- 最终 prompt：
  - 仅预设：`preset_prompt`
  - 预设 + 额外：`preset_prompt, extra_prompt`

### 3) 预设识别规则

- `/视频` 后第一个 token 命中预设名：按预设模式处理
- 否则：整段作为普通提示词

## 配置设计（新增 video 段）

建议新增如下配置项（用于 UI 的 `_conf_schema.json` 与运行时 `config`）：

- `video.enabled`：是否启用视频功能
- `video.server_url`：默认 `https://api.x.ai`（内部补全 `/v1/chat/completions`）
- `video.api_key`：Grok API Key
- `video.model`：默认 `grok-imagine-0.9`
- `video.timeout_seconds`：请求超时（1~3600）
- `video.max_retries`：失败自动重试次数（0~5）
- `video.retry_delay`：基础重试间隔（秒），指数退避 `retry_delay * 2^attempt`
- `video.send_mode`：`auto`/`url`/`file`
  - `url`：优先 `Video.fromURL(video_url)`
  - `file`：下载后 `Video.fromFileSystem(video_path)`
  - `auto`：先 URL，失败再下载并尝试本地文件，仍失败则回退发送纯链接
- `video.max_cached_videos`：仅对 `file/auto` 的下载缓存生效；用于长期运行自动清理
- `video.presets`：列表，元素格式 `"预设名:英文提示词"`

## 代码结构设计

新增核心模块建议拆分为两层（尽量贴合现有 `draw_service/edit_router` 风格）：

1) `core/grok_video_service.py`
   - 负责 API 调用、重试/超时、URL 提取、预设拼接
   - 输出：`video_url`（必要时由上层决定是否下载）

2) `core/video_manager.py`
   - 负责视频下载（stream 写入，避免内存爆）、落盘、缓存清理
   - 输出：本地 `Path`

主入口 `main.py`：

- 初始化 `VideoManager` + `GrokVideoService`
- 新增命令 `/视频`
- 新增 LLM 工具（与 `gitee_edit_image` 一致走「读取消息图片 + 调用服务 + 发送结果」路径）
- 状态 emoji：
  - 接收请求后 `mark_processing(event)`
  - 成功发送视频后 `mark_success(event)`
  - 任意失败 `mark_failed(event)`

## 稳定性与边界情况

- 参考图转换失败：遍历多张图，取第一张可用的；全部失败则报错
- API 返回内容非结构化：做多策略 URL 抽取（直链/HTML/Markdown）
- 任务长耗时：命令入口快速返回，实际生成在后台任务中执行，避免阻塞/超时
- 发送视频失败：按 `send_mode` 做降级；最终至少发出可点击的 URL
- 超时上限：`timeout_seconds` 统一 clamp 到 `<=3600`

