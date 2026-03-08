# 视频功能说明（v4）

本文件用于说明 `astrbot_plugin_gitee_aiimg` 的视频生成功能在 v4 配置结构下的使用与配置要点。

## 命令

发送/引用图片后：

- `/视频 [@provider_id] <提示词>`
- `/视频 [@provider_id] <预设名> [额外提示词]`
- `/视频预设列表`

说明：

- `@provider_id` 为可选的“临时指定一次使用哪个 provider”，不会改变你配置的默认链路顺序。
- `预设名` 来自配置 `features.video.presets`（格式：`预设名:英文提示词`）。

## 配置（v4）

### 1) 开关与链路

- `features.video.enabled`：是否启用视频功能（命令与 LLM 工具都会受影响）
- `features.video.llm_tool_enabled`：是否允许 LLM 调用视频工具（不影响命令）
- `features.video.chain`：按顺序填写 provider（第一个=主用，其余=兜底）

### 2) 预设

- `features.video.presets`：列表，元素格式 `预设名:英文提示词`
  - 仅预设：`prompt = preset_prompt`
  - 预设 + 额外：`prompt = preset_prompt + ", " + extra_prompt`

### 3) 发送方式与超时

- `features.video.send_mode`：`auto` / `url` / `file`
  - `url`：优先 `Video.fromURL(video_url)`
  - `file`：下载后 `Video.fromFileSystem(video_path)`
  - `auto`：先 URL，失败再下载；仍失败则回退发送纯链接
- `features.video.send_timeout_seconds`：发送 Video 组件等待时间
- `features.video.download_timeout_seconds`：下载视频超时（仅 `file/auto` 有意义）
- `storage.max_cached_videos`：下载缓存上限（仅对 `file/auto` 生效）

## Provider（视频）

视频 provider 需要在 `providers` 中添加 `grok_video` 类型的实例，然后把它的 `id` 填到 `features.video.chain` 里。

`grok_video` 实例的关键字段：

- `id`：服务商 ID（用户自定义字符串，必须唯一）
- `server_url`：默认 `https://api.x.ai`（插件内部会补全 `/v1/chat/completions`）
- `api_key`：Grok API Key
- `model`：默认 `grok-imagine-0.9`
- `timeout_seconds` / `max_retries` / `retry_delay` / `empty_response_retry`：稳定性参数

## 兜底逻辑（fallback）

视频生成会按 `features.video.chain` 的顺序依次尝试 provider：

- 当前 provider 失败（网络/超时/返回内容异常/无法解析 video URL） -> 自动切换下一个 provider
- 全部失败 -> 返回失败提示（并尽量给出可点击的 URL 或错误原因）
