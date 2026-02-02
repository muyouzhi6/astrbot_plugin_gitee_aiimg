# astrbot_plugin_gitee_aiimg

> 当前版本：v3.0.0（配置界面已重构，不再兼容旧版 providers/旧字段）

多服务商图像插件：统一支持文生图（Text-to-Image）、改图/图生图（Image-to-Image/Edit）、Bot 自拍参考照（上传参考人像后再“自拍”）、以及 Grok imagine 视频生成。

## 你只需要记住两件事

1) 生图看 `draw`：先选 `draw.provider`，再把对应的 `draw.<provider>` 配完整。
2) 改图看 `edit`：先选 `edit.provider`，再把对应的 `edit.<provider>` 配完整。

配置面板会只显示你选中的那一套服务商参数，避免“看一堆字段不知道填哪个”。

## 降级服务商怎么用（fallback）

用途：当“主服务商”请求失败时，按顺序自动切换到备用服务商继续尝试。

配置位置：

- 生图：`draw.fallback_1 / draw.fallback_2 / draw.fallback_3`
- 改图：`edit.fallback_1 / edit.fallback_2 / edit.fallback_3`

填写规则：

- 直接从下拉框选择服务商别名（例如 `gemini_native`、`grok`、`gitee_async`）
- 留空表示不启用该级降级
- 会按 `1 → 2 → 3` 的顺序尝试

例子 1（生图）：主用 Grok，失败就降级到 Gemini 原生，再不行降级到 Gitee

- `draw.provider = grok`
- `draw.fallback_1 = gemini_native`
- `draw.fallback_2 = gitee`

例子 2（改图）：主用 Gemini OpenAI 网关（有些网关不支持 images.edit），失败就降级到 Gemini 原生，再不行到千问异步

- `edit.provider = gemini_openai`
- `edit.fallback_1 = gemini_native`
- `edit.fallback_2 = gitee_async`

## 你的日志里“聊天可用但改图 404”的原因

你贴的这段 “Provider is available” 检测的是 `chat_completion`（也就是 `/v1/chat/completions` 能不能通），
但我们改图用的是 `images.edit`（也就是 `/v1/images/edits` 这一类接口）。

很多第三方网关（例如只转发聊天的网关）会出现：

- 聊天能用（PONG 成功）
- 图片接口 404（因为根本没实现 `/v1/images/*`）

解决方式：

- 自拍/改图：优先用 `edit.provider = gemini_native`（Gemini 原生改图稳定）
- 如果你用的是“聊天出图”的网关：把对应服务商的 `api_mode` 改成 `chat`（走 `/v1/chat/completions` 解析图片），不要用 `images`
- 如果你的网关/模型不支持“图文输入改图”：把 `supports_edit` 关掉，并设置 `edit.fallback_1 = gemini_native`

## Base URL 怎么填（最重要）

插件会自动把 OpenAI 兼容的 `base_url` 规范化成“包含 `/v1`”的形式，所以你可以：

- 填基础域名：`https://api.x.ai`、`https://ai.gitee.com`
- 或者填带版本：`https://api.x.ai/v1`、`https://ai.gitee.com/v1`

不要填这种“具体接口地址”：

- `.../chat/completions`（这是聊天接口，不是图片）
- `.../v1/chat/completions`

如果你填错了，通常会看到 `404`（base_url/路径不对）或 `不支持 images.edit`（服务商本身不支持改图）。

## “我网关只有 /v1/chat/completions，但我确实能出图”怎么配？

这类网关经常是“聊天出图”（模型把图片塞在 chat content 里），但没有 `/v1/images/*`。

现在插件支持两种模式（在 `grok/gemini_openai/openai_compat` 里都有 `api_mode`）：

- `api_mode=chat`：走 `/v1/chat/completions`，从回复里解析 `![](data:image/...base64,...)` 或图片 URL（适配你这种网关）
- `api_mode=images`：走 `/v1/images/*`（只有当你的服务商真的实现了 Images API 才能用）

如果你“第一次能出图、第二次 404”，基本是网关后端节点不一致导致：有的节点实现了图片能力，有的没有。建议用 `api_mode=chat` 更稳。

## 支持的服务商（生图 / 改图）

- `grok`：Grok 图片（OpenAI 兼容 Images API）
- `gemini_native`：Gemini 原生 generateContent（支持 1K/2K/4K）
- `gemini_openai`：Gemini 的 OpenAI 兼容图片接口/网关（按你的网关文档填写 base_url/model）
- `openai_compat`：通用 OpenAI 兼容 Images API（任意兼容服务商）
- `jimeng`：即梦/豆包聚合接口（接口形态参考 `data/plugins/doubao`）
- `gitee`：Gitee 生图（仅生图）
- `gitee_async`：Gitee 千问异步改图（仅改图）

## 分辨率/尺寸（默认 4K）

- `gemini_native`：用 `resolution`（默认 `4K`）
- `grok/gemini_openai/openai_compat`：用 `size`（默认 `4096x4096`）；若服务商不支持会自动降级到 `2048x2048`
- LLM 工具也支持 `output` 参数传 `4K` / `2K` / `1K` 或 `4096x4096` 这种尺寸
- 注意：有些服务商/网关会无视你请求的 4K，实际只返回 2K（或更低）。这通常是服务商侧限制，不是插件能强制突破的。

## 使用

### 文生图

```
/生图 一张电影感的街拍写真
```

（也兼容 `/aiimg`）

### 改图/图生图

发送（或引用）图片，然后：

```
/改图 把衣服换成黑色风衣，背景换成雨夜街头
```

（也兼容 `/aiedit`）

### 自拍参考照（上传人像后再自拍）

1) 设置参考照（二选一）：

- WebUI 上传：`selfie.reference_images`
- 聊天设置：发送图片 + `/自拍参考 设置`

2) 生成自拍：

```
/自拍 跟图1同款写真，穿上同款衣服，姿势也模仿
```

提示：支持“图片 + 文本同一条消息”一起发送（图片在前或在后都可以）。

### 视频生成

发送（或引用）图片，然后：

```
/视频 让人物微笑并轻微点头，镜头缓慢推近
```

## LLM 工具：aiimg_generate

- `mode`: `auto` / `text` / `edit` / `selfie_ref`
- `backend`: `auto` 或指定服务商别名（`grok`/`gemini`/`gitee_async`/`jimeng`/`openai_compat` 等）
- `output`: `4K` / `2K` / `1K` / `1024x1024` / `4096x4096` 等
