# AstrBot Gitee AI 图像生成插件

[![Plugin Version](https://img.shields.io/badge/Version-v4.3.0-4f8cc9?style=for-the-badge)](./CHANGELOG.md)
[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.16.0%2C%20%3C5-ff69b4?style=for-the-badge)](https://github.com/AstrBotDevs/AstrBot)
[![Platform](https://img.shields.io/badge/Primary-aiocqhttp-4caf50?style=for-the-badge)](#平台与限制)

多服务商文生图 / 改图 / 自拍参考照 / 视频生成插件，支持命令调用、`LLM tool` 调用、预设提示词、批量出图、请求模式控制、多 `API Key`轮询、失败兜底与超时配置。

> [!IMPORTANT]
> 这份文档对应 `v4.3.0` 配置结构。
> 
> - `v4` 与旧版 `v3 / v2` 配置不兼容，升级后请重新检查 **AstrBot 管理面板中的插件配置**。
> - 插件主维护场景是 `QQ / aiocqhttp`。
> - 批量结果的“合并转发”当前只有 `aiocqhttp` 原生支持；其他平台会在开启回退时自动改为普通消息逐条发送。
> - 历史更新内容见 [CHANGELOG.md](./CHANGELOG.md)。

## 新版本重点

- 新增文生图预设：`/文生图 预设名 补充提示词`
- 新增统一批量命令：`/批量n aiimg ...`、`/批量n aiedit ...`、`/批量n 自拍 ...`
- 支持批量配合预设：`/批量n 文生图 预设名 补充提示词`、`/批量n 改图预设名 补充提示词`
- 新增 `LLM` 批量工具 `aiimg_batch_generate`：先规划多条不重复提示词，再一次性批量执行
- 文生图批量并发和改图 / 自拍批量并发拆开配置
- Provider 级新增 `generate_request_mode` / `edit_request_mode`
- 请求模式兼容旧配置：`auto` 不会再覆盖旧的 `enable_stream_*` 布尔配置

## 功能概览

本插件支持：

- 文生图 `Text-to-Image`
- 图生图 / 改图 `Image-to-Image/Edit`
- 自拍参考照模式
- 图生视频 / 文生视频（取决于所选后端能力）
- 文生图预设、改图预设、视频预设
- 指令批量出图
- `LLM tool` 单图调用与批量调用

核心设计是把 **服务商实例 `providers`** 和 **功能链路 `features.*.chain`** 分开。你可以给同一类能力挂多个 provider，插件会按链路顺序兜底切换。

## 快速上手

### 1. 先配 `providers`

每个 provider 都需要一个唯一的 `id`。目前内置了多类模板，例如：

- Gemini 原生 `generateContent`
- Vertex AI Anonymous
- OpenAI 兼容 `Images API`
- OpenAI 兼容 `Chat` 出图解析
- OpenAI 兼容完整路径
- Flow2API
- Grok / Grok2API
- Gitee / Gitee 异步改图
- 即梦 / 豆包聚合
- 视频后端

一般选择建议：

- 标准 `POST /v1/images/generations` / `POST /v1/images/edits`：用 `OpenAI 兼容通用（Images）`
- 只在 `chat.completions` 回复里回图片：用 `OpenAI 兼容（Chat 出图解析）`
- 你的网关路径不是标准 `/v1/...`：用 `OpenAI 兼容-完整路径`
- 直连 Gemini 官方：用 `Gemini 原生`

### 2. 再配 `features.*.chain`

- `features.draw.chain`：文生图链路
- `features.edit.chain`：改图链路
- `features.selfie.chain`：自拍链路
- `features.video.chain`：视频链路

链路按顺序兜底。前面的 provider 失败后，插件会自动切到后面的 provider。

### 3. 按需开关功能

- `features.<mode>.enabled`：是否启用该能力
- `features.<mode>.llm_tool_enabled`：是否允许 `LLM` 通过工具调用该能力

## 配置说明

> [!TIP]
> **本插件没有独立 WebUI 界面。** 所有的配置都在 **AstrBot 的管理面板**（默认 `http://localhost:6185`）中进行。
> 登录面板后，在左侧菜单点击“插件”，找到本插件并点击进入配置表单即可。

## 命令速查

| 功能 | 命令 |
| --- | --- |
| 普通文生图 | `/aiimg [@provider_id] <提示词> [比例]` |
| 文生图预设 | `/文生图 [@provider_id] <预设名> [补充提示词]` |
| 改图 | `发送或引用图片 + /aiedit [@provider_id] <提示词>` |
| 改图预设 | `发送或引用图片 + /预设名 [@provider_id] [额外提示词]` |
| 自拍 | `/自拍 [@provider_id] <提示词>` |
| 自拍参考图管理 | `发送图片 + /自拍参考 设置`、`/自拍参考 查看`、`/自拍参考 删除` |
| 批量文生图 | `/批量n aiimg [@provider_id] <提示词>` |
| 批量改图 | `发送或引用图片 + /批量n aiedit [@provider_id] <提示词>` |
| 批量自拍 | `/批量n 自拍 [@provider_id] <提示词>` |
| 批量文生图预设 | `/批量n 文生图 [@provider_id] <预设名> [补充提示词]` |
| 批量改图预设 | `发送或引用图片 + /批量n <改图预设名> [额外提示词]` |
| 视频 | `发送或引用图片 + /视频 [@provider_id] <提示词或预设名>` |
| 文生图预设列表 | `/文生图预设列表` |
| 改图预设列表 | `/预设列表` |
| 视频预设列表 | `/视频预设列表` |
| 重发最近结果 | `/重发图片` |
| 查看改图帮助 | `/改图帮助` |

## 文生图预设

### 配置位置

在 `features.draw.presets` 里配置，格式是：

```text
预设名:英文提示词
```

示例：

```text
手办:Transform into collectible figurine style
胶片人像:Cinematic portrait, soft rim light, film grain, realistic skin texture
```

### 调用方式

```text
/文生图 手办 将这只猫做成高细节手办
/文生图 @gemini_chat 胶片人像 黑色高领毛衣，窗边逆光，半身构图
```

规则说明：

- `/文生图` 后面第一个 token 如果命中文生图预设名，就按“预设 + 补充提示词”处理
- 预设名后面的**全部文本**都会作为补充提示词，可包含空格，也可写成多行消息
- 如果第一个 token 没命中预设，就按普通 `/文生图` 文生图处理
- 指定 provider 时，`@provider_id` 要放在预设名前面

实际拼接方式是：

```text
预设提示词

补充要求：
你的补充提示词
```

## 改图预设

### 配置位置

在 `features.edit.presets` 里配置，格式同样是：

```text
预设名:英文提示词
```

示例：

```text
手办:Transform into figurine style
Q版化:Convert to chibi illustration style
```

### 调用方式

发送或引用图片后：

```text
/手办 加个透明亚克力底座
/手办 @grok2api 换成偏暖色棚拍
/Q版化
```

规则说明：

- 每个改图预设都会动态注册成一个独立命令，例如 `/手办`
- 预设命令后面的文本会作为额外提示词附加到预设后面
- 预设命令也支持 `@provider_id` 覆盖，例如 `/手办 @provider_xxx 补充词`
- `/预设列表` 可以查看当前所有改图预设

## 批量出图

### 基本语法

批量命令统一使用：

```text
/批量n ...
```

其中 `n` 是数量，例如：

```text
/批量4 aiimg 一个粉发少女，4 个不同镜头角度
/批量6 aiedit 把这张照片分别改成不同灯光和情绪
/批量8 自拍 同一套穿搭，不同姿势、表情和俯仰角
/批量5 文生图 手办 将这辆车做成桌面手办
/批量4 文生图 @gemini_chat 胶片人像 夜景路灯，表情和构图都不重复
/批量3 手办 加不同底座和背景陈列
```

### 支持的批量入口

- `/批量n aiimg ...`
- `/批量n 文生图 ...`
- `/批量n aiedit ...`
- `/批量n 自拍 ...`
- `/批量n 改图预设名 ...`

### 行为说明

- 单次数量上限由 `features.batch.max_count` 控制，默认 `8`
- 文生图批量并发由 `features.draw.batch_concurrency` 控制，默认 `2`
- 改图 / 自拍批量并发由 `features.edit.batch_concurrency` 控制，默认 `2`
- 改图批量和自拍批量都要求当前消息里能读到输入图片；文生图批量不需要图片
- 批量结果会按顺序直接发送单张图片
- 除原插件自带表情反馈外，不额外发送标题、提示词、状态、失败摘要这类通知文本

### 批量结果展示

- 批量任务成功的图片会一张一张直接发出
- 不额外插入摘要、说明、失败提示等文本消息
- 如果某几张失败，只保留原插件自己的表情反馈，不额外发通知

## 自拍参考照

### 设置参考照

二选一即可：

1. 发送图片后执行：

```text
/自拍参考 设置
```

2. 直接在 WebUI（AstrBot 管理面板）的 `features.selfie.reference_images` 上传

### 查看和删除

```text
/自拍参考 查看
/自拍参考 删除
```

### 生成自拍

```text
/自拍 日常自拍照，微笑，窗边自然光
/自拍 @provider_xxx 黑色外套，楼梯间，低头看镜头
```

说明：

- 如果面板里已经上传了参考照，优先使用面板配置
- 如果同时没有面板参考照，也没有通过命令保存参考照，`/自拍` 会直接报错
- 自拍链路为空时，可通过 `features.selfie.use_edit_chain_when_empty=true` 复用改图链路

## 视频生成

发送或引用图片后：

```text
/视频 镜头缓慢推进，人物轻微转头
/视频 @grok_video 黄昏街景，镜头跟拍
/视频 电影感 拉近镜头，轻微风吹头发
/视频预设列表
```

如果第一个 token 命中 `features.video.presets` 里的预设名，就会按“视频预设 + 额外提示词”处理。

## LLM 工具

插件提供两个核心 `LLM tool`：

### `aiimg_generate`

适合单张图调用。

主要参数：

- `prompt`
- `mode`: `auto` / `text` / `edit` / `selfie_ref`
- `backend`: `auto` 或具体 `provider_id`
- `output`: 例如 `2048x2048`、`4K`

自动模式行为：

- 如果语义明显是在要求 “Bot 自拍”，并且已经配置了自拍参考照，会优先走 `selfie_ref`
- 如果当前消息里带图，会优先走改图
- 否则走文生图

### `aiimg_batch_generate`

适合“同主题、多变化”的一组图，一次调用完成“规划提示词 + 批量执行”。

主要参数：

- `prompt`
- `count`：默认 `4`，建议 `2-8`，最终不会超过 `features.batch.max_count`
- `mode`: `auto` / `text` / `edit` / `selfie_ref`
- `backend`
- `output`

工具行为：

- 会先让 `LLM` 规划多条彼此不重复、但整体都符合要求的提示词
- 每条规划项都必须包含 `title`、`prompt`、`variation_focus`
- 规划结果会做去重和数量校验，不合格会重试规划
- 图片生成完成后，插件会直接把结果发给用户；工具返回文本只做状态摘要，不需要二次帮用户“转述”

这正适合下面这种需求：

- 同场景、同穿搭，不同姿势 / 角度 / 表情的写真集
- 参考同一张自拍，批量改出不同构图版本
- 同一文生图主题，快速出多个候选方案筛图

## Provider 请求模式

部分支持双路径的 provider 可分别配置：

- `generate_request_mode`
- `edit_request_mode`

可选值：

- `auto`：由后端自己决定
- `stream`：强制优先走流式
- `non_stream`：强制直接走非流式

补充说明：

- 显式设置 `stream` / `non_stream` 时，它们优先级最高
- 如果你是从旧配置升级，且旧配置里还保留 `enable_stream_generate` / `enable_stream_edit`，当新的 `*_request_mode=auto` 时，插件会继续沿用旧布尔值，不会被 `auto` 覆盖
- 单路径后端即使显示了这个配置项，也可能会忽略该设置；校验阶段会给出提示

## 关键配置项

### 批量相关

- `features.batch.max_count`：单次批量最大张数
- `features.draw.batch_concurrency`：文生图批量并发
- `features.edit.batch_concurrency`：改图 / 自拍批量并发

### 并发与防抖

- `debounce_interval`：防抖时间，防止同一用户短时间重复提交同类任务
- `max_user_concurrency`：同一用户同时执行的图像任务上限
- `max_user_video_concurrency`：同一用户同时执行的视频任务上限

### 功能开关

- `features.draw.enabled`
- `features.edit.enabled`
- `features.selfie.enabled`
- `features.video.enabled`
- `features.<mode>.llm_tool_enabled`

## 平台与限制

### 官方维护 / 推荐环境

- `AstrBot >= 4.16.0, < 5`
- 主要维护平台：`QQ / aiocqhttp`

### 已知平台限制

- 批量结果默认就是普通消息逐张发送
- 视频发送依赖适配器是否支持 `Video.fromFileSystem` 或 `Video.fromURL`
- 某些需要 URL 回退输入的后端，依赖当前 AstrBot 环境具备文件服务能力

### 关于“其他平台能不能跑”

单图文生图 / 改图这类能力，只要当前适配器支持普通文本和图片消息，很多时候都能工作；但本插件的主要测试和维护场景仍是 `QQ / aiocqhttp`。如果你跑在其他平台，建议先用小流量自测，尤其是：

- 批量结果展示
- 视频发送
- 合并转发回退行为
- 大图发送失败后的兜底链路

## Gitee AI API Key 获取

1. 访问 <https://ai.gitee.com/serverless-api?model=z-image-turbo>
2. 在对应模型页面开通服务并创建 `API Key`
3. 将 `API Key` 填到对应 provider 的 `api_keys`

## Gitee 支持的图像尺寸

> 仅对 Gitee 文生图能力生效。其他后端是否支持，取决于服务商本身。

| 比例 | 可用尺寸 |
| --- | --- |
| `1:1` | `256x256`、`512x512`、`1024x1024`、`2048x2048` |
| `4:3` | `1152x896`、`2048x1536` |
| `3:4` | `768x1024`、`1536x2048` |
| `3:2` | `2048x1360` |
| `2:3` | `1360x2048` |
| `16:9` | `1024x576`、`2048x1152` |
| `9:16` | `576x1024`、`1152x2048` |

## 常见问题

### `/文生图` 预设没有生效

先检查：

- `features.draw.presets` 里是否真的配置了该预设名
- 调用时预设名是否放在 `/文生图` 后面的第一个 token
- 如果用了 `@provider_id`，它必须写在预设名前面

### `/批量n aiedit ...` 没反应

改图批量要求当前消息里能读到输入图片。最稳妥的发法是：

- 先发图再跟命令
- 或直接回复图片消息执行命令

### 批量结果为什么没有额外说明文字

这是当前设计：批量结果默认只发图片本体，避免刷出不拟人的通知文本。

### 为什么 `request_mode=stream` 没起作用

并不是所有 provider 模板都支持“双路径请求模式”。单路径后端会忽略这个设置，插件会在校验时提示你。

## 原仓库展示内容（保留）

这一节保留原仓库 README 里的推广与展示内容，方便插件市场页和仓库首页继续正常展示。

### Gitee AI API Key 获取方法（原展示）

1. 访问 <https://ai.gitee.com/serverless-api?model=z-image-turbo>

2. ![PixPin_2025-12-05_16-56-27](https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841)

3. 免费额度

![PixPin_2025-12-05_16-56-49](https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb)

4. 可以涩涩，警惕违规被举报

5. 好用可以给个 💗

### 出图展示区（原展示）

![出图展示 1](https://github.com/user-attachments/assets/c2390320-6d55-4db4-b3ad-0dde7b447c87)

![出图展示 2](https://github.com/user-attachments/assets/3d8195e5-5d89-4a12-806e-8a81e348a96c)

![出图展示 3](https://github.com/user-attachments/assets/c270ae7f-25f6-4d96-bbed-0299c9e61877)

插件开发 QQ 群：`215532038`

![QQ群二维码](https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee)
