# 配置与命令参考

[返回插件首页](../README.md) · [服务商接入教程](providers.md) · [映像工作台](studio.md) · [视频教程](video.md)

## 阅读顺序

先在 [服务商教程](providers.md) 中选择并配置 provider, 再配置对应功能链路. 只需要图形界面操作时阅读 [工作台指南](studio.md). 本页用于查询命令、预设、输出参数与高级设置.

## 配置概念：providers 与 chain

新用户最常见的卡点：只填了 `providers`，忘了填 `features.*.chain`，导致什么都没有生效。

```
providers（定义后端）          features.*.chain（选用哪个后端）
┌──────────────────────┐       ┌──────────────────────────────────┐
│ id: "my_gpt"         │◄──────│ provider_id: "my_gpt"            │
│ model: "gpt-image-2" │       │ output: "4K"                     │
│ api_keys: [...]      │       ├──────────────────────────────────┤
│ base_url: "..."      │       │ provider_id: "my_gemini"  (兜底) │
└──────────────────────┘       └──────────────────────────────────┘
```

**三步完成配置：**

1. 在映像工作台的模型连接, 或 AstrBot 插件配置的 `providers` 中新增服务商, 设置唯一 `id`, 填写对应的 Key、模型与 API 地址.
2. 在 `features.draw.chain`（文生图）、`features.edit.chain`（改图）等对应功能里，填入刚才的 `provider_id`
3. 在工作台保存并生效; 使用 AstrBot 配置页时按其提示保存或重载, 再发 `/aiimg 测试` 验证.

chain 里可以填多个 provider, 第一个是主用, 后面是备用. 对允许回退的失败按顺序切换; 创建结果不明的付费视频任务不会自动重建.

### 额外参数与生成质量

所有服务商模板均提供 **额外请求体(高级)** 的键值编辑入口, 支持字符串, 数字, 布尔值和嵌套 JSON.
OpenAI Images 模型支持自定义质量时, 添加 `quality` 并填写 `high`, `medium`, `low` 或 `auto` 等上游支持的值; 留空字典则沿用服务商默认行为.
Chat 出图使用网关定义的字段和嵌套位置, 不应假定支持 OpenAI Images 的所有参数.
Gemini 原生可填写 `{"generationConfig":{"temperature":0.6}}`, 嵌套合并会保留默认的比例, 分辨率和响应类型; Gemini 原生没有 OpenAI 的通用 `quality` 字段.
额外参数覆盖同名默认项. Gitee 异步改图转换为表单字段 (图片与 task_types 由插件管理), 即梦 GET 接口转换为查询参数, Vertex 写入生成请求的 variables.

## 接口模板速查

每个 provider 都需要一个唯一的 `id`。根据你的服务商接口类型选对模板：

| 接口类型                                       | 选用模板                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 标准 `POST /v1/images/generations` 或 `/edits` | `openai_images`                                                                                   |
| Chat 回复里包含图片 URL / base64               | `openai_chat` 或 `flow2api`                                                                       |
| 自定义完整路径（非标准 `/v1/...`）             | `openai_full_url_images`                                                                          |
| 直连 Gemini 官方 / Meinianda 香蕉系列          | `gemini_native`                                                                                   |
| 即梦 / 豆包                                    | `jimeng`                                                                                          |
| Gitee AI 文生图                                | `gitee_images`                                                                                    |
| Gitee AI 异步改图                              | `gitee_async`                                                                                     |
| Grok / xAI                                     | `grok_images` 或 `grok_chat`                                                                      |
| 视频生成                                       | `agnes_video` / `grok_video`（xAI 官方） / `flow2api_video` / `sora2_video`（通用 OpenAI Videos） |

provider 模板中的通用 `timeout` 默认均为 `600` 秒。升级时会保留现有 provider 的 URL、Key、模型、超时和其它自定义值；旧配置缺少新字段时才使用新版运行时默认值。`gemini_native` 额外支持 `max_retries`，默认重试 `2` 次，设为 `0` 可关闭重试。

## 命令速查

| 功能           | 命令                                                            |
| -------------- | --------------------------------------------------------------- |
| 普通文生图     | `/aiimg [@provider_id] <提示词> [输出]`                         |
| 文生图预设     | `/文生图 [@provider_id] <预设名> [补充提示词] [输出]`           |
| 改图           | `发送或引用图片 + /aiedit [@provider_id] <提示词> [输出]`       |
| 改图预设       | `发送或引用图片 + /预设名 [@provider_id] [额外提示词] [输出]`   |
| 自拍           | `/自拍 [@provider_id] <提示词> [输出]`                          |
| 自拍参考图管理 | `发送图片 + /自拍参考 设置`、`/自拍参考 查看`、`/自拍参考 删除` |
| 批量文生图     | `/批量n aiimg [@provider_id] <提示词> [输出]`                   |
| 批量改图       | `发送或引用图片 + /批量n aiedit [@provider_id] <提示词> [输出]` |
| 批量自拍       | `/批量n 自拍 [@provider_id] <提示词> [输出]`                    |
| 批量文生图预设 | `/批量n 文生图 [@provider_id] <预设名> [补充提示词] [输出]`     |
| 批量改图预设   | `发送或引用图片 + /批量n <改图预设名> [额外提示词] [输出]`      |
| 视频           | `发送或引用图片 + /视频 [@provider_id] <提示词或预设名>`        |
| 文生图预设列表 | `/文生图预设列表`                                               |
| 改图预设列表   | `/预设列表`                                                     |
| 视频预设列表   | `/视频预设列表`                                                 |
| 重发最近结果   | `/重发图片`                                                     |
| 查看改图帮助   | `/改图帮助`                                                     |

群聊中的图像命令必须带 AstrBot 当前配置的 `wake_prefix`. 即使其他插件提前把消息标记为已唤醒, 裸 `绘图` / `改图` / `自拍` 等普通聊天文本也不会触发本插件; 私聊仍遵循 AstrBot 原有的免前缀配置.

## 输出尺寸与比例

命令和 LLM 工具的 `output` 支持以下形式：

- 精确尺寸：`2048x1152`
- 自适应比例：`16:9`
- 分辨率：`4K`
- 自适应比例与分辨率：`16:9 4K`

命令会识别提示词中的明确比例、分辨率和精确尺寸; 末尾控制 token 仍会从提示词中移除, LLM tool 直接传入的自然提示词也会自动提取参数。示例：

```text
/aiimg 电影感海边日落 16:9 4K
/aiedit 保持人物不变，替换为夜景街道 4K
/自拍 黑色外套，楼梯间，低头看镜头 9:16 2K
/批量4 aiimg 同一主题的不同镜头 16:9 4K
/aiimg 电影感海边日落, 画面比例 16:9, 输出 4K
```

输出优先级为：prompt 中明确写出的参数 > LLM tool 的 `aspect_ratio` / `resolution` > 兼容 `output` > 当前 provider 的 `chain.output` > 功能的 `default_output` > 普通单图改图的输入图比例 > provider 默认值。

- Gemini Native 会传递自适应比例与分辨率; Vertex AI Anonymous 会传递自适应比例, 并在模型支持时传递分辨率。
- `gpt-image-2` 在 OpenAI Images 和 OpenAI Chat 模板中会映射为精确像素尺寸, 并保留 LLM 指定的常规比例与分辨率.
- 声明 `allowed_sizes` 的 OpenAI Images backend 会映射到最接近的合法像素尺寸。
- 普通单图改图只有在更高优先级没有指定比例时才继承输入图比例。
- 自拍和多图改图不会从参考图推断输出比例；自拍缺省使用 `features.selfie.default_aspect_ratio`，默认 `3:4`。
- 批量 LLM 任务会为每个规划项保存独立比例；用户未固定整组比例时，planner 会按构图选取至少两种比例。
- backend 不支持的输出维度会被忽略, 最终能力以对应服务商为准。

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
/批量4 aiimg 一个粉发少女，4 个不同镜头角度 16:9 4K
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

### 同步批量命令行为

- 单次数量上限由 `features.batch.max_count` 控制，默认 `8`，可设置 `1-32`
- 文生图批量并发由 `features.draw.batch_concurrency` 控制，默认 `2`，最高 `30`
- 改图 / 自拍批量并发由 `features.edit.batch_concurrency` 控制，默认 `2`，最高 `30`
- 改图批量和自拍批量都要求当前消息里能读到输入图片；文生图批量不需要图片
- 批量结果会按顺序直接发送单张图片
- 除原插件自带表情反馈外，不额外发送标题、提示词、状态、失败摘要这类通知文本

以上只描述 `/批量n ...` 直接命令。LLM 调用 `aiimg_batch_generate` 且后台模式生效时，Tool 会立即返回接单状态，图片在后台逐张发送，整组完成、部分成功、失败或取消后，Bot 还会按当前人格主动回应。

### 同步批量命令结果展示

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

2. 直接在 WebUI 的 `features.selfie.reference_images` 上传

### 查看和删除

```text
/自拍参考 查看
/自拍参考 删除
```

### 生成自拍

```text
/自拍 自然人像摄影，微笑，室内中性日光
/自拍 @provider_xxx 黑色外套，楼梯间，低头看镜头
```

说明：

- 如果 WebUI 里已经上传了参考照，优先使用 WebUI 配置
- 如果同时没有 WebUI 参考照，也没有通过命令保存参考照，`/自拍` 会直接报错
- 自拍链路为空时，可通过 `features.selfie.use_edit_chain_when_empty=true` 复用改图链路

### 自拍提示词前缀

`features.selfie.prompt_prefix` 可以设置一段固定的提示词前缀，在每次自拍时自动拼接到用户输入之前。适合把 Bot 的外貌描述、固定风格要求写死，不必每次都重复输入。

示例值：`A young woman with long black hair, realistic style, high quality, `

留空则使用插件内置的默认前缀。

内置默认前缀只负责参考图身份保持、自然质感和基础画面质量，不会替用户指定摄影风格、白平衡、视角、动作或构图。无论使用内置默认还是自定义前缀，插件都会补充拍摄设备的逻辑一致性要求：普通手持自拍时拍摄设备保持在画面外；明确要求对镜自拍、手机入镜或展示设备时允许相应设备自然出现，但不会无故复制其它设备；他拍、定时拍摄、人物手势和手持物品均遵循用户要求。

## LLM 工具

单图和批量图片使用以下两个核心工具; 视频使用 `grok_generate_video`, 其 `auto` / `selfie` 模式及参数见 [视频生成](video.md). 这些工具需要在 AstrBot 工具管理中保持启用, 并允许当前人格使用.

### `aiimg_generate`

适合单张图调用。

主要参数：

- `prompt`
- `mode`: `auto` / `text` / `edit` / `selfie_ref`
- `backend`: `auto` 或具体 `provider_id`
- `aspect_ratio`: `auto` 或 `16:9`、`9:16`、`4:3` 等
- `resolution`: `auto` 或 `1K`、`2K`、`4K`
- `output`: 兼容旧调用；没有用户明确要求时应留空

自动模式行为：

- 如果语义明显是在要求 “Bot 自拍”，并且已经配置了自拍参考照，会优先走 `selfie_ref`
- 如果当前消息里带图，会优先走改图
- 否则走文生图

### `aiimg_batch_generate`

适合“同主题、多变化”的一组图，一次调用完成“规划提示词 + 批量执行”。

主要参数：

- `prompt`
- `count`：默认 `4`，允许 `1-32`；用户明确指定数量时会以用户原话为准，最终不会超过 `features.batch.max_count`
- `mode`: `auto` / `text` / `edit` / `selfie_ref`
- `backend`
- `output`: 与单图工具相同, 支持精确尺寸、比例、分辨率和组合形式

工具行为：

- 会先让 `LLM` 规划多条彼此不重复、但整体都符合要求的提示词
- 每条规划项都必须包含 `title`、`prompt`、`variation_focus`
- 规划结果会做去重和数量校验，不合格会重试规划
- 图片生成完成后，插件会直接把结果发给用户；工具返回文本只做状态摘要，不需要二次帮用户“转述”
- 后台模式生效时，Tool 在容量预留和输入固化后立即返回；图片完成、部分成功、失败或取消后，Bot 会主动发送人格化终态回应
- 后台模式关闭、平台不受支持或 AstrBot 开启 streaming response 时，工具回退到原同步路径

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

- `features.batch.max_count`：单次批量最大张数，可设置 `1-32`
- `features.draw.batch_concurrency`：文生图批量并发，可设置 `1-30`
- `features.edit.batch_concurrency`：改图 / 自拍批量并发，可设置 `1-30`

### 图片输出编码

每个图片 provider 都可以通过 `output_format` 选择保存格式：

- `webp_lossless`：逐像素无损 WebP。推荐用于 Meinianda Gemini 4K，通常比原始 PNG 小很多，同时不改变任何像素。
- `webp`：高质量有损 WebP。默认质量 `97`，适合 Lossless WebP 仍超过平台限制时使用。
- `jpeg`：高质量 JPEG。默认质量 `95`、`4:4:4` 色度采样，兼容性最好，彩色文字和细线质量优于旧版默认编码。
- `png`：无损优化 PNG。不会改变像素，但 AI 生成的复杂 4K 图片通常只能减少少量体积。
- `auto`：不转换，完整保留上游返回的原始字节和格式。

`image_encoding` 可以统一调整编码参数：

- `image_encoding.jpeg_quality`：JPEG 质量，默认 `95`。
- `image_encoding.jpeg_subsampling`：JPEG 色度采样，默认 `4:4:4`。
- `image_encoding.webp_quality`：有损 WebP 质量，默认 `97`。
- `image_encoding.webp_lossless_effort`：无损 WebP 压缩强度，默认 `80`，只影响编码时间和体积，不影响像素。
- `image_encoding.webp_method`：WebP 编码方法，默认 `4`，范围 `0-6`。
- `image_encoding.png_compress_level`：PNG 无损压缩等级，默认 `9`。

实测 Meinianda `gemini-3.1-flash-image` 的 `3584x4800` 4K PNG 为 `21.362MiB`。插件实际编码后，PNG 无损优化仍为 `21.091MiB`，Lossless WebP 为 `16.856MiB` 且逐像素一致；有损 WebP `quality=97` 为 `3.360MiB`。因此推荐优先使用 `webp_lossless`，少数仍超过 QQ 限制的图片再改用 `webp`。

### 视频发送

- `features.video.send_mode`：视频发送方式。`auto`=优先通过 URL 发送，URL 失败再下载本地；`url`=仅通过 URL 发送；`file`=下载后以本地文件发送
- `features.video.send_timeout_seconds`：发送 Video 组件等待超时，默认 `90` 秒
- `features.video.download_timeout_seconds`：`send_mode=file/auto` 触发下载时的超时，默认 `300` 秒

### 并发与防抖

- `debounce_interval`：防抖时间，防止同一用户短时间重复提交同类任务
- `max_user_concurrency`：同一用户同时执行的图像任务上限
- `max_user_video_concurrency`：同一用户同时执行的视频任务上限

### 发送前处理

- `send.weixin_compress_images`：个人微信发送前压缩图片，默认开启，仅对 `weixin_oc` 生效。
- `send.weixin_image_max_side`：个人微信图片最长边，默认 `4096`。
- `send.weixin_image_max_size_kb`：个人微信图片目标大小，默认 `10240KB`。
- `send.weixin_api_timeout_seconds`：个人微信发送超时，默认 `60` 秒。

这组配置只影响 `weixin_oc`。QQ / OneBot 会直接发送 provider `output_format` 生成的图片；文件超过 `20MiB` 时仍会回退为文件发送。若个人微信发送 4K 图片时出现 `upload_to_cdn TimeoutError`，优先调高 `send.weixin_api_timeout_seconds`，或降低 `send.weixin_image_max_size_kb`。

### 存储与缓存

- `storage.max_cached_images`：本地图片最大缓存数，默认 `50`；超出时自动清理一半旧缓存
- `storage.max_cached_videos`：本地视频最大缓存数，默认 `20`；仅在 `send_mode=file/auto` 触发下载时生效（`0`=不清理）

### 网络安全

- `network.media_allow_private`：是否允许从私有/内网地址下载图片或视频，默认 **关闭**（防止 SSRF 攻击）。自建服务且服务端在内网时可开启
- `network.max_image_bytes`：图片下载大小上限，默认 `50MB`（52428800 字节）
- `network.max_video_bytes`：视频下载大小上限，默认 `50MB`
- `network.max_redirects`：最大 HTTP 重定向次数，默认 `5`
- `network.dns_resolve_timeout_seconds`：DNS 解析超时，默认 `2` 秒

### 功能开关

- `features.draw.enabled`
- `features.edit.enabled`
- `features.selfie.enabled`
- `features.video.enabled`
- `features.<mode>.llm_tool_enabled`

## 平台与限制

### 官方维护 / 推荐环境

- 实际 AstrBot 运行环境：`Python >= 3.12`
- `AstrBot >= 4.16.0, < 5`
- 主要维护平台：`QQ / aiocqhttp`
- 兼容平台：个人微信私聊 `weixin_oc`
- GitHub CI：Ubuntu 上额外检查插件源码的 Python `3.10 / 3.11 / 3.12 / 3.13` 兼容性，并在 Windows、macOS 的 Python `3.12` 上回归；Python `3.10 / 3.11` 结果不代表对应 AstrBot 版本可在该解释器上部署

### 已知平台限制

- 批量结果默认就是普通消息逐张发送
- 视频发送依赖适配器是否支持 `Video.fromFileSystem` 或 `Video.fromURL`
- 某些需要 URL 回退输入的后端，依赖当前 AstrBot 环境具备文件服务能力
- `weixin_oc` 官方适配器仅支持个人私聊，不支持微信群聊；大图发送受微信 CDN 上传耗时影响。
- LLM 后台任务只正式支持 `aiocqhttp` 和 `weixin_oc`，并且只支持单 AstrBot 进程；多进程共享同一任务账本会 fail-closed，不会抢跑任务。
- 后台任务数据库必须位于本机可写磁盘。只读目录或 NFS / SMB 共享目录不属于支持范围，初始化失败时会关闭后台模式，不影响原同步路径。
- 不保证把正在运行的 spool 目录跨 Windows / Linux 搬迁后继续恢复；请在同一主机和数据目录内完成重启恢复。
- AstrBot 开启 streaming response 时，LLM Tool 自动回退同步执行，不会假装后台接单。

### 关于“其他平台能不能跑”

单图文生图 / 改图这类能力，只要当前适配器支持普通文本和图片消息，很多时候都能工作；但本插件的主要测试和维护场景仍是 `QQ / aiocqhttp`。如果你跑在其他平台，建议先用小流量自测，尤其是：

- 批量结果展示
- 视频发送
- 合并转发回退行为
- 大图发送失败后的兜底链路

## Gitee 支持的图像尺寸

> 仅对 Gitee 文生图能力生效。其他后端是否支持，取决于服务商本身。

| 比例   | 可用尺寸                                       |
| ------ | ---------------------------------------------- |
| `1:1`  | `256x256`、`512x512`、`1024x1024`、`2048x2048` |
| `4:3`  | `1152x896`、`2048x1536`                        |
| `3:4`  | `768x1024`、`1536x2048`                        |
| `3:2`  | `2048x1360`                                    |
| `2:3`  | `1360x2048`                                    |
| `16:9` | `1024x576`、`2048x1152`                        |
| `9:16` | `576x1024`、`1152x2048`                        |

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

这只适用于 `/批量n ...` 同步命令：结果默认只发图片本体，避免刷出机械通知。LLM 后台批量任务会在整组完成、部分成功、失败或取消后，由 Bot 按当前人格主动回应。

### 为什么 `request_mode=stream` 没起作用

并不是所有 provider 模板都支持“双路径请求模式”。单路径后端会忽略这个设置，插件会在校验时提示你。

### 个人微信生成图为什么发送失败或超时

个人微信 `weixin_oc` 发送图片会先上传到微信 CDN。即使图片低于 10MB，也可能因为网络或默认超时过短导致 `upload_to_cdn TimeoutError`。

建议：

- 保持 `send.weixin_compress_images=true`
- 将 `send.weixin_api_timeout_seconds` 设置为 `60-120`
- 如果仍超时，降低 `send.weixin_image_max_size_kb`
- 视频发送能力取决于当前 `weixin_oc` 适配器是否支持对应 `Video` 组件
