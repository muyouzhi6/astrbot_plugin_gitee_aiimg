# AstrBot Gitee AI 图像生成插件 （有免费额度）

> **当前版本**: v2.0.0

本插件为 AstrBot 接入 Gitee AI 的图像生成能力，支持**文生图**和**图生图**两种模式，支持通过自然语言或指令调用，支持多 Key 轮询。

## 功能特性

### 文生图 (Text-to-Image)

- 支持通过 LLM 自然语言调用生成图片
- 支持通过指令 `/aiimg` 生成图片
- 支持多种图片比例和分辨率
- 支持自定义模型和负面提示词

### 图生图 (Image-to-Image) 🆕

- 支持通过 LLM 自然语言调用编辑图片
- 支持通过指令 `/aiedit` 编辑图片
- 支持多种编辑任务类型：身份保持、风格迁移、背景替换等
- 支持多图输入（如人脸+风格图）

### 通用特性

- 支持多 API Key 轮询调用
- 自动清理旧图片，节省存储空间
- 100% 异步 I/O，不阻塞事件循环

## 配置

在 AstrBot 的管理面板中配置以下参数：

### 文生图配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `base_url` | Gitee AI API 地址 | `https://ai.gitee.com/v1` |
| `api_key` | Gitee AI API Key（支持多 Key 轮询） | `[]` |
| `model` | 使用的模型名称 | `z-image-turbo` |
| `size` | 默认图片大小 | `1024x1024` |
| `num_inference_steps` | 推理步数 | `9` |
| `negative_prompt` | 负面提示词 | `""` |

### 图生图配置 🆕

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `base_url` | 图生图 API 地址（留空使用文生图地址） | `""` |
| `api_key` | 图生图 API Key（留空使用文生图 Key） | `[]` |
| `model` | 图生图模型名称 | `Qwen-Image-Edit-2511` |
| `num_inference_steps` | 图生图推理步数 | `4` |
| `guidance_scale` | 引导系数 | `1.0` |
| `poll_interval` | 轮询间隔（秒） | `5` |
| `poll_timeout` | 轮询超时（秒） | `300` |

### 配置说明

- **api_key**: 支持配置多个 Key 以实现轮询调用，可有效分摊 API 额度消耗
- **negative_prompt**: 可自定义负面提示词，留空则使用内置默认值
- **edit_api_key**: 图生图可单独配置 Key，也可留空复用文生图的 Key

## Gitee AI API Key获取方法

1.访问<https://ai.gitee.com/serverless-api?model=z-image-turbo>

2.<img width="2241" height="1280" alt="PixPin_2025-12-05_16-56-27" src="https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841" />

3.免费额度<img width="240" height="63" alt="PixPin_2025-12-05_16-56-49" src="https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb" />

4.可以涩涩，警惕违规被举报

5.好用可以给个🌟

## 支持的图像尺寸

> ⚠️ **注意**: 仅支持以下尺寸，使用其他尺寸会报错

| 比例 | 可用尺寸 |
|------|----------|
| 1:1 | 256×256, 512×512, 1024×1024, 2048×2048 |
| 4:3 | 1152×896, 2048×1536 |
| 3:4 | 768×1024, 1536×2048 |
| 3:2 | 2048×1360 |
| 2:3 | 1360×2048 |
| 16:9 | 1024×576, 2048×1152 |
| 9:16 | 576×1024, 1152×2048 |

## 使用方法

### 文生图

#### 指令调用

```
/aiimg <提示词> [比例]
```

示例：

- `/aiimg 一个可爱的女孩` (使用默认比例 1:1)
- `/aiimg 一个可爱的女孩 16:9`
- `/aiimg 赛博朋克风格的城市 9:16`

#### 自然语言调用

直接与 bot 对话，例如：

- "帮我画一张小猫的图片"
- "生成一个二次元风格的少女"

### 图生图 🆕

#### 指令调用

```
/aiedit <提示词> [任务类型]
```

**任务类型说明：**

| 类型 | 说明 |
|------|------|
| `id` | 保持身份特征（默认） |
| `style` | 风格迁移 |
| `subject` | 主体替换 |
| `background` | 背景替换 |
| `element` | 元素编辑 |

示例：

- 发送图片 + `/aiedit 把背景换成海边 background`
- 发送图片 + `/aiedit 转换成油画风格 style`
- 发送两张图片 + `/aiedit 根据图1的人物和图2的风格生成结婚照 id,style`

#### 自然语言调用

发送图片并对话，例如：

- 发送图片 + "帮我把这张图的背景换成星空"
- 发送图片 + "把这张照片转成动漫风格"

## 注意事项

- 请确保您的 Gitee AI 账号有足够的额度。

- 生成的图片会临时保存在 `data/plugin_data/astrbot_plugin_gitee_aiimg/images` 目录下
- 插件会自动清理旧图片，无需手动管理
- `/aiimg` 命令和 LLM 调用均有 10 秒防抖机制，避免重复请求

## 更新日志

### v2.0.0 (2025-01)

**🆕 新功能：图生图**

- 新增 `/aiedit` 指令支持图像编辑
- 新增 `edit_image` LLM 工具，支持自然语言调用图生图
- 支持多种任务类型：身份保持、风格迁移、背景替换等
- 支持多图输入（如人脸+风格参考图）
- 默认使用 `Qwen-Image-Edit-2511` 模型

**⚙️ 配置项扩展**

- 新增 `edit_base_url`、`edit_api_key`、`edit_model` 等图生图专用配置
- 图生图配置可独立配置，也可复用文生图配置

### v1.2 (2024-12)

**🚀 性能优化**

- 100% 异步 I/O，不再阻塞事件循环
- HTTP 客户端复用，减少连接开销
- 自动清理旧图片，保留最近 50 张

**🐛 Bug 修复**

- 修复内存泄漏问题
- 统一防抖机制，`/aiimg` 命令现也有 10 秒防抖

**✨ 新功能**

- 新增 `negative_prompt` 配置项，可自定义负面提示词

### v1.1

- 初始版本
- 支持 LLM 工具调用和 `/aiimg` 命令
- 支持多种图片比例
- 支持多 Key 轮询

## 出图展示区

<img width="1152" height="2048" alt="29889b7b184984fac81c33574233a3a9_720" src="https://github.com/user-attachments/assets/c2390320-6d55-4db4-b3ad-0dde7b447c87" />

<img width="1152" height="2048" alt="60393b1ea20d432822c21a61ba48d946" src="https://github.com/user-attachments/assets/3d8195e5-5d89-4a12-806e-8a81e348a96c" />

<img width="1152" height="2048" alt="3e5ee8d438fa797730127e57b9720454_720" src="https://github.com/user-attachments/assets/c270ae7f-25f6-4d96-bbed-0299c9e61877" />


本插件开发QQ群：215532038

<img width="1284" height="2289" alt="qrcode_1767584668806" src="https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee" />
