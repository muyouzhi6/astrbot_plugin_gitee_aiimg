# AstrBot Gitee AI 图像生成插件

本插件为 AstrBot 接入 Gitee AI 的图像生成能力，支持通过自然语言或指令调用。

## 功能特性

- 支持通过 LLM 自然语言调用生成图片。
- 支持通过指令 `/aiimg` 生成图片。
- 支持多种图片比例和分辨率。
- 支持自定义模型。

## 安装

1. 将本插件目录 `astrbot_plugin_gitee_aiimg` 放入 AstrBot 的 `data/plugins/` 目录下。
2. 重启 AstrBot。

## 配置

在 AstrBot 的管理面板中配置以下参数：

- `base_url`: Gitee AI API 地址，默认为 `https://ai.gitee.com/v1`。
- `api_key`: Gitee AI API Key，请在 Gitee AI 控制台申请。
- `model`: 使用的模型名称，默认为 `z-image-turbo`。
- `size`: 默认图片大小，例如 `1024x1024`。
- `num_inference_steps`: 推理步数，默认 9。

## 使用方法

### 指令调用

```
/aiimg <提示词> [比例]
```

示例：
- `/aiimg 一个可爱的女孩` (使用默认比例 1:1)
- `/aiimg 一个可爱的女孩 16:9`
- `/aiimg 赛博朋克风格的城市 9:16`

支持的比例：
- 1:1 (1024x1024, 512x512, etc.)
- 4:3
- 3:4
- 3:2
- 2:3
- 16:9
- 9:16

### 自然语言调用

直接与 bot 对话，例如：
- "帮我画一张小猫的图片"
- "生成一个二次元风格的少女"

## 注意事项

- 请确保您的 Gitee AI 账号有足够的额度。
- 生成的图片会临时保存在 `data/plugins/astrbot_plugin_gitee_aiimg/images` 目录下。