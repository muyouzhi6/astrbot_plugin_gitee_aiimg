# AstrBot Gitee AI 图像生成插件 （有免费额度）

本插件为 AstrBot 接入 Gitee AI 的图像生成能力，支持通过自然语言或指令调用，支持多key轮询。

## 功能特性

- 支持通过 LLM 自然语言调用生成图片。
- 支持通过指令 `/aiimg` 生成图片。
- 支持多种图片比例和分辨率。
- 支持自定义模型。


## 配置

在 AstrBot 的管理面板中配置以下参数：

- `base_url`: Gitee AI API 地址，默认为 `https://ai.gitee.com/v1`。
- `api_key`: Gitee AI API Key，请在 Gitee AI 控制台申请。
- `model`: 使用的模型名称，默认为 `z-image-turbo`。
- `size`: 默认图片大小，例如 `1024x1024`。
- `num_inference_steps`: 推理步数，默认 9。


## Gitee AI API Key获取方法：
1.访问https://ai.gitee.com/serverless-api?model=z-image-turbo

2.<img width="2241" height="1280" alt="PixPin_2025-12-05_16-56-27" src="https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841" />

3.免费额度<img width="240" height="63" alt="PixPin_2025-12-05_16-56-49" src="https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb" />

4.可以涩涩，警惕违规被举报

5.好用可以给个🌟

##图像尺寸只支持以下，如果不在其中会报错
    "1:1 (256×256)": (256, 256),
    "1:1 (512×512)": (512, 512),
    "1:1 (1024×1024)": (1024, 1024),
    "1:1 (2048×2048)": (2048, 2048),
    "4:3 (1152×896)": (1152, 896),
    "4:3 (2048×1536)": (2048, 1536),
    "3:4 (768×1024)": (768, 1024),
    "3:4 (1536×2048)": (1536, 2048),
    "3:2 (2048×1360)": (2048, 1360),
    "2:3 (1360×2048)": (1360, 2048),
    "16:9 (1024×576)": (1024, 576),
    "16:9 (2048×1152)": (2048, 1152),
    "9:16 (576×1024)": (576, 1024),
    "9:16 (1152×2048)": (1152, 2048),

## 使用方法

### 指令调用

```
/aiimg <提示词> [比例]
```

示例：
- `/aiimg 一个可爱的女孩` (使用默认比例 1:1)
- `/aiimg 一个可爱的女孩 16:9`
- `/aiimg 赛博朋克风格的城市 9:16`


### 自然语言调用

直接与 bot 对话，例如：
- "帮我画一张小猫的图片"
- "生成一个二次元风格的少女"

## 注意事项

- 请确保您的 Gitee AI 账号有足够的额度。

- 生成的图片会临时保存在 `data/plugins/astrbot_plugin_gitee_aiimg/images` 目录下。


### 出图展示区

<img width="1152" height="2048" alt="29889b7b184984fac81c33574233a3a9_720" src="https://github.com/user-attachments/assets/c2390320-6d55-4db4-b3ad-0dde7b447c87" />

<img width="1152" height="2048" alt="60393b1ea20d432822c21a61ba48d946" src="https://github.com/user-attachments/assets/3d8195e5-5d89-4a12-806e-8a81e348a96c" />

<img width="1152" height="2048" alt="3e5ee8d438fa797730127e57b9720454_720" src="https://github.com/user-attachments/assets/c270ae7f-25f6-4d96-bbed-0299c9e61877" />






