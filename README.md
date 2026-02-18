# AstrBot Gitee AI 图像生成插件（多服务商 / 多网关）

> **当前版本**：v4.1.0（全新配置结构，和 v3/v2 不兼容，需要重新在 WebUI 配置）

本插件支持：
- 文生图（Text-to-Image）
- 图生图/改图（Image-to-Image/Edit）
- 自拍参考照模式（参考人像 + 额外参考图）
- 视频生成（Image-to-Video，Grok imagine）

核心设计：**服务商实例（providers）** 与 **功能链路（features.*.chain）** 分离。你可以配置同一模型的多家服务商，并按顺序兜底切换。

---

## v4 配置（重点）

### 1) 先配置 providers（在配置面板最底部）

你可以添加多个服务商实例，每个实例都要填一个唯一的 `id`（用户自定义字符串，必须唯一）。

模板包含（按你的生态做了拆分）：
- Gemini 原生（generateContent）
- Gemini OpenAI 兼容（Images / Chat）
- OpenAI 兼容通用（Images / Chat）
- Flow2API（Chat SSE 出图）
- Grok2API（/v1/images/generations）
- Gitee（Images）
- Gitee 异步改图（/async/images/edits）
- 即梦/豆包聚合（jimeng）
- Grok 视频（chat.completions）
- 魔搭社区（OpenAI兼容，按实际网关能力决定是否可用）

### 2) 再配置 features（在配置面板顶部）

- `features.draw.chain`：文生图链路
- `features.edit.chain`：改图链路
- `features.selfie.chain`：自拍链路（可选；留空可复用改图链路）
- `features.video.chain`：视频链路

链路按顺序兜底：第一个是主用，失败自动切到后面的 provider。
若 `features.selfie.use_edit_chain_when_empty=true`：自拍链会自动把改图链补成后备兜底（去重后追加）。

### 3) 可选：关闭某个功能 / 关闭对应 LLM 调用

- `features.<mode>.enabled`：是否启用该功能（命令也会受影响）
- `features.<mode>.llm_tool_enabled`：是否允许 LLM 调用该功能（命令不受影响）

---

## 指令用法（v4）

### 文生图

```
/aiimg [@provider_id] <提示词> [比例]
```

示例：
- `/aiimg 一个可爱的女孩 9:16`
- `/aiimg @gitee 一只猫 1:1`

不填比例时：将使用 `features.draw.default_output`（以及链路里单个 provider 的 `output` 覆盖）来决定默认输出。

补充说明：
- 比例对应尺寸可在 `features.draw.ratio_default_sizes` 中覆盖（仅对 `/aiimg 比例` 生效）。
- 若配置了不支持的尺寸，会自动回退到该比例的默认尺寸，并在日志中提示。

如果平台临时异常导致“生成成功但图片没发出去”，可用：
```
/重发图片
```
重发最近一次生成/改图结果（不会重新生成，不消耗次数）。

### 改图/图生图

发送/引用图片后：
```
/aiedit [@provider_id] <提示词>
```

示例：
- 发送图片 + `/aiedit 把背景换成海边`
- 发送图片 + `/aiedit @grok2api 把照片转成动漫风格`

预设命令（来自 `features.edit.presets`，会动态注册成 `/手办化` 这种命令）：
```
/预设列表
/手办化 [@provider_id] [额外提示词]
```

### 自拍参考照

1) 设置参考照（二选一）：
- 聊天设置：发送图片 + `/自拍参考 设置`
- WebUI 上传：`features.selfie.reference_images`

2) 生成自拍：
```
/自拍 [@provider_id] <提示词>
```

触发规则说明：
- 只有明确 `/自拍`（或 LLM tool 传 `mode=selfie_ref`）会强制走自拍参考照流程。
- `mode=auto` 仅在“提示词明确指向自拍 + 已配置参考照”时才会自动尝试自拍；否则回退为文生图/改图。

### 视频生成

发送/引用图片后：
```
/视频 [@provider_id] <提示词>
/视频 [@provider_id] <预设名> [额外提示词]
/视频预设列表
```

---

## 注意事项

- 如果你没有配置 providers 或链路为空：插件会提示你去 WebUI 补配置。
- 网关是否支持某个接口（尤其是 images.edit）取决于服务商实现本身；插件会自动兜底到后续 provider。
- `@provider_id` 仅是“临时指定一次使用哪个 provider”，不会改变你的默认链路顺序。
- Gitee 文生图仅支持白名单尺寸；若输出尺寸不合法会自动兜底到可用尺寸并记录日志。

---

## Gitee AI API Key 获取方法（保留原文）

1.访问<https://ai.gitee.com/serverless-api?model=z-image-turbo>

2.<img width="2241" height="1280" alt="PixPin_2025-12-05_16-56-27" src="https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841" />

3.免费额度<img width="240" height="63" alt="PixPin_2025-12-05_16-56-49" src="https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb" />

4.可以涩涩，警惕违规被举报

5.好用可以给个🌟

---

## 支持的图像尺寸（Gitee，保留原文）

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

---

## 出图展示区（保留原文）

<img width="1152" height="2048" alt="29889b7b184984fac81c33574233a3a9_720" src="https://github.com/user-attachments/assets/c2390320-6d55-4db4-b3ad-0dde7b447c87" />

<img width="1152" height="2048" alt="60393b1ea20d432822c21a61ba48d946" src="https://github.com/user-attachments/assets/3d8195e5-5d89-4a12-806e-8a81e348a96c" />

<img width="1152" height="2048" alt="3e5ee8d438fa797730127e57b9720454_720" src="https://github.com/user-attachments/assets/c270ae7f-25f6-4d96-bbed-0299c9e61877" />

本插件开发QQ群：215532038

<img width="1284" height="2289" alt="qrcode_1767584668806" src="https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee" />
