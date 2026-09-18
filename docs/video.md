# 视频与自拍视频

[返回插件首页](../README.md) · [接入图片服务商](providers.md) · [形象库](studio.md#形象库与日程联动)

## 视频生成

先添加视频服务商, 再加入 `features.video.chain`. 没有图片时使用文生视频; 发送或引用图片后, 同一指令使用图片作为参考. 视频预设可在 `features.video.presets` 中配置, 第一个词命中预设名时按“视频预设 + 额外提示词”处理.

```text
/视频 镜头缓慢推进，人物轻微转头
/视频 @agnes_video 黄昏街景，镜头跟拍
/视频 电影感 拉近镜头，轻微风吹头发
/视频预设列表
```

### Happy Horse 1.1

在 `https://ztyunjuan.com` 使用 `happy-horse-1.1` 时, 选择 **OpenAI 视频**模板 (`sora2_video`), API 地址填写 `https://ztyunjuan.com`, 然后将该服务商加入视频链路. 插件对该站点的 Happy Horse 模型使用 JSON 请求, 将消息中的单张图片编码为 `start_frame` Data URI, 无需上传第三方图床. 其它 OpenAI 视频渠道仍按原协议上传 `input_reference` 文件.

按服务方文档, 时长为 3-15 的整数秒, 清晰度支持 720p / 1080p, 不支持 360p. 当前模板的 `size` 是像素尺寸, 不是清晰度名称: 720p 横屏填 `1280x720`, 竖屏填 `720x1280`; 1080p 横屏填 `1920x1080`, 竖屏填 `1080x1920`. 其它画幅使用服务方模型文档列出的精确尺寸, 不按短边自行推算. `extra_body` 可填写 `{"audio":true}` 启用结果音轨.

普通多图参考可按服务方文档在 `extra_body.reference_images` 配置 URL 或 Data URI, 最多 9 张; 聊天消息自动取图仍只传单张首帧. 已发送消息图片时, 不要在 `extra_body` 同时配置 `start_frame`, `input_reference` 或普通图片参考, 插件会拒绝冲突输入. 1.1 不支持上传音频参考或参考视频.

网关的 `seconds` 实际要求字符串, 插件保留字符串传输并校验整数范围. 模型清单列出 Happy Horse 不代表当前账号分组的上游渠道可用; 若 JSON 请求返回上游只支持 Seedance / Kling / Veo, 请让服务方检查 Happy Horse 的模型映射和渠道, 不要反复创建任务或修改分辨率来重试.

### Agnes 注册与配置

> [!TIP]
> **Agnes Video 2.5 Flash 限时免费体验**: 直接到 [Agnes AI 平台](https://platform.agnes-ai.com/) 注册, 在平台创建 API Key 后即可接入本插件. 截至 2026-09-16, [官方模型文档](https://wiki.agnes-ai.com/en/docs/agnes-video-25-flash) 标明当前价格为 `$0/second`. 活动截止时间、可用额度、限流及后续价格以平台最新公告和账号页面为准, 不代表永久免费或无限量使用.

**注册地址和 API 地址不同**, 不要把 `platform.agnes-ai.com` 填进 API 地址栏.

| 配置项            | 填写值                                       |
| ----------------- | -------------------------------------------- |
| 平台注册与管理    | `https://platform.agnes-ai.com/`             |
| 插件模板          | **Agnes 视频**, `__template_key=agnes_video` |
| API 地址          | `https://apihub.agnes-ai.com/v1`             |
| API Key           | 在 Agnes 平台创建的 Key                      |
| 模型 ID           | `agnes-video-2.5-flash`                      |
| 生成模式          | `auto`                                       |
| 时长              | 字符串 `"4"` 至 `"12"`, 模板默认 `"5"`       |
| 分辨率            | `720P`, Flash 不支持 `1080P` 或 `4K`         |
| 画幅              | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`  |
| 最短 API 请求间隔 | `61` 秒, 创建与查询共用限速                  |
| 任务总超时        | `1800` 秒                                    |

在工作台的 **服务商 > 模型连接** 中添加 **Agnes 视频**, 按表填写并保存, 再到 **回退链路 > 视频** 添加该服务商. 在 AstrBot 插件配置中开启视频和 LLM 视频调用; 若 AstrBot 工具管理页停用了 `grok_generate_video`, 也要启用该工具.

以下是插件配置片段, **不是直接发给 Agnes 的请求体**. 将 provider 追加到现有 `providers`, 将视频设置合并到现有 `features.video`, 不要覆盖原有图片服务商、形象或其他功能配置. Key 使用自己的值, 不要提交到公开仓库.

```json
{
  "providers": [
    {
      "__template_key": "agnes_video",
      "id": "agnes_video",
      "label": "Agnes Video 2.5 Flash",
      "base_url": "https://apihub.agnes-ai.com/v1",
      "api_key": "YOUR_AGNES_API_KEY",
      "model": "agnes-video-2.5-flash",
      "mode": "auto",
      "seconds": "5",
      "size": "720P",
      "aspect_ratio": "16:9",
      "request_interval_seconds": 61,
      "timeout_seconds": 1800,
      "request_timeout_seconds": 120,
      "poll_interval_seconds": 61,
      "max_retries": 3,
      "extra_body": {}
    }
  ],
  "features": {
    "video": {
      "enabled": true,
      "llm_tool_enabled": true,
      "chain": [
        {
          "__template_key": "provider",
          "provider_id": "agnes_video"
        }
      ]
    }
  }
}
```

先发送 `/视频 @agnes_video 日光穿过窗帘, 镜头缓慢向前移动` 测试文生视频. 再发送或引用一张图片, 使用 `/视频 @agnes_video 让画面自然动起来` 测试图生视频. `@agnes_video` 指向示例服务商 ID; 修改 ID 后, 指令与链路中的 ID 也要同步修改.

### Agnes 参数与限流

模板目前适配 **`agnes-video-2.5-flash`**, 不要将模型 ID 替换成 `agnes-video-2.5` 或其它名称. 每次只生成 1 个视频, 插件固定 `n=1`. `720P` 是分辨率档位, 实际成片像素、时长和首帧构图处理以上游返回文件为准.

- `auto`: 没有图片时使用 `text`, 有消息图片时使用首帧 `keyframe`; 额外请求体配置了 `images` 或 `audios` 时可自动使用 `reference`.
- `keyframe`: 使用首帧或尾帧控制. 消息图片自动编码为本次首帧, 不会上传到第三方图床; 已有消息图片时, 不要同时在额外请求体里固定另一个 `first_frame`.
- `reference`: 使用图片或音频参考. 最多 5 张图片和 3 段音频, Flash 不接受 `videos` 视频参考.
- **额外请求体**支持 `seed`, `mode`, `first_frame`, `last_frame`, `images`, `audios` 等模型字段. 例如固定随机种子填写 `{"seed":42}`, `seed` 为整数; 消息首帧图片需小于 15 MB, 手填媒体 URL 需能被 Agnes 访问且在任务完成前保持有效.
- 不要混用不同模式的媒体字段, 例如 `keyframe` 不同时填写 `images` 或 `audios`. Flash 文档没有通用的 `quality=max` 档位, 不要照搬图片模型的质量参数.

默认最短请求间隔为 **61 秒**, 创建任务和查询状态共用同一 Key 的持久化限速, 满足 `RPM=1`; 插件重载不会立即重置额度. 若同一 Key 还用于其它插件、服务或手动请求, 本插件无法代它们统一限速, 需合并计算额度. 不要在 `RPM=1` 的账号上照搬每 1 至 2 秒轮询的示例.

默认总超时为 1800 秒, 查询临时网络错误和限流最多重试 3 次, 重试仍遵守请求间隔. 创建请求结果不明、轮询超时或查询中断时停止链路, 不重新创建任务; 日志保留已知的 `video_id`, 可到平台检查. 上游队列满时等待后再手动尝试, 不要连续提交同一任务.

### 先自拍再转视频

需要 Bot 本人出镜时, 直接说“拍个你跳舞的视频我看看”. 插件按以下顺序执行, 不需要手动先要一张照片再发第二条指令:

```text
当前 Bot 形象 + 接单时的日程穿搭
  -> 自拍图片链路生成动作起始底图
  -> 生成后的底图作为视频参考
  -> 视频链路生成并发送最终视频
```

先完成以下配置:

1. 在 **形象库** 保存 Bot 身份参考并选择当前形象, 或使用原有 `/自拍参考` 和插件配置中的参考图.
2. 为 **自拍链路** 配置支持参考图编辑的图片服务商. 只配置 Agnes 视频服务商不够, 它不负责生成自拍底图.
3. 开启 `features.selfie.enabled`、`features.selfie.llm_tool_enabled`、`features.video.enabled` 和 `features.video.llm_tool_enabled`.
4. 需要日程穿搭时安装并配置 [astrbot_plugin_life_scheduler](https://github.com/muyouzhi6/astrbot_plugin_life_scheduler), 确保有可读取的当天缓存. 没有日程缓存仍可用身份参考拍摄, 但不会虚构一份已读取的穿搭.

自拍开关和链路示例如下. `my_image_editor` 必须替换成已经配置好的图片服务商 ID; 合并这些字段时保留原有自拍参考和其它设置.

```json
{
  "features": {
    "selfie": {
      "enabled": true,
      "llm_tool_enabled": true,
      "chain": [
        {
          "__template_key": "provider",
          "provider_id": "my_image_editor"
        }
      ],
      "use_edit_chain_when_empty": false
    }
  }
}
```

LLM 视频工具仍叫 `grok_generate_video`, 名字沿用旧版, 实际按视频链路选择服务商, 不限定 Grok. 它的 `mode=selfie` 是**插件复合任务模式**, 不是 Agnes provider 的 `mode`; Agnes provider 保持 `auto` 即可. 下面是聊天模型应调用的工具参数, 普通用户只需自然语言提出视频要求:

```json
{
  "prompt": "参考人物自然跳舞, 摆动双臂并轻轻迈步, 保持面容和服装一致, 固定全身镜头",
  "mode": "selfie",
  "selfie_prompt": "全身站姿, 为跳舞留出活动空间, 穿今天日程中的衣服, 单人单张画面"
}
```

`selfie_prompt` 描述静态起始画面, `prompt` 描述视频运动. 形象、参考图和缓存日程在接单时读取并固定; 用户明确指定穿搭或场景时优先采用用户要求. 中间底图进入图片画廊, 聊天只发送最终视频. 缺少身份参考或底图生成失败时结束任务, 不会改成无参考文生视频. 人物和运动效果仍取决于所选生成模型, 不承诺每个动作都能准确复现.

自拍图片阶段和视频阶段使用各自服务商, **Agnes 视频限时免费不代表自拍图片阶段也免费**. 普通素材动画使用工具的 `mode=auto`; 不要为同一个自拍视频同时调用图片工具和视频工具, 避免重复生成.

### 旧视频渠道升级

3365 与 SD2.0 专用模板已移除, 升级时请移除旧服务商和链路引用. 美年达等 OpenAI Videos 渠道继续使用 `sora2_video`, xAI 官方继续使用 `grok_video`. 历史视频不会因模板移除而删除.
