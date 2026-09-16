# 后台任务与历史图片联动

[返回插件首页](../README.md) · [配置与命令](configuration.md) · [工作台指南](studio.md)

## LLM 后台生图

后台模式只作用于 `aiimg_generate`、`gitee_draw_image`、`gitee_edit_image` 和 `aiimg_batch_generate`。`/文生图`、`/改图`、`/自拍`、`/批量` 等直接命令仍保持原同步行为。

```json
{
  "features": {
    "background_llm_image": {
      "enabled": true,
      "max_running": 2,
      "max_queued": 16
    }
  }
}
```

- `max_running` 是所有单图和 batch child 共用的图片 Provider 并发数，可设置 `1-30`；一般建议从 `2` 开始，再根据机器资源和上游限流情况调整。
- `max_queued` 按图片张数预留容量。例如一组 `4` 张批量任务会原子占用 `4` 个容量，容量不足时整组拒绝，不会只接一半。
- Tool 完成参数校验、完整提示词构建和输入图片固化后立即返回，真正的 planner、图片 Provider 调用和发送在后台执行。
- 用户继续聊天或询问照片时，Bot 能看到任务处于 `planning`、`queued`、`running`、`sending` 或终态，并能读取有界状态摘要；批量完整提示词可由只读 Tool `aiimg_task_status` 分页查询。
- 图片先作为独立 image-only 消息发送；只有原 conversation 和 ContextAware session 仍安全可用时，Bot 才会进入 Agent pipeline，按当前人格自然说明完成、部分成功或失败。模型请求失败、超时、ContextAware session 已清空或 conversation 已切换时，完成通知会静默终结，不发送固定统计话术，也不影响普通对话。
- 同一会话中多个任务同时完成时，终态回应会按 UMO 串行进入 Agent pipeline，避免抢写历史或乱序说话；普通用户消息不使用这把通知锁，仍可继续聊天。
- `/stop` 会取消当前会话中该用户的后台图片任务；成功的 `/reset`、`/new` 会通过发送闸门阻止晚到图片污染新会话。权限不足而失败的 reset 不会误取消任务。
- AstrBot 或插件重启后，尚未完成的 Provider 请求不会自动续跑或重复扣费，而是标记为 `interrupted`；恢复过程只收敛任务与通知账本，不重入旧会话或发送固定中断文案。
- 非优雅重启后若旧进程的 owner lease 尚未过期，插件不会阻塞 AstrBot 启动；它会低频后台重试，lease 过期后自动接管并收敛账本。
- 插件每 5 分钟输出不含提示词的后台健康摘要并执行 passive WAL checkpoint；账本异常、通知积压或健康检查连续失败时停止后台接单并保留同步路径。

> [!IMPORTANT]
> 后台模式默认关闭，只支持单 AstrBot 进程、单个有效 Gitee 插件 owner，以及 `aiocqhttp` / `weixin_oc`。AstrBot 开启 `provider_settings.streaming_response` 时会自动回退同步路径，因为流式回复无法可靠使用发送前后的确认 Hook。

> [!NOTE]
> QQ / 微信 adapter 当前没有暴露端到端 receipt 或幂等发送键。发送调用成功返回只代表 adapter transport accepted；发生 timeout、connection reset 或进程崩溃窗口时，任务会记录为 `unknown` 并禁止自动重发，避免重复图片。

## 群聊图片引用与主体特征保留

配合 **ContextAware >=3.6.0 / AstrBot >=4.26.8**，`aiimg_generate` 可使用当前会话图片目录中的历史图片，而不要求用户重新发送或引用。

| 用户意图                             | 工具模式及参考                                             |
| ------------------------------------ | ---------------------------------------------------------- |
| “把刚才 Alice 的猫图改成水彩”        | `mode=edit`，猫图 `subject`                                |
| “抱着上图的猫自拍”                   | `mode=selfie_ref`，猫图 `object`；固定人物参考自动保留在前 |
| “穿这件衣服，抱着刚才的猫自拍”       | `selfie_ref`，衣服 `clothing`、猫图 `object`               |
| “把你刚给我生成的背景换掉，其他不动” | `edit`，选对应生成结果 ID 为 `subject`                     |
| “再拍一张新自拍”                     | `selfie_ref`，需要保留的额外参考应重新明确选择             |

工具增加两个可选参数：`reference_image_ids` 和同长度的 `reference_roles`。角色支持 `subject/style/clothing/object/pose/background`。没有历史引用时原有调用方式保持不变；指定了引用但图片不可用、角色缺失或模式不兼容时，任务明确失败，不会省略图片或降级纯文生图。传参由聊天模型完成，用户无需输入 ID；自然语言选择仍需在具体模型上验证。

输入按“固定自拍身份 → 明确选择的参考 → 当前/引用附件”排列，附件按内容去重；显式选择历史图时不会自动混入 @头像。固定身份与显式不同角色保持各自位置，不会替换永久自拍参考。合计最多 8 张、单张 20 MiB、合计 64 MiB，包含身份图与当前附件。输入来自 ContextAware 保留的数据，可能已经过 Core 规范化或压缩，不承诺是原始上传像素；4K 输出参数保持原有规则。

多张带角色的输入只路由到已声明保留有序多参考的后端（Gemini native、Gitee edit、OpenAI chat image、GPT Image 原生 Images API、Vertex anonymous）。会静默只取首图、拼图或能力未知的后端不会用于这类任务；自动链路可继续尝试兼容后端，全部不支持时明确失败，不会少传一张图凑合生成。声明代表本地适配器完整传递图片；上游模型的数量限制和生成一致性仍由实际服务决定。

后台任务在接单时保存输入和哈希，后续不读取旧消息。**成功发送的单图和后台批量子图结果**会登记回 ContextAware，关联请求者、conversation、任务和父参考，供后续明确编辑。结果 ID 是短期索引；reset/new、插件重载或缓存过期后可能不可用。后台输入副本不受聊天缓存淘汰影响；进程重启/旧凭据失效后，不会恢复旧图片索引，任务元数据记录 `result_registration` 状态。

本版联动覆盖 `aiimg_generate` 单图同步/后台路径和 `aiimg_batch_generate` 后台批量的 `edit` / `selfie_ref`。例如“抱着上图的猫拍几张”：批量工具使用 `selfie_ref`、猫图 ID 和 `object` 用途，未指定数量时默认 4 张；每张共用同一份接单时保存的参考输入，规划不同动作和构图，分别登记成功发送的结果。历史参考批量要求后台模式生效；模式不可用时明确失败，不会丢掉参考继续生成。普通直接命令仍按原消息附件规则工作。仅安装 Gitee、未安装兼容 ContextAware 时，原有功能仍可使用，历史引用不可用。

### 按需提取动物和物体特征

仅对本次显式选择为 `object` 的参考图，插件可调用当前会话的视觉聊天模型，提取脸型、眼睛比例、花纹、材质和风格化特征，再连同参考图片交给生图模型。这样“抱着这只猫”可以保留具体主体的视觉特征，降低被替换成同类别普通动物的概率。人物身份、服装、画风等其它角色保持各自用途；用户明确要求改变的特征仍以用户要求为准。

- 后台任务先接单，再在后台识图；整批仅调用一次识图，全部子任务共享结果，不扫描群聊全部图片。
- 识图只发送选中 `object` 的最长边 768 像素预览，不携带聊天历史或工具；生图仍使用已保存的参考图字节，输出尺寸不变。
- 识图最长等待 45 秒；当前聊天模型明确不支持图片时跳过，超时或解析失败也保留原始参考图继续生成。任务元数据 `reference_vision` 记录 `described`、`vision_unavailable`、`vision_timeout`、`vision_failed` 或 `not_needed`。
- 使用视觉聊天模型会增加一次模型调用和一定出图等待时间；后台正常对话不受阻塞。同步单图自拍会在调用内等待这一步。
- Gemini native 为每张输入明确编号，并使用中性的编辑/合成指令。自拍模板避免把所有额外参考一概当作服装或场景。

具体主体的一致性仍受上游生图模型影响；传图成功、特征提取成功均不等于逐像素复刻。复杂风格转换、遮挡或多个主体仍需检查实际成图。
