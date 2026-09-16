# 第一次配置: 先加服务商, 再绑定功能

[返回插件首页](../README.md) · [美年达申请与模板说明](providers.md#美年达) · [完整配置参数](configuration.md)

本页使用美年达做演示, 截图来自 **AstrBot 原生插件配置界面**. 示例中的地址、Key 和人物配置都是虚构值, 不能直接拿去请求接口.

## 按这个顺序操作

1. 打开 AstrBot 的 **插件管理**, 找到 `astrbot_plugin_gitee_aiimg`, 点击 **配置**.
2. 滚动到最底部 **模型服务商**, 点击 **添加条目**, 先把模型连接保存好.
3. 再回到上方的 **文生图服务商链路 / 改图服务商链路 / 自拍服务商链路 / 视频服务商链路**, 点击 **添加条目**.
4. 在链路的 **服务商 ID** 中填写已经保存的 `id`. 这里是手填文本, 必须逐字一致.
5. 点底部 **保存并关闭**, 再发一条测试指令. 只添加服务商但不加入链路, 该模式不会调用它.

## 先添加美年达香蕉

香蕉真人写实和自拍使用 **Gemini 原生** 模板. `id` 是后面绑定链路的钥匙, 示例使用 `meinianda_banana`.

![AstrBot 原生配置: 美年达香蕉服务商](assets/tutorials/native-config-01-banana.png)

图中红框的含义:

- `服务商 ID`: 填 `meinianda_banana`, 后面自拍链要使用完全相同的值.
- `API Base URL`: 填 `https://meinianda.top`.
- `API Key 池`: 填自己的美年达令牌, 图中 `DEMO_ONLY_NOT_A_REAL_KEY` 只是演示占位.
- `模型名称`: 填分组实际提供的模型 ID, 示例为 `gemini-3.1-flash-image`.

## 再添加 GPT Image

美年达的 `gpt-image-2`、`gpt-image-2.5-flare`、`gpt-image-2.5-sunburst` 等 GPT Image 系列, 统一使用 **OpenAI 图** 模板. 示例单独使用 `meinianda_gpt_image`, 不要复用香蕉的 ID.

![AstrBot 原生配置: 美年达 GPT Image 服务商](assets/tutorials/native-config-02-gpt-image.png)

`Base URL` 示例为 `https://meinianda.top/v1`; 模型名按当前分组填写. 只使用香蕉自拍时, GPT Image 这一项可以先不添加.

## 把香蕉绑定到自拍

回到上方的 **自拍服务商链路**, 添加一个 `Provider`, 在 **服务商 ID** 填 `meinianda_banana`. `output` 留空就使用自拍模式的默认输出.

![AstrBot 原生配置: 自拍链路填写与服务商相同的 ID](assets/tutorials/native-config-03-selfie-chain.png)

自拍模式还需要一张 Bot 参考人像, 并打开 **启用自拍参考照** 和 **允许 LLM 调用自拍参考照**. 参考图可以在同一页的文件管理控件上传.

![AstrBot 原生配置: 开启自拍并上传参考人像](assets/tutorials/native-config-05-selfie-settings.png)

## 配置文生图主备链路

文生图可以让 GPT Image 做主用, 香蕉做备用. 按顺序添加两个 `Provider`: 第一项填 `meinianda_gpt_image`, 第二项填 `meinianda_banana`. 上游失败时插件才会按顺序回退.

![AstrBot 原生配置: 文生图主用 GPT Image, 备用香蕉](assets/tutorials/native-config-04-draw-chain.png)

改图也可以添加 `meinianda_banana`. 视频链路要填写视频服务商的 ID, 不能把图片服务商当成视频服务商.

## 让 Bot 一次拍一组照片

开启对应模式的 **允许 LLM 调用** 后, 直接用自然语言描述一组目标即可:

> 给我拍 4 张你在窗边的照片, 同一套衣服, 每张换一个姿势、表情和机位.

LLM 会调用 `aiimg_batch_generate` 一次规划 4 个不同镜头, 共享同一套 Bot 身份, 再逐张生成. 没写数量时默认 4 张, 单次上限由 `features.batch.max_count` 控制. 打开 [后台生图](background-and-context.md#llm-后台生图) 后, 接单消息会先返回, 图片在后台完成并主动发送.

这和 `/批量4 自拍 ...` 不一样: 直接命令按同一规格批量提交, 自然语言 LLM 批量才会先规划不同姿势、表情和构图.

## 出问题先查这三项

- 服务商保存了, 但链路里的 `provider_id` 是否和 `providers[].id` 完全一致.
- 香蕉是否用了 **Gemini 原生**, GPT Image 是否用了 **OpenAI 图**.
- 自拍是否已有参考人像, 且 **启用自拍参考照** 和对应链路都已打开.
