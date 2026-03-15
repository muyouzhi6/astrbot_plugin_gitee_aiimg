# AstrBot Gitee AI 鍥惧儚鐢熸垚鎻掍欢锛堝鏈嶅姟鍟?/ 澶氱綉鍏筹級

> **当前版本**：v4.2.12（全新配置结构，和 v3/v2 不兼容，需要重新在 WebUI 配置）
鏈彃浠舵敮鎸侊細
- 鏂囩敓鍥撅紙Text-to-Image锛?- 鍥剧敓鍥?鏀瑰浘锛圛mage-to-Image/Edit锛?- 鑷媿鍙傝€冪収妯″紡锛堝弬鑰冧汉鍍?+ 棰濆鍙傝€冨浘锛?- 瑙嗛鐢熸垚锛圛mage-to-Video锛孏rok imagine锛?
鏍稿績璁捐锛?*鏈嶅姟鍟嗗疄渚嬶紙providers锛?* 涓?**鍔熻兘閾捐矾锛坒eatures.*.chain锛?* 鍒嗙銆備綘鍙互閰嶇疆鍚屼竴妯″瀷鐨勫瀹舵湇鍔″晢锛屽苟鎸夐『搴忓厹搴曞垏鎹€?
---

## v4 閰嶇疆锛堥噸鐐癸級

### 1) 鍏堥厤缃?providers锛堝湪閰嶇疆闈㈡澘鏈€搴曢儴锛?
浣犲彲浠ユ坊鍔犲涓湇鍔″晢瀹炰緥锛屾瘡涓疄渚嬮兘瑕佸～涓€涓敮涓€鐨?`id`锛堢敤鎴疯嚜瀹氫箟瀛楃涓诧紝蹇呴』鍞竴锛夈€?
妯℃澘鍖呭惈锛堟寜浣犵殑鐢熸€佸仛浜嗘媶鍒嗭級锛?- Gemini 鍘熺敓锛坓enerateContent锛?- Vertex AI Anonymous锛圙oogle Console 閫嗗悜锛屾棤 Key锛涢渶鑳借闂?Google锛?- Gemini OpenAI 鍏煎锛圛mages / Chat锛?- OpenAI 鍏煎閫氱敤锛圛mages / Chat锛?- OpenAI鍏煎-瀹屾暣璺緞锛堟墜濉畬鏁?endpoint URL锛?- Flow2API锛圕hat SSE 鍑哄浘锛?- Grok2API锛?v1/images/generations锛?- Gitee锛圛mages锛?- Gitee 寮傛鏀瑰浘锛?async/images/edits锛?- 鍗虫ⅵ/璞嗗寘鑱氬悎锛坖imeng锛?- Grok 瑙嗛锛坈hat.completions锛?- 榄旀惌绀惧尯锛圤penAI鍏煎锛屾寜瀹為檯缃戝叧鑳藉姏鍐冲畾鏄惁鍙敤锛?
閫夋嫨寤鸿锛堜粠涓婂埌涓嬩紭鍏堢骇锛夛細
- 浣犵殑鏈嶅姟鍟嗗疄鐜颁簡鏍囧噯 `POST /v1/images/generations` / `POST /v1/images/edits`锛氱敤 `OpenAI 鍏煎閫氱敤锛圛mages锛塦
- 浣犵殑鏈嶅姟鍟嗕笉瀹炵幇 Images API锛屼絾浼氬湪 Chat 鍥炲閲岃繑鍥炲浘鐗囷紙markdown/data:image/base64/URL锛夛細鐢?`OpenAI 鍏煎锛圕hat 鍑哄浘瑙ｆ瀽锛塦
- 浣犵殑鏈嶅姟鍟嗚矾寰勪笉鏍囧噯锛堝甫鍓嶇紑銆佷笉鏄?/v1/...锛夛細鐢?`OpenAI鍏煎-瀹屾暣璺緞`
- 浣犵洿杩?Gemini 瀹樻柟 generateContent锛氱敤 `Gemini 鍘熺敓锛坓enerateContent锛塦
- 浣犲笇鏈涗娇鐢?Vertex AI Anonymous锛堟棤闇€ Key锛屼絾渚濊禆 Google + recaptcha锛夛細鐢?`Vertex AI Anonymous`

濡傛灉浣犻渶瑕佸畬鍏ㄨ嚜瀹氫箟璇锋眰璺緞锛堣€屼笉鏄彧濉?`base_url`锛夛紝鍙娇鐢?`OpenAI鍏煎-瀹屾暣璺緞`锛?
```json
{
  "id": "custom_full_url",
  "__template_key": "openai_full_url_images",
  "full_generate_url": "https://api.example.com/v1/images/generations",
  "full_edit_url": "https://api.example.com/v1/images/edits",
  "api_keys": ["sk-xxx"],
  "model": "gpt-image-1",
  "supports_edit": true,
  "timeout": 120,
  "max_retries": 2,
  "default_size": "1024x1024",
  "extra_body": {}
}
```

璇存槑锛?- `full_generate_url` / `full_edit_url` 蹇呴』鏄畬鏁?endpoint锛堝寘鍚矾寰勶級锛屼緥濡?`.../v1/images/generations`銆乣.../v1/images/edits`銆?- `full_edit_url` 鍙暀绌猴紝鐣欑┖鏃朵細澶嶇敤 `full_generate_url`锛堥€傜敤浜庣敓鎴愬拰鏀瑰浘鍏辩敤鍚屼竴璺緞鐨勭綉鍏筹級銆?
### 2) 鍐嶉厤缃?features锛堝湪閰嶇疆闈㈡澘椤堕儴锛?
- `features.draw.chain`锛氭枃鐢熷浘閾捐矾
- `features.edit.chain`锛氭敼鍥鹃摼璺?- `features.selfie.chain`锛氳嚜鎷嶉摼璺紙鍙€夛紱鐣欑┖鍙鐢ㄦ敼鍥鹃摼璺級
- `features.video.chain`锛氳棰戦摼璺?
閾捐矾鎸夐『搴忓厹搴曪細绗竴涓槸涓荤敤锛屽け璐ヨ嚜鍔ㄥ垏鍒板悗闈㈢殑 provider銆?鑻?`features.selfie.use_edit_chain_when_empty=true`锛氳嚜鎷嶉摼浼氳嚜鍔ㄦ妸鏀瑰浘閾捐ˉ鎴愬悗澶囧厹搴曪紙鍘婚噸鍚庤拷鍔狅級銆?
### 3) 鍙€夛細鍏抽棴鏌愪釜鍔熻兘 / 鍏抽棴瀵瑰簲 LLM 璋冪敤

- `features.<mode>.enabled`锛氭槸鍚﹀惎鐢ㄨ鍔熻兘锛堝懡浠や篃浼氬彈褰卞搷锛?- `features.<mode>.llm_tool_enabled`锛氭槸鍚﹀厑璁?LLM 璋冪敤璇ュ姛鑳斤紙鍛戒护涓嶅彈褰卞搷锛?- `private_text_notice_in_private`锛氭槸鍚﹀湪绉佽亰鍙戦€佲€滃紑濮?澶辫触鈥濇枃瀛楁彁绀猴紱榛樿 `false`锛岀兢鑱婃案涓嶅彂閫侊紝涓旀彁绀哄彂閫佸け璐ヤ笉浼氬奖鍝嶇敓鍥?鏀瑰浘/瑙嗛鏈韩銆?
---

## 鎸囦护鐢ㄦ硶锛坴4锛?
### 鏂囩敓鍥?
```
/aiimg [@provider_id] <鎻愮ず璇? [姣斾緥]
```

绀轰緥锛?- `/aiimg 涓€涓彲鐖辩殑濂冲 9:16`
- `/aiimg @gitee 涓€鍙尗 1:1`

涓嶅～姣斾緥鏃讹細灏嗕娇鐢?`features.draw.default_output`锛堜互鍙婇摼璺噷鍗曚釜 provider 鐨?`output` 瑕嗙洊锛夋潵鍐冲畾榛樿杈撳嚭銆?
琛ュ厖璇存槑锛?- 姣斾緥瀵瑰簲灏哄鍙湪 `features.draw.ratio_default_sizes` 涓鐩栵紙浠呭 `/aiimg 姣斾緥` 鐢熸晥锛夈€?- 鑻ラ厤缃簡涓嶆敮鎸佺殑灏哄锛屼細鑷姩鍥為€€鍒拌姣斾緥鐨勯粯璁ゅ昂瀵革紝骞跺湪鏃ュ織涓彁绀恒€?
濡傛灉骞冲彴涓存椂寮傚父瀵艰嚧鈥滅敓鎴愭垚鍔熶絾鍥剧墖娌″彂鍑哄幓鈥濓紝鍙敤锛?```
/閲嶅彂鍥剧墖
```
閲嶅彂鏈€杩戜竴娆＄敓鎴?鏀瑰浘缁撴灉锛堜笉浼氶噸鏂扮敓鎴愶紝涓嶆秷鑰楁鏁帮級銆?
### 鏀瑰浘/鍥剧敓鍥?
鍙戦€?寮曠敤鍥剧墖鍚庯細
```
/aiedit [@provider_id] <鎻愮ず璇?
```

绀轰緥锛?- 鍙戦€佸浘鐗?+ `/aiedit 鎶婅儗鏅崲鎴愭捣杈筦
- 鍙戦€佸浘鐗?+ `/aiedit @grok2api 鎶婄収鐗囪浆鎴愬姩婕鏍糮

棰勮鍛戒护锛堟潵鑷?`features.edit.presets`锛屼細鍔ㄦ€佹敞鍐屾垚 `/鎵嬪姙鍖朻 杩欑鍛戒护锛夛細
```
/棰勮鍒楄〃
/鎵嬪姙鍖?[@provider_id] [棰濆鎻愮ず璇峕
```

### 鑷媿鍙傝€冪収

1) 璁剧疆鍙傝€冪収锛堜簩閫変竴锛夛細
- 鑱婂ぉ璁剧疆锛氬彂閫佸浘鐗?+ `/鑷媿鍙傝€?璁剧疆`
- WebUI 涓婁紶锛歚features.selfie.reference_images`

2) 鐢熸垚鑷媿锛?```
/鑷媿 [@provider_id] <鎻愮ず璇?
```

瑙﹀彂瑙勫垯璇存槑锛?- 鍙湁鏄庣‘ `/鑷媿`锛堟垨 LLM tool 浼?`mode=selfie_ref`锛変細寮哄埗璧拌嚜鎷嶅弬鑰冪収娴佺▼銆?- `mode=auto` 浠呭湪鈥滄彁绀鸿瘝鏄庣‘鎸囧悜鑷媿 + 宸查厤缃弬鑰冪収鈥濇椂鎵嶄細鑷姩灏濊瘯鑷媿锛涘惁鍒欏洖閫€涓烘枃鐢熷浘/鏀瑰浘銆?
### 瑙嗛鐢熸垚

鍙戦€?寮曠敤鍥剧墖鍚庯細
```
/瑙嗛 [@provider_id] <鎻愮ず璇?
/瑙嗛 [@provider_id] <棰勮鍚? [棰濆鎻愮ず璇峕
/瑙嗛棰勮鍒楄〃
```

---

## 娉ㄦ剰浜嬮」

- 濡傛灉浣犳病鏈夐厤缃?providers 鎴栭摼璺负绌猴細鎻掍欢浼氭彁绀轰綘鍘?WebUI 琛ラ厤缃€?- 缃戝叧鏄惁鏀寔鏌愪釜鎺ュ彛锛堝挨鍏舵槸 images.edit锛夊彇鍐充簬鏈嶅姟鍟嗗疄鐜版湰韬紱鎻掍欢浼氳嚜鍔ㄥ厹搴曞埌鍚庣画 provider銆?- `@provider_id` 浠呮槸鈥滀复鏃舵寚瀹氫竴娆′娇鐢ㄥ摢涓?provider鈥濓紝涓嶄細鏀瑰彉浣犵殑榛樿閾捐矾椤哄簭銆?- Gitee 鏂囩敓鍥句粎鏀寔鐧藉悕鍗曞昂瀵革紱鑻ヨ緭鍑哄昂瀵镐笉鍚堟硶浼氳嚜鍔ㄥ厹搴曞埌鍙敤灏哄骞惰褰曟棩蹇椼€?
---

## Gitee AI API Key 鑾峰彇鏂规硶锛堜繚鐣欏師鏂囷級

1.璁块棶<https://ai.gitee.com/serverless-api?model=z-image-turbo>

2.<img width="2241" height="1280" alt="PixPin_2025-12-05_16-56-27" src="https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841" />

3.鍏嶈垂棰濆害<img width="240" height="63" alt="PixPin_2025-12-05_16-56-49" src="https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb" />

4.鍙互娑╂订锛岃鎯曡繚瑙勮涓炬姤

5.濂界敤鍙互缁欎釜馃専

---

## 鏀寔鐨勫浘鍍忓昂瀵革紙Gitee锛屼繚鐣欏師鏂囷級

> 鈿狅笍 **娉ㄦ剰**: 浠呮敮鎸佷互涓嬪昂瀵革紝浣跨敤鍏朵粬灏哄浼氭姤閿?
| 姣斾緥 | 鍙敤灏哄 |
|------|----------|
| 1:1 | 256脳256, 512脳512, 1024脳1024, 2048脳2048 |
| 4:3 | 1152脳896, 2048脳1536 |
| 3:4 | 768脳1024, 1536脳2048 |
| 3:2 | 2048脳1360 |
| 2:3 | 1360脳2048 |
| 16:9 | 1024脳576, 2048脳1152 |
| 9:16 | 576脳1024, 1152脳2048 |

---

## 鍑哄浘灞曠ず鍖猴紙淇濈暀鍘熸枃锛?
<img width="1152" height="2048" alt="29889b7b184984fac81c33574233a3a9_720" src="https://github.com/user-attachments/assets/c2390320-6d55-4db4-b3ad-0dde7b447c87" />

<img width="1152" height="2048" alt="60393b1ea20d432822c21a61ba48d946" src="https://github.com/user-attachments/assets/3d8195e5-5d89-4a12-806e-8a81e348a96c" />

<img width="1152" height="2048" alt="3e5ee8d438fa797730127e57b9720454_720" src="https://github.com/user-attachments/assets/c270ae7f-25f6-4d96-bbed-0299c9e61877" />

鏈彃浠跺紑鍙慟Q缇わ細215532038

<img width="1284" height="2289" alt="qrcode_1767584668806" src="https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee" />


