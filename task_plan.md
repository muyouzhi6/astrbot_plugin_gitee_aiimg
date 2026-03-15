# Task Plan: Fix gitee_aiimg plugin regressions

## Goal
修复插件中的视频生成误报、重试次数显示异常、私聊无可见回复问题，并完成一轮深度代码审查与风险清点。

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define technical approach
- [x] Identify affected code paths
- [x] Document decisions with rationale
- **Status:** complete

### Phase 3: Implementation
- [x] Fix false error reporting path
- [x] Fix retry attempt counting
- [x] Fix private chat reply path
- [x] Address additional hidden issues
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Verify all reported issues
- [x] Document test results in progress.md
- [x] Fix issues found during validation
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize code review findings
- [x] Review changed files
- [ ] Deliver to user
- **Status:** in_progress

## Key Questions
1. 哪个 provider 分支先记录失败，再继续 fallback，导致“可生成但先报错”？
2. attempt 计数是日志格式问题，还是重试状态在每次调用中被重置？
3. 私聊回复是消息发送入口漏调，还是会话类型判断异常？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 先检查日志涉及的 provider 调度链路 | 三个显性问题都可能集中在主调度入口 |
| 重点核查 `_send_image_with_fallback()` 与 `_send_video_result()` | 私聊可见性问题最可能在发送兼容逻辑 |
| 对中间 fallback 失败降级为信息日志 | 避免“前一个失败但后一个成功”制造误报感 |
| 把重试日志拆成总次数 + parse/request 子计数 | 让日志能反映真实重试上下文 |
| 修复插件级静态检查暴露的隐藏 bug | 包括 `Path` 漏导入、Grok2API 视频 key 轮换未生效、损坏的图片后端语法问题 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `grok2api_images_backend.py` after quick edit failed syntax check | 1 | Repaired broken docstring/strings and restored merged image assignment |

## Notes
- Validation completed with `ruff check` and `python -m compileall -q`
- Final delivery should include changed files and remaining low-risk observations
