# Scene Plan Gate Failure Record

- 时间：2026-07-26 23:31:17 +08:00
- Corpus：`2026.07.6`
- Runner：`python -m evaluation.run_scene_plan_gate`
- 完成 case：0/120
- Final results：未发布
- Final predictions：未发布
- Partial predictions：未生成（第一个 case 在模型调用时失败）
- 错误：DeepSeek `/chat/completions` 返回 HTTP 402 Payment Required

## 2026-07-27 重试

- 时间：2026-07-27 00:30:21 +08:00
- 完成 case：0/120
- 结果：第一个 case 再次返回 HTTP 402，未发布 final/partial artifact
- 运行边界：`MODEL_TRUST_ENV=false`，未使用 Demo

## Rerank/Diversity 修复后重试

- 时间：2026-07-27 00:42:06 +08:00
- 完成 case：0/120
- 结果：第一个 case 仍返回 HTTP 402，未发布 final/partial artifact

## Gold-plan 诊断后重试

- 时间：2026-07-27 01:01:19 +08:00
- 完成 case：0/120
- 结果：第一个 case 仍返回 HTTP 402，未发布 final/partial artifact

环境从本地 ignored `deploy/.env` 注入；本文档不记录 URL 之外的配置值、API key、
request payload 或用户数据。模型额度恢复后必须从头重新运行 120 条。
