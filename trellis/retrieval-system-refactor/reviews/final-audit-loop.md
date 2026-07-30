# Final Audit Loop

## Round 1

独立审计完整重读技术设计并对照代码。结果未放行，发现：

1. Retrieval Trace producer 与 PostgreSQL sanitizer 字段错配。
2. FastAPI 并发请求共享 psycopg connection，事务边界可能交叉。
3. Context Token Budget 未覆盖摘要、事实、偏好，当前消息重复注入。
4. 真实 PostgreSQL、120 场景发布指标、Trace 完整率及人工相关性抽样未运行。

## Fix

- 对齐 Trace 的 `*_id` / `*_ids` 契约并增加持久化边界回归断言。
- 对共享 graph/checkpoint 数据库路径增加进程内互斥，避免单进程请求事务交叉；
  当前明确接受串行吞吐，不声称已实现连接池。
- Context 对消息、事实、偏好、摘要统一计入预算；删除 recent message 与 Prompt
  中的当前消息重复；第二轮审计发现重复文本历史边界后，修正为只删除最后一个
  匹配实例并增加回归测试。

## Verification

- 全套：106 total，102 passed，4 skipped。
- Source / Corpus / Legacy：0 / 0 / 0。
- Traceability：遗漏需求 0，未验证任务 1。

真实发布 Gate 的外部依赖缺口保持在 `gate-report.md`，未将其改写为通过。
