# 检索系统与 Skill Runtime 会话交接（2026-07-26）

## 1. 当前结论

本任务尚未完成，不能进入最终验收。

当前分支：`codex/retrieval-skill-runtime-v1`

已完成的主要实现：

- Skill Runtime 编译、版本、哈希、防篡改和结构化场景策略；
- 40 份文档、380 个 chunks 的摄取、manifest、metadata 和覆盖率 Gate；
- PostgreSQL + pgvector 七表、版本发布/回滚、FTS、向量检索和 Trace；
- 生产场景分析、检索计划、混合检索、CrossEncoder、Evidence Gate 和生成链；
- 120 条 Golden Dataset、三个逐条人工 review artifact 和离线指标；
- Golden review 的 `07.4 -> 07.5` 正文级稳定映射；
- `recommended_action` 结构化输出、评测和 Retrieval Trace 持久化；
- 旧检索路径删除和部署/迁移脚本。

## 2. 本会话完成的关键修复

1. 真实 PostgreSQL/pgvector 集成发现并修复 UUID/string 边界问题。
2. 真实 LLM 生成未知 hard filter 时，在 agent 信任边界执行白名单归一化，底层数据库继续 fail-closed。
3. 移除 Golden label 注入和 selector priority 调参；live runner 只把 query/case ID 送入生产节点。
4. 对 120 条 Golden 逐条阅读正文重标：
   - review 目录下分别保存 63、33、24 条记录；
   - 不再共用一个通用 forbidden chunk；
   - forbidden 为空时不进入 rejection rate 分母；
   - relevant evidence 不得命中 excluded topics。
5. 修复安全正文因出现“操控/强迫”等警告词而被误标为 `manipulation`：
   - manipulation 只根据标题/heading 判定；
   - PUA 风险文档 7/7 chunks 仍保留 manipulation topic。
6. 发布真实 corpus `2026.07.5`：
   - 40 documents；
   - 380 chunks；
   - 380 embeddings；
   - `BAAI/bge-small-zh-v1.5`，512 维，L2 normalization。
7. 将场景推荐方向作为 `recommended_action` 必填枚举写入 Scene Snapshot、live Gate 和 Trace。
8. live runner 增加逐 case start/done 日志和原子 partial prediction checkpoint。

## 3. CR 与验证证据

已通过的独立 CR：

- 真实 PostgreSQL repository、Trace、发布/回滚 CR：PASS；
- live runner 无标签泄漏 CR：PASS；
- Golden 逐条内容审阅：初次 BLOCK，修复后 PASS；
- metadata 与 `07.4 -> 07.5` review 映射 CR：PASS；
- `recommended_action` 与 Trace CR：初次发现缺 Trace 的 P1，修复后 PASS。

本会话最后一轮相关测试：

```text
tests.test_retrieval_service
tests.test_postgres_repository
tests.test_agent_integration
tests.test_evaluation

60 tests OK（连接真实 TEST_DATABASE_URL）
```

随后重新发布 `2026.07.5`，输出：

```text
published_corpus=2026.07.5 chunks=380
```

其他已记录验证：

- Golden/metadata 定向测试：28 tests OK；
- action 相关定向测试：44 tests OK；
- 独立 action CR：67 total，62 pass，5 个 DB tests 在 reviewer 环境跳过；主线已单独用真实 DB 跑过。

## 4. 正在运行的 Gate 与暂停点

真实 Gate 使用：

- PostgreSQL/pgvector：`127.0.0.1:55432`；
- corpus：`2026.07.5`；
- 真实场景模型；
- BGE embedding；
- CrossEncoder reranker；
- 逐条人工重标 Golden；
- production `build_context -> analyze_scene_and_plan -> retrieve_evidence`。

按用户要求暂停时，runner 已被明确停止：

```text
case_done=68/120 id=GD-08-05
PID 37000 stopped at 2026-07-26 01:31:44 +08:00
partial checkpoint bytes=158597
```

日志：

```text
C:\Users\26561\AppData\Local\Temp\crushpilot-live-gate-075b-20260726-010017.out.log
C:\Users\26561\AppData\Local\Temp\crushpilot-live-gate-075b-20260726-010017.err.log
```

partial checkpoint：

```text
trellis/retrieval-system-refactor/live-gate-predictions.partial.json
```

旧的完整结果 `live-gate-results.json` 是修复前 Golden 的失败基线，不是当前发布证据。保留的明确基线：

```text
trellis/retrieval-system-refactor/live-gate-baseline-2026.07.4-pre-review.json
```

## 5. 未完成事项

- 当前 `07.5` 真实 120 Gate 尚未完整结束；
- 当前 runner 不支持从 partial checkpoint 恢复，暂停后需从头执行；
- Gate 指标不通过时尚未进行按类别误差分析和实现修复；
- `gate-report.md`、最终 traceability/status 尚未更新到最终真实结果；
- 最后一次代码变更后的全量静态测试已运行，但完整 runtime Gate 与最终验收测试仍未完成；
- 用户要求的最终审核尚未开始；最终必须新开独立 subagent，先读技术设计全文，再逐项对照代码和运行证据；
- 最终审核发现问题后必须继续 CR/fix/loop，不能直接结束。

## 6. 工作区注意事项

- 不要运行 `git reset --hard` 或覆盖当前工作区；
- integration tests 会清空指定测试数据库；若再次对 55432 运行，需要随后重新发布 corpus；
- `.env` 含真实模型配置，禁止写入日志、文档或 Git；
- `live-gate-predictions.json` 与 `live-gate-results.json` 当前仍是旧基线；只有新 runner 完整结束才会覆盖；
- 不得重新引入 Golden label 到 runtime state，也不得按 Golden selector 给 corpus priority。

## 7. 暂停前最后验证

最后一次全量静态回归：

```text
Ran 119 tests in 19.021s
OK (skipped=5)
```

5 个 skip 均为未给该进程设置 `TEST_DATABASE_URL` 的真实 PostgreSQL tests；同一轮代码此前已用 `127.0.0.1:55432` 跑过 60 项组合测试并通过。暂停时 `07.5` 已重新发布，未在重发布后再次运行会清库的 integration tests。
