# Gate Report

状态：**BLOCKED FOR RELEASE**。静态、数据库和治理验证通过；真实 Golden 120
发布 Gate 仍有四项未满足，`12-migration-cutover` 与 `13-final-convergence` 保持
未完成。

## 最新完整真实 Gate

运行环境：Corpus `2026.07.6`、PostgreSQL/pgvector、BGE small zh、CrossEncoder
reranker、真实 Scene Planner，120/120 条完成，Trace 120/120。

| 指标 | 当前值 | 阈值 | 结果 |
|---|---:|---:|---|
| Gold Evidence Recall@20 | 0.63095 | ≥ 0.90 | BLOCK |
| nDCG@10 | 0.19043 | ≥ 0.75 | BLOCK |
| Boundary Key Knowledge Recall | 0.47059 | 1.00 | BLOCK |
| Forbidden Chunk Rejection | 0.92308 | ≥ 0.95 | BLOCK |
| No-Evidence Accuracy | 1.00000 | ≥ 0.90 | PASS |
| Source Coverage | 1.00000 | 1.00 | PASS |
| Retrieval Trace Completeness | 1.00000 | 1.00 | PASS |

该次 Gate 已包含 required-topic 的真实 SQL 加权，但早于最新 Scene Planner planning
description 改动。后者尚未形成新的完整 Gate 证据：scene-only 评估在 69/120 时被模型
服务返回 HTTP 402 中断。

正式 `run_scene_plan_gate` 于 2026-07-27 00:30:21 +08:00 再次重试，在第一个
case 得到 HTTP 402；见
[`scene-plan-gate-failure-2026-07-26.md`](scene-plan-gate-failure-2026-07-26.md)。

## 已验证

- Python 全套：152/152 PASS，使用隔离数据库 `crushpilot_test`，0 skipped。
- PostgreSQL/pgvector integration：schema、跨版本约束、staging/publish/rollback、
  exact vector、Trace round-trip 全部通过。
- Source Coverage：`unprocessed_sources=0`。
- Corpus Version：`corpus_version_mismatches=0`。
- Legacy Retrieval：`legacy_retrieval_paths=0`。
- Corpus `2026.07.6`：40 documents、380 chunks、380 embeddings。
- Gate runner：无 Golden label 注入、原子发布结果、缺 Trace fail-closed，阶段 CR PASS。
- Required-topic lexical：独立 user/topic tsquery、topic 1.5× 排名权重、否定词噪声
  隔离，真实 PostgreSQL smoke 与阶段 CR PASS。
- Scene topic ontology：共享 search terms + planning meanings，无 Golden selector/category
  泄漏，阶段 CR PASS。
- Gate metrics：action direction accuracy 与新链 stage latency P50/P95 已实现并通过 CR；
  需下一次完整 Gate 生成最终 artifact。
- 部署默认 Corpus：backend 与 publisher 均绑定 `CURRENT_CORPUS_VERSION=2026.07.6`。

## 未完成与阻塞

1. DeepSeek 当前返回 `402 Payment Required`，无法完成最新代码的 120 Scene/Gate 回放。
2. Recall、nDCG、Boundary 和 Forbidden 四个发布 Gate 未通过。
3. ADR-013 仍为 Proposed，需要架构所有者明确批准或拒绝。
4. Phase 12 回放报告和 Phase 13 最终收敛尚未完成。
5. 最终独立全文对照审核尚未开始；必须在所有 Gate 证据齐全后由新 subagent 执行。
