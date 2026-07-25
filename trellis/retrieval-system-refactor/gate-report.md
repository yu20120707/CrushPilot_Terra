# Gate Report

状态：代码实现与静态 Gate 已收敛；真实发布 Gate 未满足，`13-final-convergence`
保持 `pending`，不得宣称已完成全部要求。

## 已验证

- Python 全套：106 total，102 passed，4 skipped；跳过项全部是缺少
  `TEST_DATABASE_URL` 的真实 PostgreSQL + pgvector 集成测试。
- Golden Dataset：120 条语义唯一场景 / 14 类 / 七类字段；证据为当前
  380-chunk Corpus 的真实 UUID，存在性与三集合互斥测试通过。
- Hard Negative：4 对相同关键词、相反动作和不同 gold evidence，测试通过。
- Source Coverage：`unprocessed_sources=0`。
- Corpus Version：`corpus_version_mismatches=0`。
- Legacy Retrieval：`legacy_retrieval_paths=0`。
- Traceability：遗漏需求 0；未验证任务 1（`13-final-convergence`）。
- `compileall` 与 `git diff --check` 通过。
- 本地真实 `BAAI/bge-small-zh-v1.5` 加载及 512 维有限向量 smoke 通过。
- 已配置模型的 JSON Schema 调用 smoke 通过。

## 尚未验证

- 真实 PostgreSQL + pgvector 的 schema、跨版本约束、staging/publish/rollback、
  exact vector round-trip；需要提供 `TEST_DATABASE_URL` 后重跑 4 项集成测试。
- 真实 embedding + reranker + generation model 的 120 场景完整预测，以及第
  25.3 节各项发布阈值。
- Retrieval Trace 在真实 PostgreSQL 中的 100% 完整率。
- Golden Dataset evidence 逐条相关性的人工抽样。
- 构建报告显示 `not_applicable_when` 与 `action_labels` 存在较高缺失率；字段允许
  缺失，但发布前需要人工质量抽样结论。

因此最终 Convergence 五项计数尚不能全部计算为 0，当前没有发布通过声明。
