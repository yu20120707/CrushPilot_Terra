# Phase 0 Code Review

## Scope

Reviewed `backend/evaluation/**`, `backend/tests/test_evaluation.py` and the Phase 0 Trellis files against `SPEC-EVAL-001..003`, `SPEC-P0-001` and `SPEC-TR-001`.

## Findings and loop

1. Fixed: the first metric test incorrectly expected a no-evidence case to add reciprocal-rank credit.
2. Fixed: generic placeholder queries were replaced by category-specific, de-identified scenarios; Hard Negative rows now alternate the same keyword between continue and stop strategies.
3. Fixed: Reranker Win Rate previously returned `0.0` when no baseline pairs existed; it now returns `null` so absence of evaluation is not reported as a measured loss.

## Follow-up external CR finding

后续独立 CR 发现并完成两轮修复：

1. 已将模板扩展替换为 120 条语义唯一场景，并把 evidence 映射为真实 Corpus UUID；存在性、集合互斥和 Hard Negative 测试通过。
2. 已修复空预测可让 No-Evidence Accuracy 虚假过线的问题；该指标只在明确 no-evidence 标注集上计算，要求显式 `no_evidence=true` 且检索结果为空。
3. 已让 Context-Dependent Query Accuracy 对每组 Hard Negative 成对检查 `action_direction` 和对应 gold evidence；固定 task type 或固定方向得不到满分。

定向 8 tests 通过。`11-evaluation` 实现可关闭；真实检索阈值和人工 evidence 相关性抽样仍属于发布 Gate，不作通过声明。
