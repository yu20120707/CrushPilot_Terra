# 检索系统与 Skill Runtime 会话交接（2026-07-26）

## 当前状态

- 目标仍为 active，不能宣告完成或发布。
- 当前分支：`codex/retrieval-skill-runtime-v1-resume`，跟踪
  `origin/codex/retrieval-skill-runtime-v1`。
- HEAD 基线：`3812430 feat: implement retrieval and skill runtime foundation`。
- 当前改动尚未 commit/push；不得切回含历史密钥的旧分支。
- 主证据库：`127.0.0.1:55432/postgres`，`2026.07.6` published。
- 隔离测试库：`127.0.0.1:55432/crushpilot_test`。

## 本轮完成

1. 修复 Gate runner 可信度：完整 trace 分母、真实 source coverage、stale artifact 清理、
   原子 predictions/results、失败 Gate 写结果并退出 1；独立 CR PASS。
2. Corpus 升级并发布 `2026.07.6`：40 documents、380 chunks、380 embeddings；
   Golden labels 仅做正文稳定映射，无标签注入；独立 CR PASS。
3. Corpus embedding input 加入 title、heading、共享中文 topic terms；向量 query 使用相同
   topic ontology。BGE base 实测劣于 BGE small，保持 small。
4. 修复缺失上下文与明确拒绝优先级；删除过宽 regex/action map；多轮 CR 最终 PASS。
5. 修复 required-topic 假权重：
   - user query 与 topic expansion 使用两个 PostgreSQL tsquery；
   - topic `ts_rank_cd` 真实 1.5×；
   - topic 侧过滤孤立 `不/没/没有`，自然 query 保留否定；
   - 50 项相关测试、真实 PostgreSQL smoke、独立 CR PASS。
6. Scene Planner Prompt 复用共享 ontology，并增加通用 planning meaning；
   无 Golden category/selector/chunk ID；阶段 CR 经 P2 测试修复后 PASS。
7. Gate 结果新增 action-direction accuracy 与 lexical/vector/rerank latency P50/P95；
   18 项评测测试、独立 CR PASS。
8. Compose backend/publisher 默认 Corpus 同步为 07.6，并新增防漂移测试。
9. Phase 12 从错误的 completed 恢复为 in_progress。ADR-013 提议用隔离 Golden 回放
   替代无法追溯的在线 shadow；状态为 Proposed，等待架构所有者批准。
10. 完整测试 153/153 PASS，隔离 PostgreSQL integration 5/5 PASS，0 skipped。
11. 全文需求矩阵发现并闭合三项假绿：离线 metadata suggestion 严格 Schema
    Validation 与 missing/invalid 报告、`known_facts` 来源约束、`TRACE_DEBUG`
    受控样本脱敏/限长/真实 PostgreSQL 持久化；各阶段独立 CR 均 PASS。
12. 最终 Prompt 已在排序后递归移除内部 score/rank 字段；production-shaped
    adversarial E2E 证明模型不可见内部评分，信息不足场景会提出关键补充问题。
13. 修复 reranker 正常/降级路径在 Diversity 前提前截 Top 6 的问题；现在统一将
    Top 15 交给 Diversity 后再选 Final Top 6，backfill 回归和两轮 CR PASS。

## 最新完整 Gate

最新完整 120 Gate（已包含 lexical 真实加权，早于最新 planning meaning）：

```text
Recall@20                  0.630952  BLOCK (>=0.90)
nDCG@10                    0.190427  BLOCK (>=0.75)
Boundary Recall            0.470588  BLOCK (=1.00)
Forbidden Rejection        0.923077  BLOCK (>=0.95)
No-Evidence Accuracy       1.000000  PASS
Source Coverage            1.000000  PASS
Trace Count/Completeness   120 / 1.0 PASS
```

相较前一轮，Recall 从 0.58929 提升到 0.63095，Boundary 从 0.41176 提升到
0.47059，说明 lexical 修复有效；nDCG 基本不变，Scene planning 与 reranking 仍是主瓶颈。

Trace 分层：

```text
lexical top20   0.4479
vector top20    0.5908
fused top20     0.6310
reranked        0.4062
selected        0.2946
```

## 当前外部阻塞

最新 planning meaning 改动后的 scene-only 120 评估在 69/120 时失败：

```text
HTTP 402 Payment Required
https://api.deepseek.com/chat/completions
```

该轮没有完整 artifact，不能用于证明改善。此前仅 search-terms glossary 的完整
scene-only 评估把 plan topic coverage 从 0.65556 提高到 0.70000，但 reciprocity
退化，因此又补了 planning meanings；最终效果必须在额度恢复后重新完整验证。

正式 runner 于 2026-07-27 00:30:21 +08:00 再次执行，第一个 case 即返回
HTTP 402，因此仍没有发布 final/partial artifact。耐久记录：
[`scene-plan-gate-failure-2026-07-26.md`](../../trellis/retrieval-system-refactor/scene-plan-gate-failure-2026-07-26.md)。

## 必须保留的边界

- 不得修改设计 Gate 阈值来换取通过。
- 不得将 Golden expected/gold/required labels、case category、selector 或 chunk ID
  注入 runtime。
- 不得按 Golden case 添加路由或优先级。
- 不得对主证据库运行会清表的 integration tests；使用 `crushpilot_test`。
- 不得把 fused-vs-reranked 描述为 old-vs-new shadow。
- ADR-013 未获用户/架构 owner 批准前不能生效。
- 不得重新引入 JSON 索引、`str.count()`、完整 Markdown 路由或旧检索回滚。
- `deploy/.env` 只保留本地，禁止输出或提交。
- 先前暴露的 DeepSeek credential 必须由用户确认已经撤销/轮换；若未确认，更新本地
  ignored `deploy/.env` 后才能最终 push。不得在文档中记录 key。

## 恢复后的第一步

1. 检查模型最小 JSON Schema 请求是否仍返回 402。
2. 若额度恢复，重新运行正式 scene-only 120 planning eval；保存完整结果并比较
   topic coverage/category regression。
3. 规划改动有效后重跑完整 Golden 120 Gate。
4. 继续按 scene plan → lexical/vector → RRF → rerank → Evidence Gate 分层修复，
   每轮代码后独立 CR。
5. Gate 全绿后生成 migration replay report，完成 Phase 12/13。
6. 新开独立 subagent，完整读取设计文档逐条对照 code/runtime/artifact，循环到 PASS。

## 验证命令

```powershell
Set-Location C:\Users\26561\Desktop\CrushPilot_Terra\backend
$envFile = '..\deploy\.env'
Get-Content -LiteralPath $envFile | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
    [Environment]::SetEnvironmentVariable(
      $matches[1].Trim(), $matches[2].Trim().Trim('"'), 'Process'
    )
  }
}
$env:PYTHONPATH='.'
$env:MODEL_TRUST_ENV='false'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'

py -m evaluation.run_scene_plan_gate `
  --database-url 'postgresql://postgres:postgres@127.0.0.1:55432/postgres' `
  --corpus-version '2026.07.6'

py -m evaluation.run_live_gate `
  --database-url 'postgresql://postgres:postgres@127.0.0.1:55432/postgres' `
  --corpus-version '2026.07.6'

# Gate 未过阈值时退出码为 1，但完整 results/predictions 仍应原子发布。
$env:TEST_DATABASE_URL='postgresql://postgres:postgres@127.0.0.1:55432/crushpilot_test'
py -m unittest discover -s tests -p 'test_*.py'

Set-Location ..
py scripts/verify_source_coverage.py
py scripts/verify_corpus_version.py
py scripts/verify_traceability.py
py scripts/detect_legacy_retrieval_paths.py
```

完成 `git add` 后扫描 Git index 的确切待提交快照。脚本只输出可疑路径，不输出
匹配内容；明确的 placeholder 和环境变量引用不会误报：

```powershell
Set-Location C:\Users\26561\Desktop\CrushPilot_Terra
py scripts/scan_staged_secrets.py
```
