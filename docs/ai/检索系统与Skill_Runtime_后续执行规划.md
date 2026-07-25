# 检索系统与 Skill Runtime 后续执行规划

## 目标

继续完成 `CrushPilot_检索系统与Skill_Runtime_技术设计_v1.0.md` 的全部要求，并经过阶段 CR、真实 Gate、最终独立文档对照审核后再宣告完成。

执行级别保持 Level 3。不得因为当前代码量大或已有部分绿测而降级验收标准。

## Phase A：恢复并完成真实 120 Gate

### A1. 环境检查

```powershell
Set-Location C:\Users\26561\Desktop\CrushPilot_Terra

py -c "import psycopg; c=psycopg.connect('postgresql://postgres:postgres@127.0.0.1:55432/postgres'); print(c.execute('select version,status from knowledge_corpus_versions').fetchall()); print(c.execute('select corpus_version,count(1) from knowledge_chunks group by corpus_version').fetchall())"
```

期望：

```text
2026.07.5 published
2026.07.5 380 chunks
2026.07.5 380 embeddings
```

若数据库被测试清空，重新发布：

```powershell
$env:PYTHONPATH='backend'
py scripts/publish_corpus.py `
  --database-url 'postgresql://postgres:postgres@127.0.0.1:55432/postgres' `
  --version '2026.07.5'
```

运行前从 `deploy/.env` 注入当前进程环境，但不要打印或提交其中的值。

### A2. 重新运行

当前 runner 的 partial 仅用于诊断，不能恢复，因此暂停后从头运行：

```powershell
Set-Location C:\Users\26561\Desktop\CrushPilot_Terra\backend
$env:PYTHONPATH='.'
py -m evaluation.run_live_gate `
  --database-url 'postgresql://postgres:postgres@127.0.0.1:55432/postgres' `
  --corpus-version '2026.07.5'
```

必须确认：

- 120 个 `case_done`；
- `retrieval_trace_count == 120`；
- `retrieval_trace_completeness == 1.0`；
- `forbidden_annotated_case_count == 13`；
- 结果读取的是当前 `golden_dataset_v1.jsonl`。

### A3. Gate 阈值

必须满足设计文档原阈值，不得为通过而修改：

- Gold Evidence Recall@20 ≥ 0.90；
- nDCG@10 ≥ 0.75；
- 明确拒绝/边界关键知识召回 = 100%；
- Forbidden Chunk Rejection Rate ≥ 0.95；
- No-Evidence Accuracy ≥ 0.90；
- Source Coverage = 100%；
- Retrieval Trace 完整率 = 100%。

## Phase B：失败诊断与修复循环

若 Gate 未通过：

1. 按 category 和 case ID 输出失败表；
2. 分别检查 scene plan、lexical top20、vector top20、RRF、rerank、Evidence Gate；
3. 判断是 query planning、metadata、召回、排序、证据预算还是标签问题；
4. 禁止把 expected/gold/required labels注入 runtime；
5. 禁止按 Golden chunk ID、selector 或 category 增加 priority；
6. 只做能推广到未知输入的实现修复；
7. 每一轮修复后运行定向测试并做独立 CR；
8. 重新完整运行 120 Gate，保留历史结果。

推荐先检查：

- Hard Negative 的 `recommended_action` 预测；
- no-evidence case 是否被模型错误生成可检索 topics；
- multi-intent required topics 是否都进入 plan；
- relevant chunk 是否进入 fused top20 但被 reranker 降序；
- CrossEncoder 中文 query/scene 拼接是否与模型预期一致。

## Phase C：最终验证

Gate 通过后运行：

```powershell
Set-Location C:\Users\26561\Desktop\CrushPilot_Terra\backend
$env:PYTHONPATH='.'
py -m unittest discover -s tests -p 'test_*.py'
```

数据库 integration tests 使用单独临时实例或在测试后重新发布 `07.5`，避免破坏最终 runtime 证据。

随后更新：

- `trellis/retrieval-system-refactor/gate-report.md`；
- `trellis/retrieval-system-refactor/traceability.yaml`；
- `trellis/retrieval-system-refactor/tasks/13-final-convergence/status.yaml`；
- 必要的迁移/运行说明。

## Phase D：最终独立审核

必须新开独立 subagent，任务要求写明：

1. 完整读取 `CrushPilot_检索系统与Skill_Runtime_技术设计_v1.0.md`；
2. 将设计中的每个 MUST、表、Gate、迁移步骤和验收条件转成 checklist；
3. 对照当前代码、SQL migration、脚本、测试、真实 Gate、Trace 和文档；
4. 每一项给出文件路径、行号或运行证据；
5. 输出 PASS/BLOCK，不允许仅凭测试绿灯推断实现完整；
6. 发现 P0/P1 后主 agent 必须 fix -> test -> CR -> audit loop；
7. 只有最终审核 PASS 且无未验证必需项，才能将 active goal 标记 complete。

## 暂停恢复时的第一组命令

```powershell
Set-Location C:\Users\26561\Desktop\CrushPilot_Terra
git status --short
git branch --show-current
Get-Content docs/ai/检索系统与Skill_Runtime_会话交接_2026-07-26.md
Get-Content docs/ai/检索系统与Skill_Runtime_后续执行规划.md
```

然后检查 55432 数据库和最新 Gate 日志，再决定重新发布还是直接重跑。
