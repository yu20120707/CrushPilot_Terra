# 检索系统与 Skill Runtime 后续执行规划

## 目标与约束

继续完成技术设计 v1.0 的全部要求。执行级别保持 Level 3；每一代码阶段必须
test → 独立 CR → fix loop，最终必须由新 subagent 完整读取设计文档并对照实现审核。

## Phase A：恢复模型验证

1. 从 `deploy/.env` 注入模型环境，但不得打印任何值。
2. 运行最小 JSON Schema smoke；若仍为 HTTP 402，记录时间与错误并继续处理
   不依赖模型的任务，不得用 demo 结果替代。
3. 使用正式 runner：

   ```powershell
   Set-Location C:\Users\26561\Desktop\CrushPilot_Terra\backend
   $env:PYTHONPATH='.'
   py -m evaluation.run_scene_plan_gate `
     --database-url 'postgresql://postgres:postgres@127.0.0.1:55432/postgres' `
     --corpus-version '2026.07.6'
   ```

4. 额度恢复后完整运行 120 条 scene-only planning eval：
   - 必须 120/120；
   - 比较 production baseline `0.65556`、search glossary `0.70000` 和当前版本；
   - 检查 `digital_context`、`reciprocity`、模糊拒绝、边界、多意图回归。
5. 若当前 planning meanings 无净改善，回退该增量或继续通用 ontology 修复并重新 CR。

## Phase B：真实 Golden Gate 收敛

从 `backend` 运行：

```powershell
$env:PYTHONPATH='.'
py -m evaluation.run_live_gate `
  --database-url 'postgresql://postgres:postgres@127.0.0.1:55432/postgres' `
  --corpus-version '2026.07.6'
```

runner 在发布 Gate 未通过时退出 1，但应保留完整 results/predictions。必须确认：

- 120 个 `case_done`；
- `retrieval_trace_count == 120`；
- `retrieval_trace_completeness == 1.0`；
- partial 被清理，predictions 先于 results 原子发布；
- action-direction accuracy 与 stage latency P50/P95 已写入结果。

原阈值不可修改：

- Recall@20 ≥ 0.90；
- nDCG@10 ≥ 0.75；
- Boundary Key Knowledge Recall = 1.00；
- Forbidden Chunk Rejection ≥ 0.95；
- No-Evidence Accuracy ≥ 0.90；
- Source Coverage = 1.00；
- Trace Completeness = 1.00。

失败时按每个 case 的 lexical、vector、fused、reranked、selected 五层定位。优先处理：

1. Scene Planner required topics 缺失；
2. gold 已在 fused top20 却被 reranker 降序；
3. boundary/forbidden 冲突；
4. multi-intent 证据覆盖；
5. Evidence Gate 预算或冲突过滤。

禁止 Golden 标签注入、case/category 路由、selector priority 和降阈值。

## Phase C：迁移与治理收敛

1. 请架构所有者批准或拒绝
   [`ADR-013`](../adr/ADR-013-检索迁移影子评测偏差.md)：
   - 批准：使用隔离 Golden 回放替代不可追溯的在线 old/new shadow；
   - 拒绝：需要架构所有者指定如何补足原 Phase 5 历史过程证据。
2. Gate 全绿后生成 replay report，包含召回、排序、task/action、stage latency；
   旧链只作定性历史参考，不伪造 old/new latency。
3. 验证 `legacy_retrieval_paths=0`、Source/Version/Traceability Gate。
4. 更新 Phase 12、Phase 13 status；只有所有 remaining 清零才能 completed。

## Phase D：最终验证与独立审核

1. 使用隔离 `crushpilot_test` 运行完整 152 tests（新增测试后以最新总数为准），
   必须 0 skipped。
2. 运行 source/version/traceability/legacy 四个治理 Gate。
3. 新开未参与实现的 subagent：
   - 完整读取技术设计全文；
   - 将每个 MUST、表、ADR、Gate、迁移步骤和验收条件转成 checklist；
   - 对照代码、SQL、脚本、测试、真实 Gate、Trace、部署和文档；
   - 每项提供路径/行号/运行证据；
   - 输出 PASS/BLOCK。
4. 任何 P0/P1 或缺失证据均进入 fix → test → CR → final audit loop。
5. 仅最终审核 PASS、Gate 全绿、无未验证必需项时才能提交 goal complete。

## 最终 Git 交付

- 当前安全分支：`codex/retrieval-skill-runtime-v1-resume`。
- 禁止推送旧本地分支的历史密钥。
- `git add` 后、commit 前运行 `git diff --check`、完整测试和
  `py scripts/scan_staged_secrets.py`；scanner 检查 index 中包含新增文件的确切快照。
- 用户必须确认先前暴露的 DeepSeek credential 已撤销/轮换；本地更新 ignored
  `deploy/.env`，不得提交。
- 从安全分支正常推送到远端 `codex/retrieval-skill-runtime-v1`，不得 force push。
