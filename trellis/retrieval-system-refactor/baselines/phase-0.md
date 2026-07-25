# Phase 0 Baseline

记录时间：2026-07-25（Asia/Shanghai）

## 当前旧链行为

- `analyze_intent` 产生关键词；无结果时用固定别名扩展一次。
- 在线检索合并安全卡关键词结果与本地研究卡向量结果，按 source 去重后最多返回 4 条。
- 关键词评分使用 Python `str.count()`；最终 Prompt 同时包含安全卡 evidence 和按规则读取的完整参考文档片段。
- 向量索引不存在时静默退回关键词检索；无结果在第二次尝试后继续生成。
- 当前行为是迁移前观测对象，不表示满足目标设计。

## 可复现测试基线

命令：

```powershell
$env:PYTHONPATH='backend'
py -m unittest backend.tests.test_main backend.tests.test_research -v
```

结果：28 tests，全部通过，耗时约 0.7 秒。该命令刻意限定迁移前已有的两个测试模块；并行新增测试不计入这项冻结数字。

## 延迟与命中

- 离线测试套件耗时约 0.7 秒。
- 旧链固定返回上限：4 个 chunk。
- 代表性已有断言：`工资/经济条件 -> card:income-demeaning`；`累 -> card:low-pressure-chat`。
- 未调用真实在线模型或生产数据库，故不记录伪造的端到端 P50/P95/P99。
