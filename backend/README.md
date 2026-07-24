# CrushPilot Backend

CrushPilot 后端采用 FastAPI + LangGraph。Python 项目根是本目录，源码使用
`src` 布局，HTTP 层不直接访问模型、数据库或图节点。

## 固定调用链

```text
HTTP Request
  -> FastAPI endpoint
  -> ChatService
  -> Compiled LangGraph
  -> Nodes / Tools / Model / Database
```

主要职责：

- `src/app/api`：HTTP 路由、请求依赖和异常映射。
- `src/app/services`：应用用例；聊天与会话操作统一从 `ChatService` 进入。
- `src/app/agents/assistant`：LangGraph 状态、节点、边、提示词、工具和输出模型。
- `src/app/infrastructure`：数据库、checkpointer 和模型供应商适配。
- `src/app/core`：配置、日志与应用生命周期。
- `src/app/schemas`：HTTP 请求/响应模型。

应用启动时由 lifespan 创建数据库连接、LangGraph checkpointer 和已编译图，
再注入 `ChatService`；关闭时统一释放连接。`langgraph.json` 使用无持久化的
图入口，供 LangGraph CLI/Studio 自行管理运行时。

## 本地运行

需要 Python 3.12 或 3.13，以及 [uv](https://docs.astral.sh/uv/)。

```powershell
cd backend
Copy-Item .env.example .env
uv sync
uv run uvicorn app.main:app --app-dir src --env-file .env --reload --port 8001
```

`.env` 默认建议保留 `DEMO_MODE=true`。接入真实模型时填写对应供应商变量并将
`DEMO_MODE=false`。不要提交真实 API Key。

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

## 测试与检查

```powershell
cd backend
uv run pytest
uv run ruff check src tests
python -m compileall -q src
```

测试按 `unit/agents`、`unit/services` 与 `integration/api` 分层，集成测试通过
FastAPI `TestClient` 验证完整的 endpoint → service → graph 链路。

## PyCharm

推荐直接将 `backend/` 作为项目打开，并把 `backend/.venv` 设为 Python 3.12+
解释器。若从仓库根目录打开，需将 `backend/src` 标记为 **Sources Root**，并将
`backend/tests` 标记为 **Test Sources Root**；代码中的导入应始终使用
`from app...`，不使用 `from src.app...`。

## LangGraph

```powershell
cd backend
uv run langgraph dev
```

图构造器位于 `src/app/agents/assistant/graph.py`。FastAPI 会传入持久化
checkpointer；LangGraph CLI 使用 `langgraph.json` 中导出的 `graph`。

## 离线知识维护

原始研究材料不会进入在线聊天检索。以下命令只处理本地知识卡或离线报告：

```powershell
uv run crushpilot-knowledge build-index
uv run crushpilot-research --estimate --output ../data-local/research-report.json
uv run crushpilot-research --output ../data-local/research-report.json
```
