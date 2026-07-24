# Linux 启动

已知服务器的 VS Code 手动部署步骤见 [服务器手动部署指引](服务器手动部署指引.md)。

在 Linux 主机安装 Docker Compose 后：

```bash
cd deploy
cp .env.example .env
# 编辑 .env，填写 PostgreSQL 密码和模型配置
docker compose up -d --build
curl http://localhost:8080/health
# 首次部署或知识更新后：安全派生全部原始资料并建立本地向量索引
docker compose exec backend python -m app.knowledge_base build-cards
docker compose exec backend python -m app.knowledge_base build-index
```

浏览器访问 `http://服务器地址:8080`。生产环境的 TLS 终止可由服务器既有反向代理处理；其代理必须关闭 `/api/v1/chat` 的响应缓冲，确保 SSE 能立即发送。
