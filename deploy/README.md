# Linux 启动

已知服务器的 VS Code 手动部署步骤见 [服务器手动部署指引](服务器手动部署指引.md)。

在 Linux 主机安装 Docker Compose 后：

```bash
cd deploy
cp .env.example .env
# 编辑 .env，填写 PostgreSQL 密码和模型配置
docker compose up -d --build
curl --fail http://localhost:8080/ready
```

`corpus-publisher` 会在后端启动前扫描全部 required Markdown、构建统一
Corpus、生成 FTS/Embedding 并原子发布版本。任一步失败时后端不会进入
ready，不能回退到旧 JSON 索引或硬编码路由。知识源或模型版本变更后，
更新 `CORPUS_VERSION` 并再次执行 `docker compose up -d --build`。

浏览器访问 `http://服务器地址:8080`。生产环境的 TLS 终止可由服务器既有反向代理处理；其代理必须关闭 `/api/v1/chat` 的响应缓冲，确保 SSE 能立即发送。
