# Linux 启动

在 Linux 主机安装 Docker Compose 后：

```bash
cd deploy
cp .env.example .env
# 编辑 .env，填写 PostgreSQL 密码和模型配置
docker compose up -d --build
curl http://localhost:8080/health
```

浏览器访问 `http://服务器地址:8080`。生产环境的 TLS 终止可由服务器既有反向代理处理；其代理必须关闭 `/api/v1/chat` 的响应缓冲，确保 SSE 能立即发送。
