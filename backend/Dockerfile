FROM python:3.13-slim
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./
RUN pip install --no-cache-dir uv==0.11.19 \
    && uv sync --frozen --no-dev --no-install-project
COPY backend/src ./src
COPY backend/README.md ./
RUN uv sync --frozen --no-dev
COPY knowledge ./knowledge
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    DATA_DIR="/data" \
    KNOWLEDGE_DIR="/app/knowledge"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
