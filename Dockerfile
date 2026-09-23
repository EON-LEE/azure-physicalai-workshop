FROM node:24-bookworm-slim AS web
WORKDIR /src
COPY contracts/customer-environment.schema.json contracts/customer-environment.schema.json
COPY apps/web/package*.json apps/web/
RUN cd apps/web && npm ci
COPY apps/web/ apps/web/
COPY examples/inspection-cell.json examples/inspection-cell.json
RUN cd apps/web && npm run build

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.13-slim-bookworm AS api
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev
COPY apps/__init__.py apps/__init__.py
COPY apps/api/ apps/api/
COPY contracts/ contracts/
COPY agents/ agents/
COPY scripts/__init__.py scripts/bootstrap.py scripts/run_reference_demo.py scripts/
COPY examples/ examples/
COPY --from=web /src/apps/web/dist apps/web/dist
RUN useradd --uid 10001 --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "apps.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
