ARG UV_IMAGE
ARG PYTHON_IMAGE
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE}
ARG UV_IMAGE
ARG PYTHON_IMAGE
RUN case "$UV_IMAGE" in *@sha256:*) ;; *) exit 1;; esac \
    && case "$PYTHON_IMAGE" in *@sha256:*) ;; *) exit 1;; esac
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /srv
COPY apps/learning_worker/pyproject.toml apps/learning_worker/uv.lock /srv/apps/learning_worker/
RUN uv sync --project /srv/apps/learning_worker --locked --no-dev
COPY apps/api /srv/apps/api
COPY apps/learning_worker /srv/apps/learning_worker
COPY learning /srv/learning
COPY contracts /srv/contracts
COPY simulation /srv/simulation
ENV PATH="/srv/apps/learning_worker/.venv/bin:$PATH" \
    PYTHONPATH="/srv" \
    PYTHONDONTWRITEBYTECODE="1"
RUN groupadd --gid 10001 worker && useradd --uid 10001 --gid 10001 --create-home worker
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn", "apps.learning_worker.main:create_worker", "--factory", "--host", "0.0.0.0", "--port", "8080"]
