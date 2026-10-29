FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

WORKDIR /work

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv==0.11.14 && uv sync --frozen --extra gpu --no-install-project

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

CMD ["pytest"]
