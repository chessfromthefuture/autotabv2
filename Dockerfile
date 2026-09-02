FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY autotab ./autotab
COPY models ./models
COPY streamlit ./streamlit
RUN uv pip install --system --no-cache ".[app]"

ENV PORT=8501
EXPOSE 8501
CMD ["sh", "-c", "streamlit run streamlit/autotab_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
