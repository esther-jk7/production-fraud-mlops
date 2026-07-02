# ========== Stage 1: Builder ==========
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependency for XGBoost on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== Stage 2: Runtime ==========
FROM python:3.11-slim AS runtime

WORKDIR /app

# Need libgomp1 in runtime too (XGBoost dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY models/ ./models/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]