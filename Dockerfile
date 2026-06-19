FROM python:3.11-slim

WORKDIR /app

# libpq-dev + gcc required to compile asyncpg
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

# Default: run the API service.
# Crawler Jobs override this with:
#   command: ["python", "-m", "crawler.civitai_crawler", "--base-model", "...", "--mode", "..."]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8765"]
