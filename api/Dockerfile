FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# instala deps
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# copia todo o projeto (api, agents, docs, etc.)
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host","0.0.0.0","--port","8000"]
