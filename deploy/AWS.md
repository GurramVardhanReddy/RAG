# Deploying to AWS (EC2 + Docker Compose)

The stack is three containers (Qdrant, FastAPI backend, nginx frontend) that
`docker compose` already wires together, so a single EC2 instance is the
simplest deployment. Only port 80 is public: Qdrant and the backend bind
to 127.0.0.1 and are reached through the nginx proxy.

## 1. Launch the instance

- **AMI:** Ubuntu Server 24.04 LTS (x86_64)
- **Type:** 16 GB RAM or more (e.g. `t3.xlarge`). The reranker
  (`bge-reranker-v2-m3`) plus one or two domain embedding models need
  several GB of memory on CPU; 8 GB runs out once a few domains are cached.
- **Storage:** 40 GB gp3. The backend image (CPU torch) and the model cache
  volume take most of it.
- **Security group:** inbound TCP 22 from your IP only, TCP 80 from where
  you want to serve it. Do not open 6333 or 8000.

## 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker
```

## 3. Get the code and configure

```bash
git clone https://github.com/GurramVardhanReddy/RAG.git
cd RAG
cp .env.example .env
nano .env            # set OPENAI_API_KEY; keep the key out of git
mkdir -p data/uploads
sudo chown 1000:1000 data/uploads   # the backend runs as uid 1000
```

## 4. Start

```bash
docker compose up -d --build
docker compose ps
curl -s localhost/health
```

`/health` returns HTTP 200 with `"status": "healthy"` once the backend has
reached Qdrant, and 503 with `"status": "degraded"` until then. The first
upload in each domain downloads that domain's embedding model into the
`hf_models` volume, so it is slow once and fast afterwards.

## 5. Operate

```bash
docker compose logs -f backend      # application logs
docker compose pull && docker compose up -d --build   # update after git pull
docker compose down                 # stop (volumes keep data and models)
```

## Upgrading an existing deployment

Collections created before the BM25 arm have no `bm25` sparse vector. The
backend detects this, logs a warning and serves those collections dense-only.
To get hybrid retrieval for them, delete the collection and re-upload its
documents:

```bash
curl -X DELETE localhost:6333/collections/explainable_rag__general   # run on the instance
```

## HTTPS

Put the instance behind an Application Load Balancer with an ACM
certificate, or add a TLS-terminating proxy (e.g. Caddy) in front of nginx.
Do not serve uploads of private documents over plain HTTP.
