# Imagen de despliegue de Legis (FastAPI + SPA React).
# Ligera: tras migrar embeddings y reranking a Voyage (API), ya no se instala
# torch ni modelos locales, así que la imagen es de ~cientos de MB y cabe en
# el free tier de Render.
FROM python:3.11-slim

# No escribir .pyc; salida sin buffer (logs en vivo en Render).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero: esta capa queda cacheada mientras requirements.txt
# no cambie, acelerando rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la app (se copia antes del ingest porque éste lo importa).
COPY src/ ./src/

# Corpus crudo: el ingest lo procesa para construir el índice ChromaDB.
COPY corpus/ ./corpus/

# Construye el índice ChromaDB en build time vía Voyage API. La clave se
# pasa como ARG desde Render y se inyecta SOLO durante el RUN — no se
# hace `ENV` para no dejar el secreto incrustado en la imagen final.
# En runtime, Render inyecta VOYAGE_API_KEY desde envVars del blueprint.
ARG VOYAGE_API_KEY
RUN VOYAGE_API_KEY=$VOYAGE_API_KEY python -m src.ingest

# Frontend al final: cambiarlo no invalida la capa cara del re-indexado.
COPY frontend/ ./frontend/

# El RAG abre el índice desde esta ruta dentro del contenedor.
ENV CHROMA_PERSIST_DIR=/app/chroma_db

EXPOSE 8000

# Render inyecta $PORT; en local cae a 8000. Shell form para expandir $PORT.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
