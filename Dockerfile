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

# Código de la app, SPA y el índice ChromaDB ya re-indexado con voyage-3.
# El índice se hornea en la imagen (solo lectura en runtime): el corpus es
# estático, así que no necesitamos disco persistente de pago.
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY chroma_db/ ./chroma_db/

# El RAG abre el índice desde esta ruta dentro del contenedor.
ENV CHROMA_PERSIST_DIR=/app/chroma_db

EXPOSE 8000

# Render inyecta $PORT; en local cae a 8000. Shell form para expandir $PORT.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
