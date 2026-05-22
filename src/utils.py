"""
Utilidades compartidas: logger configurado y carga de configuración desde .env.

Todo el resto del código (ingest, rag, app) debe obtener su logger vía
`get_logger(__name__)` y su configuración vía `load_config()`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    """Configuración tipada del proyecto, cargada desde variables de entorno."""

    anthropic_api_key: str = Field(..., description="API key de Anthropic")
    voyage_api_key: str = Field(..., description="API key de Voyage AI (embeddings + rerank)")

    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "chroma_db")
    corpus_raw_dir: Path = Field(default=PROJECT_ROOT / "corpus" / "raw")
    corpus_processed_dir: Path = Field(default=PROJECT_ROOT / "corpus" / "processed")

    llm_model: str = Field(default="claude-haiku-4-5-20251001")
    embed_model: str = Field(default="voyage-3")
    rerank_model: str = Field(default="rerank-2.5")

    log_level: str = Field(default="INFO")

    # === Auth beta (magic link) — opcionales para el RAG, requeridas por la API ===
    # Se cargan siempre pero solo se validan en src/auth.py al usarlas, para que
    # `python src/ingest.py` y el CLI de rag funcionen sin configurar el login.
    resend_api_key: str = Field(default="")
    resend_from_email: str = Field(default="onboarding@resend.dev")
    supabase_url: str = Field(default="")
    supabase_key: str = Field(default="")
    admin_api_key: str = Field(default="")
    app_base_url: str = Field(default="http://localhost:8000")
    session_secret: str = Field(default="")


def load_config() -> Settings:
    """
    Carga las variables del archivo .env y devuelve un objeto Settings tipado.

    Returns:
        Instancia de Settings validada por Pydantic.

    Raises:
        ValueError: Si falta ANTHROPIC_API_KEY en el entorno.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Falta la variable ANTHROPIC_API_KEY en el archivo .env. "
            "Copia .env.example a .env y configura tu API key de Anthropic."
        )

    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not voyage_key:
        raise ValueError(
            "Falta la variable VOYAGE_API_KEY en el archivo .env. "
            "Es necesaria para los embeddings y el reranking (Voyage AI). "
            "Obtén una key en https://dashboard.voyageai.com/ y añádela al .env."
        )

    return Settings(
        anthropic_api_key=api_key,
        voyage_api_key=voyage_key,
        chroma_persist_dir=Path(
            os.environ.get("CHROMA_PERSIST_DIR", PROJECT_ROOT / "chroma_db")
        ),
        corpus_raw_dir=Path(
            os.environ.get("CORPUS_RAW_DIR", PROJECT_ROOT / "corpus" / "raw")
        ),
        corpus_processed_dir=Path(
            os.environ.get("CORPUS_PROCESSED_DIR", PROJECT_ROOT / "corpus" / "processed")
        ),
        llm_model=os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"),
        embed_model=os.environ.get("EMBED_MODEL", "voyage-3"),
        rerank_model=os.environ.get("RERANK_MODEL", "rerank-2.5"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        resend_api_key=os.environ.get("RESEND_API_KEY", "").strip(),
        resend_from_email=os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev").strip(),
        supabase_url=os.environ.get("SUPABASE_URL", "").strip(),
        supabase_key=os.environ.get("SUPABASE_KEY", "").strip(),
        admin_api_key=os.environ.get("ADMIN_API_KEY", "").strip(),
        app_base_url=os.environ.get("APP_BASE_URL", "http://localhost:8000").strip().rstrip("/"),
        session_secret=os.environ.get("SESSION_SECRET", "").strip(),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger configurado con el formato estándar del proyecto.

    El nivel se toma de la variable LOG_LEVEL del entorno (por defecto INFO).
    Es idempotente: llamarlo varias veces con el mismo nombre no duplica handlers.

    Args:
        name: Nombre del logger, típicamente `__name__` del módulo que llama.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
