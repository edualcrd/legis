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

    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "chroma_db")
    corpus_raw_dir: Path = Field(default=PROJECT_ROOT / "corpus" / "raw")
    corpus_processed_dir: Path = Field(default=PROJECT_ROOT / "corpus" / "processed")

    llm_model: str = Field(default="claude-haiku-4-5-20251001")
    embed_model: str = Field(default="BAAI/bge-m3")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")

    log_level: str = Field(default="INFO")


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

    return Settings(
        anthropic_api_key=api_key,
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
        embed_model=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
        rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
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
