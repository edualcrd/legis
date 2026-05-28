"""
FastAPI app que expone el RAG de Legis a la SPA React.

Endpoints:
    POST /api/query    → llama a `answer_query()` de src/rag.py
    GET  /api/stats    → corpus stats vía `get_corpus_stats()` de src/rag.py
    POST /api/waitlist → registra solicitud de acceso beta (JSONL local)
    GET  /landing      → sirve landing.html (página comercial)
    GET  /*            → sirve frontend/index.html y assets (StaticFiles)

Decisión de diseño: las rutas usan `def` (no `async def`). El RAG es
síncrono y CPU-bound (embeddings, reranker, llamada al LLM). FastAPI
ejecuta automáticamente las rutas `def` en un threadpool, lo que evita
bloquear el event loop sin necesidad de envolver manualmente en
`asyncio.to_thread`. Más simple y correcto para este caso.

Uso:
    uvicorn src.api:app --reload --port 8000
    # Abrir http://localhost:8000/
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.auth import (
    add_user,
    consume_magic_token,
    create_magic_token,
    is_authorized_email,
    issue_session,
    send_magic_link,
    verify_session,
)
from src.rag import answer_query, get_corpus_stats
from src.utils import get_logger, load_config

logger = get_logger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
LANDING_FILE: Path = PROJECT_ROOT / "landing.html"
PRIVACY_FILE: Path = PROJECT_ROOT / "privacy.html"
TERMS_FILE: Path = PROJECT_ROOT / "terms.html"
WAITLIST_DIR: Path = PROJECT_ROOT / "data"
WAITLIST_FILE: Path = WAITLIST_DIR / "waitlist.jsonl"


# === Modelos Pydantic ===

class QueryRequest(BaseModel):
    """Cuerpo de POST /api/query."""

    query: str = Field(
        ..., min_length=1, max_length=2000,
        description="Consulta del abogado en lenguaje natural",
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Chunks finales al LLM tras reranking",
    )
    retrieve_k: int = Field(
        default=20, ge=5, le=100,
        description="Candidatos de ChromaDB antes del reranker",
    )


class Source(BaseModel):
    """Una fuente citada en la respuesta."""

    citation: str
    source: str
    doc_id: str
    page: int | str = 0
    rerank_score: float = 0.0
    mandatory: bool = False
    type: str = ""


class QueryResponse(BaseModel):
    """Respuesta de POST /api/query."""

    answer: str
    sources: list[Source]
    confidence: float


class StatsResponse(BaseModel):
    """Respuesta de GET /api/stats."""

    chunks: int
    documents: int
    juris: int
    tesis: int
    sources: list[str]
    # Nombres legibles de leyes/reglamentos/códigos indexados (sin SCJN ni IMSS),
    # para poblar dinámicamente "Fuentes activas" en el sidebar de la SPA.
    leyes: list[str] = []


class ErrorResponse(BaseModel):
    """Respuesta de error genérica con mensaje en español."""

    detail: str
    code: Literal["empty_query", "corpus_not_indexed", "internal_error"] = "internal_error"


class WaitlistRequest(BaseModel):
    """Cuerpo de POST /api/waitlist — solicitud de acceso beta."""

    nombre: str = Field(..., min_length=2, max_length=120)
    despacho: str = Field(..., min_length=2, max_length=160)
    # Regex permisivo pero descarta basura obvia (espacios, sin @, sin TLD).
    email: str = Field(..., min_length=5, max_length=200, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    whatsapp: str = Field(..., min_length=8, max_length=24)
    plan: Literal["independiente", "estudio", "firma", ""] = ""
    source: str = Field(default="landing", max_length=40)


class WaitlistResponse(BaseModel):
    """Respuesta de POST /api/waitlist."""

    ok: bool = True
    message: str


# Regex de email compartido: permisivo pero descarta basura obvia.
_EMAIL_PATTERN = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"


class RequestAccessIn(BaseModel):
    """Cuerpo de POST /api/auth/request-access."""

    email: str = Field(..., min_length=5, max_length=200, pattern=_EMAIL_PATTERN)


class MessageOut(BaseModel):
    """Respuesta genérica con un mensaje en español."""

    message: str


class VerifyOut(BaseModel):
    """Respuesta de GET /api/auth/verify."""

    valid: bool
    email: str | None = None
    session_token: str | None = None
    reason: str | None = None


class AddUserIn(BaseModel):
    """Cuerpo de POST /api/auth/add-user (solo admin, con API key)."""

    email: str = Field(..., min_length=5, max_length=200, pattern=_EMAIL_PATTERN)
    api_key: str = Field(..., min_length=1, max_length=200)


# === App ===

app = FastAPI(
    title="Legis API",
    description="RAG legal mexicano: SCJN, LFT, IMSS — sirve la SPA React de Legis.",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

ALLOWED_ORIGINS: list[str] = [
    "https://legis-psi.vercel.app",
    "https://legis.mx",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def require_session(authorization: str | None = Header(default=None)) -> str:
    """
    Dependencia que exige un JWT de sesión válido para acceder a rutas
    protegidas. Lee el header `Authorization: Bearer <token>`, lo valida con
    `verify_session` y devuelve el email autenticado. Responde 401 (en español)
    si falta el token, está malformado, expiró o la firma no es válida.

    Args:
        authorization: Valor del header Authorization inyectado por FastAPI.

    Returns:
        El email del usuario autenticado.

    Raises:
        HTTPException: 401 si no hay una sesión válida.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Falta el token de sesión. Inicia sesión para continuar.",
        )
    token = authorization.split(" ", 1)[1].strip()
    email = verify_session(token)
    if not email:
        raise HTTPException(
            status_code=401,
            detail="Tu sesión expiró o no es válida. Vuelve a iniciar sesión.",
        )
    return email


@app.post(
    "/api/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Consulta inválida"},
        401: {"model": ErrorResponse, "description": "Sesión inválida o ausente"},
        503: {"model": ErrorResponse, "description": "Corpus aún no indexado"},
        500: {"model": ErrorResponse, "description": "Error interno (LLM o RAG)"},
    },
    tags=["RAG"],
    summary="Responde una consulta legal con citas verificables",
)
def post_query(
    req: QueryRequest,
    email: str = Depends(require_session),
) -> QueryResponse:
    """
    Ejecuta el pipeline RAG completo y devuelve la respuesta con fuentes.

    Protegida: requiere un JWT de sesión válido (header Authorization: Bearer).

    Errores:
        - 400: consulta vacía o malformada
        - 401: sesión inválida o ausente
        - 503: ChromaDB no inicializado o corpus vacío (ejecutar `python src/ingest.py`)
        - 500: fallo del LLM o error inesperado
    """
    logger.info("POST /api/query — usuario: %s, query: %s", email, req.query[:120])
    try:
        result = answer_query(
            req.query,
            top_k=req.top_k,
            retrieve_k=req.retrieve_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        msg = str(exc)
        if "ingest" in msg.lower() or "colección" in msg.lower():
            raise HTTPException(status_code=503, detail=msg) from exc
        raise HTTPException(status_code=500, detail=msg) from exc
    except Exception as exc:
        logger.exception("Error inesperado en /api/query")
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado al procesar la consulta: {exc}",
        ) from exc

    return QueryResponse(**result)


@app.get(
    "/api/stats",
    response_model=StatsResponse,
    tags=["Corpus"],
    summary="Estadísticas del corpus indexado en ChromaDB",
)
def get_stats() -> StatsResponse:
    """
    Devuelve métricas agregadas del corpus para el sidebar de la SPA.

    Cuando el corpus aún no se ha indexado, devuelve la misma estructura
    con ceros y array vacío — la UI lo renderiza con guiones.
    """
    stats = get_corpus_stats()
    return StatsResponse(**stats)


@app.post(
    "/api/waitlist",
    response_model=WaitlistResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "Datos inválidos"},
        500: {"model": ErrorResponse, "description": "No se pudo persistir la solicitud"},
    },
    tags=["Waitlist"],
    summary="Registra una solicitud de acceso beta",
)
def post_waitlist(req: WaitlistRequest) -> WaitlistResponse:
    """
    Persiste la solicitud como línea JSON en data/waitlist.jsonl.

    Cada línea es un objeto JSON independiente con timestamp UTC, lo que
    permite leer el archivo con `cat`, procesarlo con jq o convertirlo a
    CSV sin parsear toda la historia. Si el directorio no existe, se crea.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nombre": req.nombre.strip(),
        "despacho": req.despacho.strip(),
        "email": req.email,
        "whatsapp": req.whatsapp.strip(),
        "plan": req.plan,
        "source": req.source,
    }

    try:
        WAITLIST_DIR.mkdir(parents=True, exist_ok=True)
        with WAITLIST_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.exception("No se pudo escribir en %s", WAITLIST_FILE)
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo guardar la solicitud: {exc}",
        ) from exc

    logger.info(
        "Solicitud beta registrada: %s (%s) plan=%s source=%s",
        record["email"], record["despacho"], record["plan"], record["source"],
    )
    return WaitlistResponse(
        ok=True,
        message="Solicitud recibida. Te contactamos en menos de 24 horas hábiles.",
    )


# === Autenticación beta (magic link) ===

@app.post(
    "/api/auth/request-access",
    response_model=MessageOut,
    tags=["Auth"],
    summary="Solicita un magic link de acceso",
)
def request_access(req: RequestAccessIn) -> MessageOut:
    """
    Si el email está autorizado y activo, genera un token de un solo uso y
    envía el magic link por correo (Resend).

    Por seguridad (anti-enumeración) responde SIEMPRE el mismo mensaje, exista
    o no el usuario: así un tercero no puede descubrir qué correos están dados
    de alta probando direcciones.
    """
    generic = MessageOut(
        message="Si tu correo está autorizado, recibirás un enlace de acceso en breve.",
    )
    try:
        if is_authorized_email(req.email):
            token = create_magic_token(req.email)
            send_magic_link(req.email, token)
            logger.info("Magic link solicitado y enviado: %s", req.email)
        else:
            logger.info("Solicitud de acceso de email NO autorizado: %s", req.email)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return generic


@app.get(
    "/api/auth/verify",
    response_model=VerifyOut,
    tags=["Auth"],
    summary="Verifica un magic token y emite la sesión",
)
def verify(token: str = Query(..., min_length=1)) -> VerifyOut:
    """
    Consume el token del magic link (lo marca usado). Si es válido, emite un
    JWT de sesión y lo devuelve junto al email; si no, devuelve la razón.
    """
    try:
        valid, email, reason = consume_magic_token(token)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not valid or not email:
        return VerifyOut(valid=False, reason=reason)

    session_token = issue_session(email)
    return VerifyOut(valid=True, email=email, session_token=session_token)


@app.post(
    "/api/auth/add-user",
    response_model=MessageOut,
    responses={403: {"model": ErrorResponse, "description": "API key inválida"}},
    tags=["Auth"],
    summary="Añade un usuario autorizado (solo admin)",
)
def post_add_user(req: AddUserIn) -> MessageOut:
    """
    Inserta un email en la tabla `usuarios`. Operación de administrador:
    requiere la ADMIN_API_KEY del entorno. Idempotente.
    """
    config = load_config()
    if not config.admin_api_key or req.api_key != config.admin_api_key:
        raise HTTPException(
            status_code=403, detail="API key de administrador inválida.",
        )
    try:
        add_user(req.email)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return MessageOut(message="Usuario añadido")


@app.get("/auth", include_in_schema=False)
def get_auth_page() -> FileResponse:
    """
    Sirve la SPA para que procese el `?token` del magic link.

    Sin esta ruta, GET /auth caería al StaticFiles montado en "/" y devolvería
    404 (no existe auth/index.html). Aquí devolvemos el index.html de la SPA,
    que en el cliente lee el token de la URL y llama a /api/auth/verify.
    """
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend no encontrado")
    return FileResponse(index, media_type="text/html")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    """
    Health check ligero para Render. No toca Supabase, Anthropic ni Voyage:
    solo confirma que el proceso responde.
    """
    return {"status": "ok"}


# === Landing comercial (separada de la SPA) ===

@app.get("/landing", include_in_schema=False)
def get_landing() -> FileResponse:
    """
    Sirve la landing comercial estática. La SPA principal queda en `/`.
    """
    if not LANDING_FILE.exists():
        raise HTTPException(status_code=404, detail="landing.html no encontrado")
    return FileResponse(LANDING_FILE, media_type="text/html")


# === Páginas legales (Aviso de Privacidad y Términos de Uso) ===
# Se sirven desde la raíz del proyecto en lugar de frontend/, para mantener
# una sola copia compartida entre la landing (Vercel) y la SPA (Render).

@app.get("/privacy.html", include_in_schema=False)
def get_privacy() -> FileResponse:
    """Sirve el Aviso de Privacidad (LFPDPPP)."""
    if not PRIVACY_FILE.exists():
        raise HTTPException(status_code=404, detail="privacy.html no encontrado")
    return FileResponse(PRIVACY_FILE, media_type="text/html")


@app.get("/terms.html", include_in_schema=False)
def get_terms() -> FileResponse:
    """Sirve los Términos de Uso."""
    if not TERMS_FILE.exists():
        raise HTTPException(status_code=404, detail="terms.html no encontrado")
    return FileResponse(TERMS_FILE, media_type="text/html")


# === Static mount ===
# IMPORTANTE: el mount al "/" va al FINAL, después de registrar todas las
# rutas de API. FastAPI evalúa rutas en orden de registro. StaticFiles con
# html=True sirve index.html para "/" y permite GET de cualquier asset.

if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
else:
    logger.warning(
        "El directorio %s no existe. La SPA no se servirá. "
        "Verifica que frontend/index.html esté presente.",
        FRONTEND_DIR,
    )
