"""
Autenticación beta de Legis por magic link (Supabase + Resend + JWT).

Flujo:
    1. El abogado pide acceso con su email (POST /api/auth/request-access).
    2. Si el email está autorizado en la tabla `usuarios` (activo=true), se
       genera un token de un solo uso (uuid4) en `magic_tokens` con 24 h de
       vigencia y se le envía por correo (Resend) un enlace
       {APP_BASE_URL}/auth?token=...
    3. Al abrir el enlace, la SPA llama GET /api/auth/verify?token=..., que
       consume el token (lo marca usado) y, si es válido, emite un JWT de
       sesión (~7 días) firmado con SESSION_SECRET.
    4. La SPA guarda el JWT y lo manda en `Authorization: Bearer ...` en cada
       consulta; /api/query lo valida con `verify_session`.

Las rutas HTTP viven en src/api.py; este módulo solo expone la lógica.
Convención del repo: código en inglés, mensajes/errores en español.

Toda la comunicación con Supabase usa la SERVICE_ROLE key (server-side,
ignora RLS). Esa key nunca llega al frontend.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
import resend
from supabase import Client, create_client

from src.utils import Settings, get_logger, load_config

logger = get_logger(__name__)

# Vigencia del magic link y de la sesión.
MAGIC_TOKEN_EXPIRY_DAYS: int = 7
SESSION_EXPIRY_DAYS: int = 7
JWT_ALGORITHM: str = "HS256"

# === Singletons / config ===

_config: Settings | None = None
_supabase: Client | None = None


def _get_config() -> Settings:
    """Carga (una vez) y cachea la configuración del proyecto."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _require(value: str, env_name: str) -> str:
    """
    Devuelve `value` si no está vacío; si no, lanza un error en español que
    indica qué variable de entorno falta. Las variables de auth son opcionales
    para el RAG, así que solo se exigen aquí, en el punto de uso.
    """
    if not value:
        raise RuntimeError(
            f"Falta la variable {env_name} en el entorno. Es necesaria para la "
            f"autenticación por magic link. Configúrala en el .env (local) o en "
            f"el dashboard de Render (producción)."
        )
    return value


def _get_supabase() -> Client:
    """
    Devuelve el cliente de Supabase, inicializado una sola vez por proceso con
    la URL y la SERVICE_ROLE key del proyecto.
    """
    global _supabase
    if _supabase is None:
        config = _get_config()
        url = _require(config.supabase_url, "SUPABASE_URL")
        key = _require(config.supabase_key, "SUPABASE_KEY")
        logger.info("Inicializando cliente Supabase...")
        _supabase = create_client(url, key)
        logger.info("Cliente Supabase listo")
    return _supabase


# === Usuarios ===

def is_authorized_email(email: str) -> bool:
    """
    Indica si `email` corresponde a un usuario autorizado y activo.

    Args:
        email: Correo a verificar (se normaliza a minúsculas + trim).

    Returns:
        True si existe en `usuarios` con activo=true, False en otro caso.
    """
    normalized = email.strip().lower()
    supabase = _get_supabase()
    try:
        result = (
            supabase.table("usuarios")
            .select("id")
            .eq("email", normalized)
            .eq("activo", True)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 — superficie de error de Supabase
        logger.exception("Error consultando usuarios en Supabase")
        raise RuntimeError(
            f"No se pudo verificar el usuario en Supabase: {exc}"
        ) from exc
    return bool(result.data)


def add_user(email: str, plan: str = "beta") -> None:
    """
    Inserta un usuario autorizado en `usuarios`. Idempotente: si el email ya
    existe, no falla (lo deja como está).

    Args:
        email: Correo a autorizar (se normaliza a minúsculas + trim).
        plan: Plan asignado (default 'beta').
    """
    normalized = email.strip().lower()
    supabase = _get_supabase()
    try:
        supabase.table("usuarios").upsert(
            {"email": normalized, "plan": plan, "activo": True},
            on_conflict="email",
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error insertando usuario en Supabase")
        raise RuntimeError(
            f"No se pudo añadir el usuario en Supabase: {exc}"
        ) from exc
    logger.info("Usuario autorizado: %s (plan=%s)", normalized, plan)


# === Magic tokens ===

def create_magic_token(email: str) -> str:
    """
    Genera y persiste un token de un solo uso para `email`.

    Args:
        email: Correo del usuario autorizado (ya verificado por el llamador).

    Returns:
        El token (uuid4 en hex) a incrustar en el magic link.
    """
    normalized = email.strip().lower()
    token = uuid.uuid4().hex
    expira_at = datetime.now(timezone.utc) + timedelta(days=MAGIC_TOKEN_EXPIRY_DAYS)

    supabase = _get_supabase()
    try:
        supabase.table("magic_tokens").insert(
            {
                "email": normalized,
                "token": token,
                "usado": False,
                "expira_at": expira_at.isoformat(),
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error creando magic token en Supabase")
        raise RuntimeError(
            f"No se pudo generar el enlace de acceso: {exc}"
        ) from exc

    logger.info("Magic token creado para %s (expira %s)", normalized, expira_at.isoformat())
    return token


def consume_magic_token(token: str) -> tuple[bool, str | None, str | None]:
    """
    Valida y consume un magic token.

    Verifica que el token exista, no esté usado y no haya expirado. Si es
    válido, lo marca como usado (un solo uso) y devuelve el email asociado.

    Args:
        token: Token recibido en el query string del magic link.

    Returns:
        Tupla (valido, email, razon):
            - (True, email, None) si el token es válido.
            - (False, None, razon) si no, con la razón en español.
    """
    if not token:
        return False, None, "Falta el token."

    supabase = _get_supabase()
    try:
        result = (
            supabase.table("magic_tokens")
            .select("email, usado, expira_at")
            .eq("token", token)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error consultando magic token en Supabase")
        raise RuntimeError(
            f"No se pudo verificar el enlace de acceso: {exc}"
        ) from exc

    rows = result.data or []
    if not rows:
        return False, None, "El enlace no es válido."

    row = rows[0]
    if row.get("usado"):
        return False, None, "El enlace ya fue usado. Solicita uno nuevo."

    expira_at = _parse_timestamp(row.get("expira_at"))
    if expira_at is None or expira_at < datetime.now(timezone.utc):
        return False, None, "El enlace expiró. Solicita uno nuevo."

    # Marca el token como usado (un solo uso).
    try:
        supabase.table("magic_tokens").update({"usado": True}).eq("token", token).execute()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error marcando magic token como usado")
        raise RuntimeError(
            f"No se pudo completar la verificación del enlace: {exc}"
        ) from exc

    email = row.get("email")
    logger.info("Magic token consumido para %s", email)
    return True, email, None


def _parse_timestamp(raw: str | None) -> datetime | None:
    """
    Parsea un timestamptz devuelto por Supabase a un datetime con tz UTC.

    Tolera el sufijo 'Z' y zonas con offset. Devuelve None si no se puede
    parsear (defensivo: se tratará como expirado aguas arriba).
    """
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Timestamp no parseable de Supabase: %r", raw)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# === Envío del magic link (Resend) ===

def send_magic_link(email: str, token: str) -> None:
    """
    Envía el correo con el magic link vía Resend.

    Args:
        email: Destinatario (usuario autorizado).
        token: Token generado por `create_magic_token`.
    """
    config = _get_config()
    api_key = _require(config.resend_api_key, "RESEND_API_KEY")
    resend.api_key = api_key

    link = f"{config.app_base_url}/auth?token={token}"
    html = _magic_link_html(link)

    try:
        resend.Emails.send(
            {
                "from": config.resend_from_email,
                "to": [email],
                "subject": "Tu acceso a Legis",
                "html": html,
            }
        )
    except Exception as exc:  # noqa: BLE001 — superficie de error de Resend
        logger.exception("Error enviando el magic link con Resend")
        raise RuntimeError(
            f"No se pudo enviar el correo de acceso: {exc}"
        ) from exc

    logger.info("Magic link enviado a %s", email)


def _magic_link_html(link: str) -> str:
    """Construye el cuerpo HTML del correo de magic link."""
    return f"""\
<div style="font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 480px; margin: 0 auto; color: #1A1A2E;">
  <h2 style="color: #185FA5;">⚖ Legis</h2>
  <p>Hola,</p>
  <p>Recibimos una solicitud de acceso a <strong>Legis</strong> con este correo.
     Entra con el siguiente enlace (válido por {MAGIC_TOKEN_EXPIRY_DAYS} días):</p>
  <p style="margin: 28px 0;">
    <a href="{link}"
       style="background: #185FA5; color: #fff; padding: 12px 24px; border-radius: 8px;
              text-decoration: none; font-weight: 600;">
      Entrar a Legis
    </a>
  </p>
  <p style="font-size: 13px; color: #6B7280;">
    Si tú no solicitaste esto, puedes ignorar este correo. El enlace solo
    funciona una vez.
  </p>
  <p style="font-size: 13px; color: #6B7280;">El precedente exacto, en segundos.</p>
</div>"""


# === Sesión (JWT) ===

def issue_session(email: str) -> str:
    """
    Emite un JWT de sesión firmado con SESSION_SECRET.

    Args:
        email: Correo del usuario autenticado.

    Returns:
        JWT codificado (str) con claims `email` y `exp` (~7 días).
    """
    config = _get_config()
    secret = _require(config.session_secret, "SESSION_SECRET")
    payload = {
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_session(token: str) -> str | None:
    """
    Valida un JWT de sesión y devuelve el email si es válido.

    Args:
        token: JWT recibido en el header Authorization.

    Returns:
        El email del claim si la firma es válida y no expiró; None si el token
        es inválido, está expirado o malformado.
    """
    if not token:
        return None
    config = _get_config()
    secret = _require(config.session_secret, "SESSION_SECRET")
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    email = payload.get("email")
    return email if isinstance(email, str) else None
