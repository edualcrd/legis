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
                "subject": "Acceso a Legis — Sistema de consulta jurídica laboral",
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
    """
    Construye el cuerpo HTML del correo de magic link.

    HTML compatible con Gmail, Outlook (incluido el motor Word de MSO) y
    Apple Mail: ancho máximo 600px, tabla principal centrada, CSS inline,
    botón CTA implementado con celda de tabla (no padding sobre <a>) para
    que Outlook respete el área clicable. Sin imágenes externas; el sello
    "§" se compone con la pila de fuentes serif del sistema operativo.
    """
    # Paleta editorial — los mismos tokens que la app y la landing.
    ink = "#0F1924"        # tinta
    ink_muted = "#6B7280"  # gris cálido
    ink_faint = "#A39A87"  # marfil sombreado
    bg = "#FBFAF6"         # papel marfil
    surface = "#FFFFFF"    # tarjeta blanca
    primary = "#185FA5"    # cobalto
    primary_deep = "#0A2F58"
    accent = "#B07D3B"     # cobre oxidado
    accent_soft = "#F4EBDC"
    line = "#DDD6C8"       # línea editorial

    # Pilas de fuentes con fallbacks robustos en clientes que no cargan web fonts.
    serif = "Fraunces, 'Iowan Old Style', 'Apple Garamond', Georgia, 'Times New Roman', serif"
    sans = ("'General Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Roboto, 'Helvetica Neue', Arial, sans-serif")
    mono = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, 'Courier New', monospace"

    expiry = MAGIC_TOKEN_EXPIRY_DAYS

    return f"""\
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="x-apple-disable-message-reformatting" />
<meta name="color-scheme" content="light only" />
<meta name="supported-color-schemes" content="light only" />
<title>Acceso a Legis</title>
</head>
<body style="margin:0; padding:0; background:{bg}; color:{ink}; -webkit-font-smoothing:antialiased; -webkit-text-size-adjust:100%; font-family:{sans};">

  <!-- Preheader oculto: lo que se ve en la lista del inbox -->
  <div style="display:none; max-height:0; overflow:hidden; mso-hide:all; font-size:1px; line-height:1px; color:{bg};">
    Tu enlace de acceso de un solo uso para Legis. Expira en {expiry} días.
  </div>

  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
         style="background:{bg}; padding:32px 16px;">
    <tr>
      <td align="center">

        <!-- Tarjeta editorial 600px -->
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="600"
               style="width:100%; max-width:600px; background:{surface}; border:1px solid {line}; border-radius:8px;">

          <!-- Cabecera: § cobre + Legis serif -->
          <tr>
            <td style="padding:36px 40px 0 40px;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td valign="middle"
                      style="font-family:{serif}; font-style:italic; font-weight:400; color:{accent}; font-size:32px; line-height:1; padding-right:10px;">
                    §
                  </td>
                  <td valign="middle"
                      style="font-family:{serif}; font-weight:500; color:{ink}; font-size:28px; line-height:1; letter-spacing:-0.5px;">
                    Legis
                  </td>
                </tr>
              </table>
              <!-- Línea cobre divisora -->
              <div style="width:56px; height:1px; background:{accent}; margin:22px 0 0 0; font-size:0; line-height:0;">&nbsp;</div>
            </td>
          </tr>

          <!-- Eyebrow editorial -->
          <tr>
            <td style="padding:28px 40px 0 40px;">
              <div style="font-family:{sans}; font-size:11px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:{accent};">
                Acceso por invitación
              </div>
            </td>
          </tr>

          <!-- Título -->
          <tr>
            <td style="padding:10px 40px 0 40px;">
              <h1 style="margin:0; font-family:{serif}; font-weight:500; color:{ink}; font-size:32px; line-height:1.1; letter-spacing:-0.6px;">
                Acceso a Legis
              </h1>
              <p style="margin:8px 0 0 0; font-family:{serif}; font-style:italic; font-size:17px; color:#3D4654; line-height:1.45;">
                Sistema de consulta jurídica laboral.
              </p>
            </td>
          </tr>

          <!-- Cuerpo -->
          <tr>
            <td style="padding:28px 40px 0 40px;">
              <p style="margin:0 0 14px 0; font-family:{sans}; font-size:15px; line-height:1.6; color:{ink};">
                Recibimos una solicitud de acceso a <strong style="font-weight:600;">Legis</strong> con este correo.
                Usa el siguiente enlace para entrar al sistema. El acceso es de un solo uso y caduca en
                <strong style="font-weight:600;">{expiry} días</strong>.
              </p>
            </td>
          </tr>

          <!-- CTA cobalto (tabla para Outlook) -->
          <tr>
            <td style="padding:28px 40px 4px 40px;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td bgcolor="{primary}" style="background:{primary}; border-radius:6px;">
                    <a href="{link}"
                       style="display:inline-block; padding:14px 28px; font-family:{sans}; font-size:15px; font-weight:500; letter-spacing:0.2px; color:#FBFAF6; text-decoration:none; border-radius:6px;">
                      Acceder al sistema
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Enlace en texto plano para clientes sin botones -->
          <tr>
            <td style="padding:18px 40px 0 40px;">
              <p style="margin:0; font-family:{sans}; font-size:12px; color:{ink_muted}; line-height:1.55;">
                Si el botón no funciona, copia y pega este enlace en tu navegador:
              </p>
              <p style="margin:8px 0 0 0; font-family:{mono}; font-size:12px; line-height:1.5; word-break:break-all; color:{primary_deep};">
                <a href="{link}" style="color:{primary_deep}; text-decoration:underline;">{link}</a>
              </p>
            </td>
          </tr>

          <!-- Pie editorial -->
          <tr>
            <td style="padding:30px 40px 36px 40px;">
              <div style="border-top:1px solid {line}; padding-top:18px;">
                <p style="margin:0; font-family:{sans}; font-size:12.5px; color:{ink_muted}; line-height:1.6;">
                  Este enlace expira en <strong style="color:{ink}; font-weight:600;">{expiry} días</strong>.
                  Si no solicitaste este acceso, ignora este mensaje — no se realizará ninguna acción.
                </p>
                <p style="margin:14px 0 0 0; font-family:{serif}; font-style:italic; font-size:13px; color:{ink_faint}; line-height:1.5;">
                  El precedente exacto, en segundos.
                </p>
              </div>
            </td>
          </tr>

        </table>

        <!-- Pie corporativo fuera de la tarjeta -->
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="600"
               style="width:100%; max-width:600px; margin-top:20px;">
          <tr>
            <td align="center" style="padding:8px 24px 4px 24px; font-family:{sans}; font-size:11px; line-height:1.6; color:{ink_faint}; letter-spacing:0.4px;">
              Legis · Asistente de consulta jurídica laboral · Ciudad de México
            </td>
          </tr>
          <tr>
            <td align="center" style="padding:4px 24px 16px 24px; font-family:{sans}; font-size:11px; color:{ink_faint};">
              <a href="mailto:eduardoalcaiderodriguez@gmail.com" style="color:{ink_muted}; text-decoration:none;">eduardoalcaiderodriguez@gmail.com</a>
            </td>
          </tr>
        </table>

      </td>
    </tr>
  </table>

</body>
</html>"""


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
