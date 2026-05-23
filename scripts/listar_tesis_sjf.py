"""
Enumera tesis del Semanario Judicial de la Federación y filtra las laborales.

Usa el endpoint público no documentado del SJF (verificado el 2026-05-17):
    POST https://sjf2.scjn.gob.mx/services/sjftesismicroservice/api/public/tesis
        ?page=N&size=100
    body: {}  (sin payload de búsqueda — el formato exacto está minificado
              en el bundle Angular y requiere DevTools para extraerlo)

Como el endpoint solo devuelve METADATOS (texto, materias, nombreArchivo
vienen null), filtramos cada tesis por palabras clave en el campo `rubro`
y guardamos el rubro + clave + localización como .txt en corpus/raw/.
El rubro de una tesis SCJN ya describe su tema en profundidad y es
suficiente para que el RAG haga matching semántico, aunque no equivale
al texto completo de la tesis.

Convención de nombres (skills/corpus-ingest.md §2):
    SCJN_J_{clave_normalizada}_{epoca}.txt  (jurisprudencia obligatoria)
    SCJN_T_{clave_normalizada}_{epoca}.txt  (tesis aislada)

Uso:
    python scripts/listar_tesis_sjf.py
    python scripts/listar_tesis_sjf.py --max-pages 200 --page-size 100
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import anthropic
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import SINTESIS_TESIS_SYSTEM_PROMPT  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

logger = get_logger("listar_tesis_sjf")

# === Enriquecimiento de tesis J con síntesis del rubro (Claude Haiku) ===
# Solo se enriquecen las jurisprudencias OBLIGATORIAS (J): son las que el
# abogado cita como precedente vinculante y las que más sufren el problema de
# "rubro escueto → relevancia débil en el reranker". Las tesis aisladas (T) se
# dejan igual para acotar costo (ver --estimar-costo).
ETIQUETA_SINTESIS: str = "TEXTO_SINTETICO"
SINTESIS_MAX_TOKENS: int = 256
SINTESIS_TEMPERATURE: float = 0.0
SINTESIS_RETRIES: int = 3
CHARS_POR_TOKEN: int = 4  # misma aproximación conservadora que src/ingest.py

CORPUS_RAW_DIR: Path = PROJECT_ROOT / "corpus" / "raw"
SJF_ENDPOINT: str = (
    "https://sjf2.scjn.gob.mx/services/sjftesismicroservice/api/public/tesis"
)
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "Legis/0.1 (corpus builder)"
)
TIMEOUT_SECONDS: int = 30
RETRIES_POR_PAGINA: int = 3
DELAY_SEGUNDOS: float = 0.5

# Palabras clave de materia laboral. Los rubros del SJF vienen en mayúsculas,
# así que comparamos contra el rubro sin acentos en MAYÚSCULAS.
PALABRAS_CLAVE_LABORAL: list[str] = [
    "TRABAJADOR", "TRABAJADORA", "TRABAJO", "LABORAL",
    "PATRON",  # sin acento — el campo viene normalizado por la SCJN sin tildes
    "OBRERO", "EMPLEADO", "EMPLEADA",
    "DESPIDO", "RESCISION", "REINSTALACION",
    "INDEMNIZACION CONSTITUCIONAL", "SALARIOS VENCIDOS",
    "JORNADA", "SALARIO MINIMO", "AGUINALDO", "VACACIONES",
    "PRIMA DE ANTIGUEDAD", "PRIMA DOMINICAL", "PRIMA VACACIONAL",
    "JUNTA DE CONCILIACION", "TRIBUNAL LABORAL",
    "SINDICATO", "HUELGA", "CONTRATO COLECTIVO",
    "CONTRATO INDIVIDUAL DE TRABAJO", "RELACION LABORAL", "RELACION DE TRABAJO",
    "IMSS", "INFONAVIT", "ISSSTE",
    "CESANTIA EN EDAD AVANZADA", "JUBILACION", "PENSION POR INVALIDEZ",
    "PENSION DE VIUDEZ", "PENSION POR CESANTIA",
    "HORAS EXTRAS", "TIEMPO EXTRAORDINARIO",
    "HOSTIGAMIENTO LABORAL", "ACOSO LABORAL", "ACOSO SEXUAL",
    "TRABAJADORA EMBARAZADA", "MATERNIDAD", "FUERO DE EMBARAZO",
    "MATERIA LABORAL", "MATERIA DEL TRABAJO",
    "REPSE", "SUBCONTRATACION", "OUTSOURCING",
    "RIESGO DE TRABAJO", "ENFERMEDAD DE TRABAJO",
    "PARTICIPACION DE LOS TRABAJADORES EN LAS UTILIDADES", "PTU",
    "FONDO DE AHORRO",
]


@dataclass
class Tesis:
    """Subconjunto de campos relevantes de un documento del SJF."""

    ius: int
    id: str
    rubro: str
    clave_tesis: str
    localizacion: str
    epoca_abr: str
    sala: str
    fecha_publicacion: str
    instancia_abr: str
    fuente: str
    is_jurisprudencia: bool  # True si [J] en localizacion, False si [TA]/[T]


def _strip_accents(text: str) -> str:
    """Elimina acentos y diéresis para comparar contra palabras clave normalizadas."""
    if not text:
        return ""
    sustituciones = str.maketrans({
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ü": "U", "Ñ": "N",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
    })
    return text.translate(sustituciones)


def es_laboral(rubro: str) -> str | None:
    """
    Determina si un rubro pertenece a materia laboral por keyword matching.

    Returns:
        La primera keyword que matchó (para estadísticas) o None.
    """
    if not rubro:
        return None
    rubro_norm = _strip_accents(rubro).upper()
    for kw in PALABRAS_CLAVE_LABORAL:
        if kw in rubro_norm:
            return kw
    return None


def normalizar_clave(clave_tesis: str) -> str:
    """
    Convierte la clave SCJN a forma segura para filename.

    Ejemplo: "P./J. 80/2026 (12a.)" → "P-J-80-2026"
    """
    if not clave_tesis:
        return "sin-clave"
    # Drop la época parentetizada al final: "(12a.)"
    sin_epoca = re.sub(r"\s*\([^)]+\)\s*$", "", clave_tesis).strip()
    # Reemplaza . / espacios por -
    normalizado = re.sub(r"[\s./]+", "-", sin_epoca).strip("-")
    # Reemplaza cualquier otro carácter inseguro
    normalizado = re.sub(r"[^A-Za-z0-9\-]", "", normalizado)
    return normalizado or "sin-clave"


def normalizar_epoca(epoca_abr: str) -> str:
    """`"12a. Época"` → `"12a"`. Si no parsea, devuelve `"sin-epoca"`."""
    if not epoca_abr:
        return "sin-epoca"
    m = re.search(r"(\d+a)", epoca_abr)
    return m.group(1) if m else "sin-epoca"


def parsear_tesis(doc: dict) -> Tesis | None:
    """
    Construye un Tesis desde un documento del API. Devuelve None si faltan
    campos críticos (rubro, claveTesis).
    """
    rubro = (doc.get("rubro") or "").strip()
    clave = (doc.get("claveTesis") or "").strip()
    if not rubro or not clave:
        return None

    localizacion = (doc.get("localizacion") or "").strip()
    is_juris = localizacion.startswith("[J]")

    return Tesis(
        ius=int(doc.get("ius") or 0),
        id=str(doc.get("id") or ""),
        rubro=rubro,
        clave_tesis=clave,
        localizacion=localizacion,
        epoca_abr=(doc.get("epocaAbr") or "").strip(),
        sala=(doc.get("sala") or "").strip(),
        fecha_publicacion=(doc.get("fechaPublicacion") or "").strip(),
        instancia_abr=(doc.get("instanciaAbr") or "").strip(),
        fuente=(doc.get("fuente") or "Semanario Judicial de la Federación").strip(),
        is_jurisprudencia=is_juris,
    )


def construir_txt(t: Tesis, texto_sintetico: str | None = None) -> str:
    """
    Genera el contenido del .txt con metadatos verificables.

    Args:
        t: Tesis parseada del API del SJF.
        texto_sintetico: Síntesis del rubro generada por Haiku (solo para J).
            Si es None, el bloque TEXTO_SINTETICO se omite — el archivo queda
            idéntico al formato previo al enriquecimiento.

    Returns:
        Contenido completo del .txt listo para escribir en corpus/raw/.
    """
    tipo = (
        "Jurisprudencia OBLIGATORIA"
        if t.is_jurisprudencia
        else "Tesis aislada (orientadora, NO obligatoria)"
    )
    bloque_sintesis = ""
    if texto_sintetico:
        bloque_sintesis = f"{ETIQUETA_SINTESIS}: {texto_sintetico.strip()}\n\n"

    return (
        f"RUBRO: {t.rubro}\n"
        f"\n"
        f"CLAVE DE TESIS: {t.clave_tesis}\n"
        f"LOCALIZACIÓN: {t.localizacion}\n"
        f"INSTANCIA: {t.sala} ({t.instancia_abr})\n"
        f"ÉPOCA: {t.epoca_abr}\n"
        f"FECHA DE PUBLICACIÓN: {t.fecha_publicacion}\n"
        f"TIPO: {tipo}\n"
        f"REGISTRO DIGITAL (IUS): {t.ius}\n"
        f"FUENTE: {t.fuente}\n"
        f"\n"
        f"{bloque_sintesis}"
        f"{_construir_nota(t.ius)}"
    )


def _construir_nota(ius: int | str) -> str:
    """Bloque NOTA estándar, compartido por la escritura nueva y el backfill."""
    return (
        f"NOTA: Esta tesis fue extraída del API público del Semanario Judicial\n"
        f"de la Federación (sjf2.scjn.gob.mx). El cuerpo completo de la tesis\n"
        f"NO está disponible en el listing endpoint — solo el rubro y los\n"
        f"metadatos. El bloque {ETIQUETA_SINTESIS} (si está presente) es una\n"
        f"síntesis del rubro generada con IA para mejorar la recuperación; NO es\n"
        f"el texto oficial de la tesis. Para verificar el texto íntegro consultar:\n"
        f"  https://sjf2.scjn.gob.mx/detalle/tesis/{ius}\n"
    )


# === Cliente Anthropic (lazy) y generación de la síntesis ===

_anthropic_client: anthropic.Anthropic | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    """Crea (una sola vez) el cliente Anthropic usando la API key de .env."""
    global _anthropic_client
    if _anthropic_client is None:
        config = load_config()
        _anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _anthropic_client


def generar_texto_sintetico(rubro: str, modelo: str) -> str | None:
    """
    Genera la síntesis del rubro con Claude Haiku (3-4 líneas).

    Aplica el prompt de SINTESIS_TESIS_SYSTEM_PROMPT, que prohíbe inventar
    artículos: el punto 3 cita un artículo SOLO si aparece textualmente en el
    rubro. El system prompt usa prompt caching ephemeral, así que las 1000+
    llamadas consecutivas amortizan su costo de entrada.

    Degrada con elegancia: si la llamada falla tras los reintentos, devuelve
    None y registra un warning — la tesis se guarda sin síntesis en vez de
    perderse.

    Args:
        rubro: Rubro de la tesis (única fuente permitida para la síntesis).
        modelo: ID del modelo Anthropic (config.llm_model, Haiku).

    Returns:
        Párrafo de síntesis, o None si la generación falló o vino vacía.
    """
    if not rubro or not rubro.strip():
        return None

    client = _get_anthropic_client()
    ultimo_error: Exception | None = None
    for intento in range(1, SINTESIS_RETRIES + 1):
        try:
            response = client.messages.create(
                model=modelo,
                max_tokens=SINTESIS_MAX_TOKENS,
                temperature=SINTESIS_TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": SINTESIS_TESIS_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[{"role": "user", "content": f"RUBRO:\n{rubro.strip()}"}],
            )
        except anthropic.APIError as exc:
            ultimo_error = exc
            espera = 2 ** intento
            logger.warning(
                "  Síntesis intento %d/%d falló: %s — reintentando en %ds",
                intento, SINTESIS_RETRIES, exc, espera,
            )
            time.sleep(espera)
            continue

        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                return block.text.strip()
        logger.warning("  Síntesis devolvió respuesta vacía para el rubro.")
        return None

    logger.error(
        "  Síntesis falló tras %d intentos (%s). Se guarda la tesis sin síntesis.",
        SINTESIS_RETRIES, ultimo_error,
    )
    return None


def nombre_archivo(t: Tesis) -> str:
    """Construye el nombre del archivo siguiendo skills/corpus-ingest.md §2."""
    prefijo = "SCJN_J" if t.is_jurisprudencia else "SCJN_T"
    clave = normalizar_clave(t.clave_tesis)
    epoca = normalizar_epoca(t.epoca_abr)
    return f"{prefijo}_{clave}_{epoca}.txt"


def pedir_pagina(
    session: requests.Session,
    page: int,
    page_size: int,
) -> dict | None:
    """
    Pide una página del listado con reintentos.

    Returns:
        El JSON parseado, o None si todos los reintentos fallaron.
    """
    params = {"page": page, "size": page_size}
    for intento in range(1, RETRIES_POR_PAGINA + 1):
        try:
            resp = session.post(
                SJF_ENDPOINT,
                json={},
                params=params,
                timeout=TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            espera = 2 ** intento
            logger.warning(
                "  Página %d intento %d/%d falló: %s — reintentando en %ds",
                page, intento, RETRIES_POR_PAGINA, exc, espera,
            )
            time.sleep(espera)
    logger.error("  Página %d: falló tras %d intentos", page, RETRIES_POR_PAGINA)
    return None


# === Backfill: enriquecer las tesis J ya descargadas ===

_RUBRO_RE = re.compile(r"^RUBRO:\s*(.+)$", re.MULTILINE)
_IUS_RE = re.compile(r"^REGISTRO DIGITAL \(IUS\):\s*(\d+)", re.MULTILINE)


def _extraer_rubro(contenido: str) -> str:
    """Devuelve el rubro de un .txt ya escrito (línea `RUBRO: ...`)."""
    m = _RUBRO_RE.search(contenido)
    return m.group(1).strip() if m else ""


def _insertar_sintesis(contenido: str, texto_sintetico: str) -> str:
    """
    Inserta el bloque TEXTO_SINTETICO antes de la NOTA en un .txt existente.

    Reconstruye además la NOTA con el helper compartido para que el archivo
    backfilleado quede idéntico en formato a uno escrito ya enriquecido. Si
    no encuentra la NOTA (formato inesperado), antepone la síntesis al final.

    Args:
        contenido: Texto actual del .txt.
        texto_sintetico: Síntesis a insertar.

    Returns:
        Nuevo contenido del .txt.
    """
    bloque = f"{ETIQUETA_SINTESIS}: {texto_sintetico.strip()}\n\n"
    m_ius = _IUS_RE.search(contenido)
    ius = m_ius.group(1) if m_ius else ""
    idx = contenido.find("NOTA:")
    if idx == -1:
        return contenido.rstrip() + "\n\n" + bloque.rstrip() + "\n"
    cabecera = contenido[:idx]
    return cabecera + bloque + _construir_nota(ius)


def enriquecer_existentes() -> int:
    """
    Backfill: añade TEXTO_SINTETICO a las tesis J ya descargadas en corpus/raw/.

    Idempotente: salta los archivos que ya contienen el bloque. No re-descarga
    del SJF — opera sobre los .txt locales. Las tesis aisladas (SCJN_T_*) se
    ignoran por diseño (solo se enriquece jurisprudencia obligatoria).

    Returns:
        Código de salida (0 si terminó, aunque algunas síntesis fallen).
    """
    modelo = load_config().llm_model
    archivos = sorted(CORPUS_RAW_DIR.glob("SCJN_J_*.txt"))
    if not archivos:
        logger.error("No se encontraron archivos SCJN_J_*.txt en %s", CORPUS_RAW_DIR)
        return 1

    logger.info("Backfill de síntesis sobre %d jurisprudencias (J)", len(archivos))
    logger.info("Modelo: %s | etiqueta: %s", modelo, ETIQUETA_SINTESIS)

    enriquecidas = 0
    ya_tenian = 0
    fallidas = 0
    sin_rubro = 0
    for i, ruta in enumerate(archivos, start=1):
        contenido = ruta.read_text(encoding="utf-8")
        if ETIQUETA_SINTESIS in contenido:
            ya_tenian += 1
            continue
        rubro = _extraer_rubro(contenido)
        if not rubro:
            logger.warning("  [%d] Sin rubro parseable: %s", i, ruta.name)
            sin_rubro += 1
            continue

        sintesis = generar_texto_sintetico(rubro, modelo)
        if sintesis is None:
            fallidas += 1
            continue

        ruta.write_text(_insertar_sintesis(contenido, sintesis), encoding="utf-8")
        enriquecidas += 1
        if i % 50 == 0 or enriquecidas <= 3:
            logger.info(
                "  [%d/%d] enriquecidas: %d | ya tenían: %d | fallidas: %d",
                i, len(archivos), enriquecidas, ya_tenian, fallidas,
            )

    logger.info("=" * 60)
    logger.info("Backfill completo")
    logger.info("  Enriquecidas nuevas:  %d", enriquecidas)
    logger.info("  Ya tenían síntesis:   %d", ya_tenian)
    logger.info("  Sin rubro:            %d", sin_rubro)
    logger.info("  Fallidas (API):       %d", fallidas)
    logger.info("")
    if enriquecidas:
        logger.info("Siguiente paso: ejecuta `python src/ingest.py` para reindexar.")
    return 0


def estimar_costo() -> int:
    """
    Estima el costo de enriquecer las tesis J pendientes (sin gastar API).

    Mide los rubros reales de los SCJN_J_*.txt no enriquecidos y aproxima los
    tokens de entrada (system prompt + rubro) y de salida. Imprime un rango de
    costo. NO llama a la API.

    Returns:
        Código de salida 0.
    """
    archivos = sorted(CORPUS_RAW_DIR.glob("SCJN_J_*.txt"))
    pendientes = []
    for ruta in archivos:
        contenido = ruta.read_text(encoding="utf-8")
        if ETIQUETA_SINTESIS in contenido:
            continue
        rubro = _extraer_rubro(contenido)
        if rubro:
            pendientes.append(rubro)

    n = len(pendientes)
    if n == 0:
        logger.info("No hay jurisprudencias J pendientes de enriquecer.")
        return 0

    sys_tokens = len(SINTESIS_TESIS_SYSTEM_PROMPT) // CHARS_POR_TOKEN
    rubro_tokens = [len(r) // CHARS_POR_TOKEN for r in pendientes]
    rubro_prom = sum(rubro_tokens) / n
    # Entrada por llamada: system prompt + framing ("RUBRO:\n") + rubro.
    in_por_llamada = sys_tokens + 4 + rubro_prom
    in_total = in_por_llamada * n
    # Salida: párrafo de 3-4 líneas ≈ 180 tokens (cap en SINTESIS_MAX_TOKENS).
    out_por_llamada = 180
    out_total = out_por_llamada * n

    # Tarifas Haiku 4.5 (ajusta si cambian). El usuario citó $0.80/M entrada.
    precio_in = 0.80 / 1_000_000
    precio_out = 4.00 / 1_000_000
    costo_in = in_total * precio_in
    costo_out = out_total * precio_out

    logger.info("=" * 60)
    logger.info("Estimación de costo — backfill de síntesis (tesis J)")
    logger.info("=" * 60)
    logger.info("Jurisprudencias J pendientes:   %d", n)
    logger.info("Tokens system prompt:           ~%d (cacheable)", sys_tokens)
    logger.info("Rubro promedio:                 ~%.0f tokens", rubro_prom)
    logger.info("Entrada total estimada:         ~%s tokens", f"{in_total:,.0f}")
    logger.info("Salida total estimada:          ~%s tokens (~%d/llamada)",
                f"{out_total:,.0f}", out_por_llamada)
    logger.info("-" * 60)
    logger.info("Costo entrada (@ $0.80/M):      ~$%.2f USD", costo_in)
    logger.info("Costo salida  (@ $4.00/M):      ~$%.2f USD", costo_out)
    logger.info("COSTO TOTAL estimado:           ~$%.2f USD", costo_in + costo_out)
    logger.info("(con prompt caching del system prompt, la entrada baja "
                "significativamente)")
    logger.info("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enumera tesis del SJF y guarda las laborales como .txt"
    )
    parser.add_argument(
        "--max-pages", type=int, default=500,
        help="Páginas máximas a recorrer (default: 500 = 50,000 tesis si size=100)",
    )
    parser.add_argument(
        "--page-size", type=int, default=100,
        help="Tesis por página (default: 100, máximo soportado por el API)",
    )
    parser.add_argument(
        "--start-page", type=int, default=0,
        help="Página inicial (útil para reanudar tras interrupción)",
    )
    parser.add_argument(
        "--enrich-existing", action="store_true",
        help="No descarga: enriquece con TEXTO_SINTETICO las tesis J ya guardadas",
    )
    parser.add_argument(
        "--estimar-costo", action="store_true",
        help="No descarga ni gasta API: estima el costo del backfill de síntesis",
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Descarga sin generar síntesis (comportamiento previo, sin costo de LLM)",
    )
    args = parser.parse_args()

    CORPUS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Modos que no descargan del SJF.
    if args.estimar_costo:
        return estimar_costo()
    if args.enrich_existing:
        return enriquecer_existentes()

    enriquecer = not args.no_enrich
    modelo_sintesis = load_config().llm_model if enriquecer else ""
    tesis_enriquecidas = 0

    logger.info("=" * 60)
    logger.info("Listado de tesis SCJN — filtro: materia laboral")
    logger.info("=" * 60)
    logger.info("Endpoint:    %s", SJF_ENDPOINT)
    logger.info("Páginas:     %d → %d (size=%d)",
                args.start_page, args.start_page + args.max_pages, args.page_size)
    logger.info("Destino:     %s", CORPUS_RAW_DIR)
    logger.info("Keywords:    %d palabras de materia laboral",
                len(PALABRAS_CLAVE_LABORAL))
    logger.info("=" * 60)

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    # Métricas
    tesis_inspeccionadas = 0
    tesis_laborales = 0
    tesis_skipped_metadatos = 0
    tesis_skipped_duplicadas = 0
    juris_vs_tesis = Counter()
    matches_por_keyword: Counter = Counter()
    paginas_fallidas: list[int] = []
    nombres_escritos: set[str] = set()

    primera_pagina = pedir_pagina(session, args.start_page, args.page_size)
    if primera_pagina is None:
        logger.error("No se pudo obtener la primera página. Abortando.")
        return 1
    total_disponible = int(primera_pagina.get("total", 0))
    logger.info("Total de tesis en el SJF: %s", f"{total_disponible:,}")
    logger.info("")

    for page_num in range(args.start_page, args.start_page + args.max_pages):
        if page_num == args.start_page:
            payload = primera_pagina
        else:
            payload = pedir_pagina(session, page_num, args.page_size)
            if payload is None:
                paginas_fallidas.append(page_num)
                continue
            time.sleep(DELAY_SEGUNDOS)

        documentos = payload.get("documents") or []
        if not documentos:
            logger.info("Página %d sin documentos — fin del listado", page_num)
            break

        matches_pagina = 0
        for doc in documentos:
            tesis_inspeccionadas += 1
            t = parsear_tesis(doc)
            if t is None:
                tesis_skipped_metadatos += 1
                continue

            kw = es_laboral(t.rubro)
            if kw is None:
                continue

            tesis_laborales += 1
            matches_pagina += 1
            matches_por_keyword[kw] += 1
            juris_vs_tesis["J" if t.is_jurisprudencia else "T"] += 1

            archivo = nombre_archivo(t)
            if archivo in nombres_escritos:
                tesis_skipped_duplicadas += 1
                continue
            nombres_escritos.add(archivo)

            # Enriquecer solo jurisprudencia obligatoria (J), si está habilitado.
            texto_sintetico: str | None = None
            if enriquecer and t.is_jurisprudencia:
                texto_sintetico = generar_texto_sintetico(t.rubro, modelo_sintesis)
                if texto_sintetico:
                    tesis_enriquecidas += 1

            destino = CORPUS_RAW_DIR / archivo
            destino.write_text(
                construir_txt(t, texto_sintetico=texto_sintetico),
                encoding="utf-8",
            )

        if page_num % 25 == 0 or matches_pagina > 0:
            logger.info(
                "Página %4d/%4d  | inspeccionadas: %s  laborales: %s  (+%d en esta página)",
                page_num + 1, args.start_page + args.max_pages,
                f"{tesis_inspeccionadas:,}", f"{tesis_laborales:,}", matches_pagina,
            )

    # === Resumen ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("Resumen")
    logger.info("=" * 60)
    logger.info("Tesis inspeccionadas:       %s", f"{tesis_inspeccionadas:,}")
    logger.info("Tesis laborales escritas:   %s", f"{tesis_laborales:,}")
    logger.info("  Jurisprudencia (J):       %s", f"{juris_vs_tesis['J']:,}")
    logger.info("  Tesis aislada  (T):       %s", f"{juris_vs_tesis['T']:,}")
    if enriquecer:
        logger.info("  J enriquecidas (síntesis):%s", f"{tesis_enriquecidas:,}")
    logger.info("Skipped (metadata faltante):%s", f"{tesis_skipped_metadatos:,}")
    logger.info("Skipped (clave duplicada):  %s", f"{tesis_skipped_duplicadas:,}")
    logger.info("Páginas con error:          %d", len(paginas_fallidas))
    if paginas_fallidas:
        logger.warning("  Páginas fallidas: %s",
                       ", ".join(str(p) for p in paginas_fallidas[:20]))

    if matches_por_keyword:
        logger.info("")
        logger.info("Top 15 keywords con más matches:")
        for kw, n in matches_por_keyword.most_common(15):
            logger.info("  %-45s  %s", kw, f"{n:,}")

    logger.info("")
    logger.info("Siguiente paso: ejecuta `python src/ingest.py` para indexar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
