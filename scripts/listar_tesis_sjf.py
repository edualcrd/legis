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

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger  # noqa: E402

logger = get_logger("listar_tesis_sjf")

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


def construir_txt(t: Tesis) -> str:
    """Genera el contenido del .txt con metadatos verificables."""
    tipo = (
        "Jurisprudencia OBLIGATORIA"
        if t.is_jurisprudencia
        else "Tesis aislada (orientadora, NO obligatoria)"
    )
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
        f"NOTA: Esta tesis fue extraída del API público del Semanario Judicial\n"
        f"de la Federación (sjf2.scjn.gob.mx). El cuerpo completo de la tesis\n"
        f"NO está disponible en el listing endpoint — solo el rubro y los\n"
        f"metadatos. Para verificar el texto íntegro consultar:\n"
        f"  https://sjf2.scjn.gob.mx/detalle/tesis/{t.ius}\n"
    )


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
    args = parser.parse_args()

    CORPUS_RAW_DIR.mkdir(parents=True, exist_ok=True)

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

            destino = CORPUS_RAW_DIR / archivo
            destino.write_text(construir_txt(t), encoding="utf-8")

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
