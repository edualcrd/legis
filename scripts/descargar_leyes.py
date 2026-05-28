"""
Descarga el corpus base de leyes federales y criterios IMSS.

Fuentes (todas son PDFs digitales estables, sin auth, sin JS):
    - Cámara de Diputados (diputados.gob.mx):
        * Leyes y códigos (bajo /LeyesBiblio/pdf/): LFT, CPEUM, LSS, LFTSE,
          Ley del INFONAVIT, Ley del ISSSTE, Ley Federal para Prevenir y
          Eliminar la Discriminación, Ley General para la Inclusión de las
          Personas con Discapacidad, Ley General de Acceso de las Mujeres a
          una Vida Libre de Violencia, Código Civil Federal y Ley Federal de
          Procedimiento Administrativo.
        * Reglamentos (rutas distintas, /LeyesBiblio/regla/ y /regley/):
          Reglamento Federal de Seguridad y Salud en el Trabajo y Reglamento
          de la LSS en materia de afiliación (RACERF).
    - IMSS (imss.gob.mx): criterios normativos descubiertos dinámicamente
      desde la página índice imss.gob.mx/patrones/criterios-normativos

Los archivos se guardan en corpus/raw/ con la convención de nombres de
skills/corpus-ingest.md §2:
    LFT_completa_{YYYY-MM-DD}.pdf
    CPEUM_completa_{YYYY-MM-DD}.pdf
    IMSS_Criterio_{id_normalizado}_{YYYY-MM-DD}.pdf

La fecha usada es la fecha de descarga (no la de última reforma) porque
diputados.gob.mx no expone esa metadata en HEAD; el campo `date` real se
actualiza al reindexar si el contenido cambia.

Uso:
    python scripts/descargar_leyes.py
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger  # noqa: E402

logger = get_logger("descargar_leyes")

CORPUS_RAW_DIR: Path = PROJECT_ROOT / "corpus" / "raw"
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "Legis/0.1 (corpus builder)"
)
TIMEOUT_SECONDS: int = 60
CHUNK_BYTES: int = 64 * 1024

# Leyes federales y códigos — URLs verificadas contra diputados.gob.mx
# (las 4 base el 2026-05-17; las añadidas el 2026-05-28, todas 200 application/pdf).
# OJO: la 1ª columna (source_code) es el código INTERNO de Legis y debe coincidir
# con SOURCE_TO_TYPE de src/ingest.py y SOURCE_LABELS de src/rag.py. El nombre del
# PDF en diputados puede diferir del código (p. ej. INFONAVIT vive en LIFNVT.pdf).
LEYES_FEDERALES: list[tuple[str, str, str]] = [
    # (source_code, output_basename, url)
    ("LFT",        "LFT_completa",        "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFT.pdf"),
    ("CPEUM",      "CPEUM_completa",      "https://www.diputados.gob.mx/LeyesBiblio/pdf/CPEUM.pdf"),
    ("LSS",        "LSS_completa",        "https://www.diputados.gob.mx/LeyesBiblio/pdf/LSS.pdf"),
    ("LFTSE",      "LFTSE_completa",      "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFTSE.pdf"),
    ("LINFONAVIT", "LINFONAVIT_completa", "https://www.diputados.gob.mx/LeyesBiblio/pdf/LIFNVT.pdf"),
    ("LISSSTE",    "LISSSTE_completa",    "https://www.diputados.gob.mx/LeyesBiblio/pdf/LISSSTE.pdf"),
    ("LFPED",      "LFPED_completa",      "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFPED.pdf"),
    ("LGIPD",      "LGIPD_completa",      "https://www.diputados.gob.mx/LeyesBiblio/pdf/LGIPD.pdf"),
    ("LGAMVLV",    "LGAMVLV_completa",    "https://www.diputados.gob.mx/LeyesBiblio/pdf/LGAMVLV.pdf"),
    ("CCF",        "CCF_completo",        "https://www.diputados.gob.mx/LeyesBiblio/pdf/CCF.pdf"),
    ("LFPA",       "LFPA_completa",       "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFPA.pdf"),
    # Opcional, fuera del foco laboral — URL verificada (200 OK) pero NO se
    # descarga por defecto. Para activarla: descomenta esta línea y añade
    # "LGSNA" a SOURCE_TO_TYPE (src/ingest.py) y SOURCE_LABELS (src/rag.py).
    # ("LGSNA",    "LGSNA_completa",      "https://www.diputados.gob.mx/LeyesBiblio/pdf/LGSNA_200521.pdf"),
]

# Reglamentos laborales — en diputados viven en rutas DISTINTAS de las leyes
# (no bajo /pdf/): el RFSST en /LeyesBiblio/regla/ y el RACERF en /regley/.
# URLs verificadas el 2026-05-28 (200 application/pdf).
REGLAMENTOS: list[tuple[str, str, str]] = [
    # (source_code, output_basename, url)
    ("RFSST",  "RFSST_completo",  "https://www.diputados.gob.mx/LeyesBiblio/regla/n152.pdf"),
    ("RACERF", "RACERF_completo", "https://www.diputados.gob.mx/LeyesBiblio/regley/Reg_LSS_MACERF.pdf"),
]

IMSS_INDEX_URL: str = "https://www.imss.gob.mx/patrones/criterios-normativos"
IMSS_PDF_REGEX = re.compile(
    r'href="([^"]+(?:criteriosnormativos|crt-)[^"]+\.pdf)"',
    re.IGNORECASE,
)


@dataclass
class DownloadResult:
    source: str
    filename: str
    bytes_written: int
    ok: bool
    error: str = ""


def _today_iso() -> str:
    """Fecha ISO en UTC para el nombre de archivo."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _imss_filename_from_url(url: str, fecha_iso: str) -> str:
    """
    Convierte la URL del PDF IMSS al nombre normalizado del corpus.

    Ejemplo:
        .../Criterio-02-2023-N-SBC-LSS-27-IV.pdf
        → IMSS_Criterio_02-2023-N-SBC-LSS-27-IV_2026-05-17.pdf
    """
    raw_name = Path(url).stem
    raw_name = re.sub(r"^(Criterio[-_])", "", raw_name, flags=re.IGNORECASE)
    raw_name = re.sub(r"^(ACUERDO[-_])", "Acuerdo-", raw_name, flags=re.IGNORECASE)
    safe_id = re.sub(r"[^A-Za-z0-9\-]", "-", raw_name).strip("-")
    if not safe_id:
        safe_id = "doc"
    return f"IMSS_Criterio_{safe_id}_{fecha_iso}.pdf"


def descargar_pdf(
    url: str,
    destino: Path,
    session: requests.Session,
) -> DownloadResult:
    """
    Descarga un PDF a `destino`, con streaming para no cargar todo en memoria.

    Args:
        url: URL del PDF.
        destino: Path absoluto donde guardar.
        session: Sesión de requests con headers preconfigurados.

    Returns:
        DownloadResult con métricas y estado.
    """
    nombre = destino.name
    logger.info("  Descargando %s ...", nombre)

    try:
        resp = session.get(url, stream=True, timeout=TIMEOUT_SECONDS, allow_redirects=True)
        resp.raise_for_status()

        ctype = resp.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower():
            return DownloadResult(
                source=nombre, filename=nombre, bytes_written=0, ok=False,
                error=f"Content-Type inesperado: {ctype}",
            )

        bytes_written = 0
        with open(destino, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        mb = bytes_written / (1024 * 1024)
        logger.info("    OK — %.2f MB → %s", mb, destino.relative_to(PROJECT_ROOT))
        return DownloadResult(
            source=nombre, filename=nombre, bytes_written=bytes_written, ok=True,
        )

    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        return DownloadResult(
            source=nombre, filename=nombre, bytes_written=0, ok=False,
            error=f"HTTP {code}: {exc}",
        )
    except requests.RequestException as exc:
        return DownloadResult(
            source=nombre, filename=nombre, bytes_written=0, ok=False,
            error=f"Error de red: {exc}",
        )


def descubrir_pdfs_imss(session: requests.Session) -> list[str]:
    """
    Lee la página índice de criterios normativos del IMSS y extrae las URLs
    de los PDFs disponibles. Convierte rutas relativas a absolutas.

    Returns:
        Lista de URLs absolutas a PDFs de criterios IMSS.
    """
    logger.info("Descubriendo PDFs en el índice de criterios IMSS...")
    try:
        resp = session.get(IMSS_INDEX_URL, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("  No se pudo cargar el índice IMSS: %s", exc)
        return []

    rutas = IMSS_PDF_REGEX.findall(resp.text)
    urls_absolutas: list[str] = []
    for ruta in dict.fromkeys(rutas):  # preserva orden y deduplica
        if ruta.startswith("http"):
            urls_absolutas.append(ruta)
        else:
            urls_absolutas.append("https://www.imss.gob.mx" + ruta)

    logger.info("  Encontrados %d PDFs de criterios IMSS", len(urls_absolutas))
    return urls_absolutas


def main() -> int:
    CORPUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    fecha_iso = _today_iso()
    logger.info("=" * 60)
    logger.info("Descarga del corpus base de Legis — fecha de descarga %s", fecha_iso)
    logger.info("Destino: %s", CORPUS_RAW_DIR)
    logger.info("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})

    resultados: list[DownloadResult] = []

    # === Leyes federales y códigos ===
    logger.info("")
    logger.info("[1/3] Leyes federales y códigos (diputados.gob.mx)")
    antes = len(resultados)
    for i, (source, basename, url) in enumerate(LEYES_FEDERALES, start=1):
        logger.info(" %d/%d %s", i, len(LEYES_FEDERALES), source)
        destino = CORPUS_RAW_DIR / f"{basename}_{fecha_iso}.pdf"
        resultados.append(descargar_pdf(url, destino, session))
        time.sleep(0.5)  # cortesía con el servidor
    ok_leyes = sum(1 for r in resultados[antes:] if r.ok)
    logger.info("  → %d/%d leyes/códigos descargados", ok_leyes, len(LEYES_FEDERALES))

    # === Reglamentos ===
    logger.info("")
    logger.info("[2/3] Reglamentos laborales (diputados.gob.mx)")
    antes = len(resultados)
    for i, (source, basename, url) in enumerate(REGLAMENTOS, start=1):
        logger.info(" %d/%d %s", i, len(REGLAMENTOS), source)
        destino = CORPUS_RAW_DIR / f"{basename}_{fecha_iso}.pdf"
        resultados.append(descargar_pdf(url, destino, session))
        time.sleep(0.5)  # cortesía con el servidor
    ok_regl = sum(1 for r in resultados[antes:] if r.ok)
    logger.info("  → %d/%d reglamentos descargados", ok_regl, len(REGLAMENTOS))

    # === Criterios IMSS ===
    logger.info("")
    logger.info("[3/3] Criterios normativos IMSS (imss.gob.mx)")
    urls_imss = descubrir_pdfs_imss(session)
    for i, url in enumerate(urls_imss, start=1):
        logger.info(" %d/%d", i, len(urls_imss))
        nombre = _imss_filename_from_url(url, fecha_iso)
        destino = CORPUS_RAW_DIR / nombre
        resultados.append(descargar_pdf(url, destino, session))
        time.sleep(0.5)

    # === Resumen ===
    exitosos = [r for r in resultados if r.ok]
    fallidos = [r for r in resultados if not r.ok]
    total_mb = sum(r.bytes_written for r in exitosos) / (1024 * 1024)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Resumen de descarga")
    logger.info("=" * 60)
    logger.info("Total intentados:  %d", len(resultados))
    logger.info("Exitosos:          %d  (%.2f MB)", len(exitosos), total_mb)
    logger.info("Fallidos:          %d", len(fallidos))

    if fallidos:
        logger.warning("Archivos con error:")
        for r in fallidos:
            logger.warning("  - %s: %s", r.source, r.error)

    logger.info("")
    logger.info("Siguiente paso: ejecuta `python src/ingest.py` para indexar.")
    return 0 if exitosos else 1


if __name__ == "__main__":
    sys.exit(main())
