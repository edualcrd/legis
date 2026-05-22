"""
Indexa el corpus legal (PDFs en corpus/raw/) en ChromaDB.

Pipeline:
    PDF (PyMuPDF) → chunking respetando límites de artículo
        → Voyage voyage-3 (embeddings vía API) → ChromaDB persistente

Aplica las reglas de:
    - skills/legal-rag.md §6 (chunking, nunca cortar artículo)
    - skills/corpus-ingest.md §2 (convención de nombres)
    - skills/corpus-ingest.md §3 (metadatos obligatorios por chunk)

El reranker (Voyage rerank-2.5) NO se usa aquí — es un componente de query
time. Se aplica en src/rag.py al reordenar los top-K de ChromaDB antes de
pasar al LLM.

IMPORTANTE: los embeddings deben generarse con el MISMO modelo que usa
src/rag.py al consultar (config.embed_model). Si cambias de modelo de
embeddings hay que RE-INDEXAR todo el corpus: los vectores viejos no son
comparables con los nuevos.

Uso:
    python src/ingest.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import fitz  # PyMuPDF
import voyageai

from src.utils import get_logger, load_config

logger = get_logger(__name__)

# === Configuración de chunking (skills/legal-rag.md §6) ===
# Tamaños en tokens. Como bge-m3 acepta hasta 8192 tokens, estos son holgados.
CHUNK_PROFILES: dict[str, dict[str, int]] = {
    "ley_federal":    {"size": 512,  "overlap": 50},
    "constitucion":   {"size": 512,  "overlap": 50},
    "jurisprudencia": {"size": 768,  "overlap": 100},
    "tesis":          {"size": 768,  "overlap": 100},
    "criterio_imss":  {"size": 512,  "overlap": 75},
    "laudo":          {"size": 1024, "overlap": 150},
    "default":        {"size": 512,  "overlap": 50},
}

# Aproximación carácter→token para el tokenizador XLM-R de bge-m3 en español.
# Es conservadora: prefiere chunks más pequeños que el límite teórico.
CHARS_PER_TOKEN: int = 4

# Mapeo de fuente del nombre de archivo al tipo de documento.
# SCJN se maneja aparte porque depende de J (jurisprudencia) vs T (tesis).
SOURCE_TO_TYPE: dict[str, str] = {
    "LFT": "ley_federal",
    "LSS": "ley_federal",
    "LFTSE": "ley_federal",
    "LINFONAVIT": "ley_federal",
    "CPEUM": "constitucion",
    "IMSS": "criterio_imss",
    "STPS": "laudo",
}

MANDATORY_SOURCES: set[str] = {"LFT", "LSS", "LFTSE", "LINFONAVIT", "CPEUM"}

# Detecta inicio de artículo: "Artículo 48.", "Artículo 1o.", "ARTÍCULO 123",
# con o sin tildes / variaciones de mayúscula y puntuación.
ARTICLE_BOUNDARY = re.compile(
    r"(?=^[\t ]*Art[íi]culo\s+\d+)",
    re.MULTILINE | re.IGNORECASE,
)
ARTICLE_NUMBER = re.compile(
    r"^[\t ]*Art[íi]culo\s+(\d+)",
    re.IGNORECASE,
)
DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

COLLECTION_NAME: str = "legis_corpus"
EMBED_BATCH_SIZE: int = 16


@dataclass
class Chunk:
    """Unidad indexable: texto + metadatos verificables + ID único."""

    text: str
    metadata: dict[str, Any]
    chunk_id: str = field(default="")


def parse_filename_metadata(filename: str) -> dict[str, Any]:
    """
    Extrae metadatos desde el nombre del archivo según skills/corpus-ingest.md §2.

    Convención: {FUENTE}_{IDENTIFICADOR}_{FECHA}.pdf
    Variante SCJN: SCJN_{J|T}_{identificador}_{epoca}.pdf

    Args:
        filename: Nombre del archivo (con o sin path), ej. "LFT_completa_2024-01-15.pdf"

    Returns:
        Dict con: source, doc_id, article, thesis_number, thesis_type,
        epoca, date, type, mandatory. Campos no aplicables van como "".
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    meta: dict[str, Any] = {
        "source": "",
        "doc_id": stem,
        "article": "",
        "thesis_number": "",
        "thesis_type": "",
        "epoca": "",
        "date": "",
        "type": "",
        "mandatory": False,
    }

    if not parts or not parts[0]:
        logger.warning("Nombre de archivo no parseable: %s", filename)
        return meta

    source = parts[0]
    meta["source"] = source

    if source == "SCJN":
        if len(parts) < 4:
            logger.warning(
                "PDF SCJN con nombre malformado (esperaba 4 partes): %s",
                filename,
            )
            return meta
        thesis_type = parts[1].upper()
        if thesis_type not in {"J", "T"}:
            logger.warning(
                "Tipo de tesis SCJN inválido en %s (esperaba J o T, vino '%s')",
                filename, thesis_type,
            )
            return meta
        meta["thesis_type"] = thesis_type
        meta["thesis_number"] = "_".join(parts[2:-1])
        meta["epoca"] = parts[-1]
        meta["type"] = "jurisprudencia" if thesis_type == "J" else "tesis"
        meta["mandatory"] = thesis_type == "J"

    elif source in SOURCE_TO_TYPE:
        meta["type"] = SOURCE_TO_TYPE[source]
        meta["mandatory"] = source in MANDATORY_SOURCES
        if len(parts) >= 2 and DATE_ISO.match(parts[-1]):
            meta["date"] = parts[-1]
        else:
            logger.warning(
                "Falta fecha ISO al final del nombre: %s — se indexa con date=\"\"",
                filename,
            )
    else:
        logger.warning(
            "Fuente desconocida '%s' en %s — se indexa con type=\"\"",
            source, filename,
        )

    return meta


def extract_text_with_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extrae texto del PDF página por página, conservando número de página.

    Args:
        pdf_path: Ruta al PDF.

    Returns:
        Lista de tuplas (page_num_1_indexed, page_text).

    Raises:
        RuntimeError: Si PyMuPDF no puede abrir el archivo.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo abrir el PDF {pdf_path.name}: {exc}"
        ) from exc

    try:
        return [(i, page.get_text("text")) for i, page in enumerate(doc, start=1)]
    finally:
        doc.close()


def extract_text_from_txt(txt_path: Path) -> list[tuple[int, str]]:
    """
    Lee un archivo de texto plano y lo devuelve como una sola "página".

    Usado para tesis SCJN guardadas desde el API del SJF como .txt
    (ver scripts/listar_tesis_sjf.py).

    Args:
        txt_path: Ruta al .txt.

    Returns:
        Lista con una tupla (1, contenido_completo).

    Raises:
        RuntimeError: Si el archivo no se puede leer como UTF-8.
    """
    try:
        text = txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"No se pudo decodificar el .txt {txt_path.name} como UTF-8: {exc}"
        ) from exc
    return [(1, text)]


def extract_document_pages(doc_path: Path) -> list[tuple[int, str]]:
    """
    Despacha la extracción según la extensión del archivo.

    Args:
        doc_path: Ruta al documento (.pdf o .txt soportados).

    Returns:
        Lista de (page_num, page_text).

    Raises:
        ValueError: Si la extensión no es soportada.
    """
    ext = doc_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_with_pages(doc_path)
    if ext == ".txt":
        return extract_text_from_txt(doc_path)
    raise ValueError(
        f"Extensión no soportada para {doc_path.name}: '{ext}'. "
        f"Se aceptan .pdf y .txt."
    )


def _detect_article_number(text_segment: str) -> str:
    """Devuelve el número de artículo si el segmento empieza con 'Artículo N'."""
    m = ARTICLE_NUMBER.match(text_segment.strip())
    return m.group(1) if m else ""


def _split_with_overlap(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Divide un texto largo (un solo artículo demasiado grande) con solape,
    cortando preferentemente en fin de oración (`.` o salto de línea).

    Args:
        text: Texto a dividir.
        chunk_size_chars: Tamaño objetivo por chunk en caracteres.
        overlap_chars: Caracteres de solape entre chunks consecutivos.

    Returns:
        Lista de subcadenas, cada una <= chunk_size_chars aprox.
    """
    if len(text) <= chunk_size_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Retrocede hasta el primer fin de oración o salto de línea para no
        # cortar a mitad de frase. Búsqueda acotada a los últimos ~100 chars.
        boundary = end
        lookback_limit = max(start + chunk_size_chars - 100, start + 1)
        for i in range(end, lookback_limit, -1):
            if i < len(text) and text[i] in ".\n":
                boundary = i + 1
                break

        chunks.append(text[start:boundary])
        next_start = boundary - overlap_chars
        # Garantía contra ciclos infinitos si overlap > avance neto.
        start = next_start if next_start > start else boundary

    return chunks


def chunk_by_articles(
    pages: list[tuple[int, str]],
    base_metadata: dict[str, Any],
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """
    Divide el texto del PDF en chunks que respetan los límites de artículo.

    Estrategia:
        1. Concatena todas las páginas manteniendo un mapa carácter→página.
        2. Divide por `Artículo N` usando lookahead (no consume el delimitador).
        3. Si un segmento de artículo cabe en chunk_size, se emite entero.
        4. Si excede el tamaño, se subdivide con solape respetando frases.

    Args:
        pages: Salida de `extract_text_with_pages`.
        base_metadata: Metadatos del documento (parse_filename_metadata).
        chunk_size_chars: Tamaño objetivo por chunk en caracteres.
        overlap_chars: Solape entre chunks dentro de un mismo artículo largo.

    Returns:
        Lista de Chunk listos para indexar.
    """
    full_text_parts: list[str] = []
    char_to_page: list[int] = []
    for page_num, page_text in pages:
        full_text_parts.append(page_text)
        full_text_parts.append("\n")
        char_to_page.extend([page_num] * (len(page_text) + 1))
    full_text = "".join(full_text_parts)

    segments = ARTICLE_BOUNDARY.split(full_text)
    if not segments:
        segments = [full_text]

    chunks: list[Chunk] = []
    doc_id = base_metadata["doc_id"]
    offset = 0
    chunk_idx = 0

    for segment in segments:
        seg_len = len(segment)
        if not segment.strip():
            offset += seg_len
            continue

        page = char_to_page[offset] if offset < len(char_to_page) else 1
        article_num = _detect_article_number(segment)

        if seg_len <= chunk_size_chars:
            pieces = [segment]
        else:
            pieces = _split_with_overlap(segment, chunk_size_chars, overlap_chars)

        for piece in pieces:
            piece_clean = piece.strip()
            if not piece_clean:
                continue
            meta = dict(base_metadata)
            if article_num:
                meta["article"] = article_num
            meta["page"] = page
            chunks.append(Chunk(
                text=piece_clean,
                metadata=meta,
                chunk_id=f"{doc_id}_chunk_{chunk_idx:04d}",
            ))
            chunk_idx += 1

        offset += seg_len

    return chunks


def index_corpus(
    raw_dir: Path,
    persist_dir: Path,
    embed_model_name: str = "voyage-3",
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Procesa todos los PDFs de raw_dir y los indexa en ChromaDB en persist_dir.

    Usa `upsert` en ChromaDB, así que es seguro re-ejecutar para reindexar
    cambios sin tener que borrar la base manualmente.

    Args:
        raw_dir: Directorio con los PDFs originales del corpus legal.
        persist_dir: Directorio donde ChromaDB persiste el índice.
        embed_model_name: Modelo de embeddings de Voyage (default voyage-3).
        collection_name: Nombre de la colección en ChromaDB.

    Returns:
        Número total de chunks indexados.

    Raises:
        FileNotFoundError: Si raw_dir no existe o no contiene PDFs.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"El directorio del corpus no existe: {raw_dir}. "
            f"Créalo y descarga PDFs siguiendo skills/corpus-ingest.md §1."
        )

    document_files = sorted(
        list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.txt"))
    )
    if not document_files:
        raise FileNotFoundError(
            f"No se encontraron documentos (.pdf o .txt) en {raw_dir}. "
            f"Descarga al menos un documento (ver skills/corpus-ingest.md §1) "
            f"y vuelve a ejecutar."
        )

    n_pdf = sum(1 for p in document_files if p.suffix.lower() == ".pdf")
    n_txt = sum(1 for p in document_files if p.suffix.lower() == ".txt")
    logger.info(
        "Encontrados %d documentos en %s (%d PDF, %d TXT)",
        len(document_files), raw_dir, n_pdf, n_txt,
    )

    logger.info(
        "Inicializando cliente Voyage para embeddings (%s)...", embed_model_name,
    )
    config = load_config()
    voyage = voyageai.Client(api_key=config.voyage_api_key)
    logger.info("Cliente Voyage listo")

    persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Conectando a ChromaDB en %s", persist_dir)
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Corpus legal mexicano de Legis"},
    )

    total_chunks = 0
    failed_files: list[str] = []

    for i, doc_path in enumerate(document_files, start=1):
        logger.info("[%d/%d] Procesando %s", i, len(document_files), doc_path.name)
        try:
            base_meta = parse_filename_metadata(doc_path.name)
            pages = extract_document_pages(doc_path)
            logger.info("  Extraídas %d páginas", len(pages))

            doc_type = base_meta.get("type") or "default"
            profile = CHUNK_PROFILES.get(doc_type, CHUNK_PROFILES["default"])
            chunk_size_chars = profile["size"] * CHARS_PER_TOKEN
            overlap_chars = profile["overlap"] * CHARS_PER_TOKEN

            chunks = chunk_by_articles(
                pages, base_meta, chunk_size_chars, overlap_chars,
            )

            if not chunks:
                logger.warning(
                    "  Sin chunks indexables (documento vacío o sin texto extraíble): %s",
                    doc_path.name,
                )
                continue

            logger.info(
                "  Generados %d chunks (perfil '%s', ~%d tokens objetivo)",
                len(chunks), doc_type, profile["size"],
            )

            for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
                texts = [c.text for c in batch]
                embeddings = voyage.embed(
                    texts, model=embed_model_name, input_type="document",
                ).embeddings
                collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[c.metadata for c in batch],
                )
                logger.info(
                    "  Indexados %d/%d chunks de %s",
                    min(batch_start + EMBED_BATCH_SIZE, len(chunks)),
                    len(chunks),
                    doc_path.name,
                )

            total_chunks += len(chunks)

        except Exception as exc:
            logger.error("  Error procesando %s: %s", doc_path.name, exc)
            failed_files.append(doc_path.name)
            continue

    logger.info("=" * 60)
    logger.info(
        "Indexación completa: %d chunks indexados en la colección '%s'",
        total_chunks, collection_name,
    )
    logger.info("ChromaDB persistido en: %s", persist_dir)
    if failed_files:
        logger.warning(
            "Archivos con error (%d): %s",
            len(failed_files), ", ".join(failed_files),
        )

    return total_chunks


if __name__ == "__main__":
    config = load_config()
    logger.info("Iniciando indexación del corpus legal de Legis")
    try:
        total = index_corpus(
            raw_dir=config.corpus_raw_dir,
            persist_dir=config.chroma_persist_dir,
            embed_model_name=config.embed_model,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if total == 0:
        logger.warning(
            "No se indexó ningún chunk. Revisa los PDFs y los avisos de arriba."
        )
        sys.exit(1)

    sys.exit(0)
