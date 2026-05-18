"""
Pipeline RAG de Legis: recuperación, reranking y generación de respuestas legales.

Flujo (alineado con skills/legal-rag.md):
    consulta del abogado
        → BAAI/bge-m3 (embedding de la query)
        → ChromaDB top-20 (similitud coseno sobre `legis_corpus`)
        → BAAI/bge-reranker-v2-m3 top-5 (cross-encoder)
        → claude-haiku-4-5 con SYSTEM_PROMPT_BASE (prompts.py)
        → respuesta en Markdown con la estructura de legal-rag.md §4

Reglas no negociables aplicadas (skills/legal-rag.md §5):
    - El LLM nunca ve los chunks sin cita asociada (format_context les antepone
      la cita exacta a usar).
    - El system prompt obliga a usar la frase "No encontré información..."
      cuando no hay respaldo en el contexto.
    - Distinción J vs T propagada al LLM vía el campo "Carácter" del contexto.

Los modelos pesados (bge-m3, bge-reranker-v2-m3, cliente Anthropic) se cargan
una sola vez por proceso vía singletons lazy. Esto importa para Streamlit:
una vez calientes, cada consulta solo paga ~1s de embed + ~300ms de rerank
+ latencia del LLM.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import anthropic
import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

from src.prompts import SYSTEM_PROMPT_BASE
from src.utils import Settings, get_logger, load_config

logger = get_logger(__name__)

# Debe coincidir con ingest.py para hablar de la misma colección.
COLLECTION_NAME: str = "legis_corpus"

DEFAULT_RETRIEVE_K: int = 20
DEFAULT_RETRIEVE_K_PER_QUERY: int = 10
DEFAULT_TOP_K: int = 5

# Carril normativo: candidatos extra por query, filtrados por tipo ley
# primaria. Garantiza que el pool del reranker SIEMPRE contenga chunks de
# LFT/CPEUM, incluso cuando bge-m3 los descarta por similitud frente a
# tesis SCJN cuyas rúbricas son más cercanas léxicamente a la consulta.
# Ver tests/output/qa_comparativo_quota.md §"Diagnóstico de raíz".
DEFAULT_RETRIEVE_K_LAW_LANE: int = 5
MAX_TOKENS_RESPONSE: int = 2048
LLM_TEMPERATURE: float = 0.0

# Query expansion: pedirle a Haiku reformulaciones técnicas antes de buscar.
EXPAND_QUERY_VERSIONS: int = 3
EXPAND_QUERY_MAX_TOKENS: int = 256

# Umbral defensivo: si TODOS los top chunks tienen score < este logit,
# el contexto es claramente tangencial y devolvemos la respuesta canónica
# de "no encontré" sin invocar al LLM (evita alucinación silenciosa y
# ahorra una llamada). Valor calibrado contra fase A de QA: el peor caso
# útil observado fue +0.009 (consulta #2, marginal); -1.5 deja margen
# amplio sin disparar falsos positivos.
DEFENSIVE_RELEVANCE_THRESHOLD: float = -1.5

# Cuota de ley primaria en el top-K final del reranker. Garantiza que al
# menos LAW_QUOTA chunks sean ley_federal o constitucion, siempre que su
# rerank_score supere LAW_SCORE_FLOOR. Ataca el sesgo del cross-encoder a
# favor de tesis SCJN cuyo rubro coincide léxicamente con la consulta
# (ver tests/output/qa_comparativo.md: consultas #3, #4, #5 desplazaban
# Arts. 47/48/50/170 LFT y CPEUM Art. 123 a favor de tesis tangenciales).
LAW_TYPES: frozenset[str] = frozenset({"ley_federal", "constitucion"})
LAW_QUOTA: int = 2
LAW_SCORE_FLOOR: float = -1.5

NO_ENCONTRE_CANONICAL: str = (
    "## Respuesta\n\n"
    "No encontré información suficiente en el corpus disponible para "
    "responder esta consulta con certeza. Te recomiendo consultar "
    "directamente el Semanario Judicial de la Federación en "
    "sjf.scjn.gob.mx o la versión vigente de la LFT en diputados.gob.mx.\n\n"
    "## Advertencia\n\n"
    "Esta respuesta es orientativa. El criterio jurídico final corresponde "
    "al abogado responsable del caso."
)
_EXPAND_QUERY_SYSTEM_PROMPT: str = (
    "Eres un experto en derecho laboral mexicano. Tu tarea es reformular "
    "una consulta coloquial de un abogado en 3 versiones, cada una dirigida "
    "a un tipo distinto de fuente, para maximizar la cobertura de un "
    "sistema de recuperación vectorial sobre LFT, CPEUM y tesis SCJN.\n\n"
    "DICCIONARIO COLOQUIAL → TÉCNICO (úsalo cuando aplique):\n"
    "- freelancer / independiente / por proyecto → subordinación, "
    "elementos de la relación de trabajo Art. 20 LFT, presunción Art. 21 LFT\n"
    "- embarazada / embarazo → estado de gravidez, fuero de maternidad, "
    "estabilidad reforzada, Art. 170 LFT, Art. 133-XV LFT, Art. 123 Apartado A V CPEUM\n"
    "- liquidación / finiquito / cuánto le toca → indemnización constitucional, "
    "tres meses, veinte días por año, prima de antigüedad, Art. 48 LFT, Art. 50 LFT, Art. 162 LFT\n"
    "- abandono → rescisión sin responsabilidad patronal, Art. 47-X LFT\n"
    "- despido → terminación de relación laboral, Art. 47 LFT, Art. 48 LFT\n"
    "- reinstalación → acción de reinstalación, Art. 48 LFT, Art. 49 LFT\n"
    "- aguinaldo → prestación anual mínima, Art. 87 LFT\n"
    "- vacaciones → período vacacional, Art. 76 LFT, Art. 77 LFT, Art. 80 LFT\n"
    "- outsourcing / subcontratación → servicios especializados, Art. 15 LFT, Art. 15-A LSS\n\n"
    "GENERA EXACTAMENTE 3 REFORMULACIONES con estos roles fijos:\n"
    "1. ORIENTADA A LFT: una oración breve con el/los número(s) de artículo "
    "predecible(s) del/los punto(s) anterior(es). Ejemplo: "
    "'Indemnización por despido injustificado Art. 48 LFT y prima de antigüedad Art. 162 LFT'.\n"
    "2. ORIENTADA A SCJN: pregunta con palabras clave que aparecerían en el "
    "rubro de una tesis o jurisprudencia laboral. Ejemplo: "
    "'Criterios SCJN sobre estabilidad laboral reforzada de trabajadora embarazada y carga de la prueba'.\n"
    "3. CONCEPTO JURÍDICO GENÉRICO: la consulta original traducida a "
    "vocabulario técnico, sin referencias específicas. Ejemplo: "
    "'Elementos constitutivos de la relación de trabajo: subordinación, dependencia, salario'.\n\n"
    "Devuelve SOLO un JSON array con 3 strings, sin explicación, sin "
    "Markdown, sin texto adicional."
)
# Captura el primer arreglo JSON en la respuesta — tolera prólogo o ```json fences.
_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)

# === Singletons lazy ===

_embed_model: HuggingFaceEmbedding | None = None
_reranker: CrossEncoder | None = None
_collection: Collection | None = None
_anthropic_client: anthropic.Anthropic | None = None
_config: Settings | None = None


@dataclass
class RetrievedChunk:
    """Chunk recuperado de ChromaDB, opcionalmente reranqueado."""

    text: str
    metadata: dict[str, Any]
    distance: float
    rerank_score: float | None = None


def _get_config() -> Settings:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model
    if _embed_model is None:
        config = _get_config()
        logger.info("Cargando modelo de embeddings %s...", config.embed_model)
        _embed_model = HuggingFaceEmbedding(model_name=config.embed_model)
        logger.info("Modelo de embeddings listo")
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        config = _get_config()
        logger.info("Cargando reranker %s...", config.rerank_model)
        _reranker = CrossEncoder(config.rerank_model)
        logger.info("Reranker listo")
    return _reranker


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        config = _get_config()
        client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
        try:
            _collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"No se encontró la colección '{COLLECTION_NAME}' en "
                f"{config.chroma_persist_dir}. Ejecuta `python src/ingest.py` "
                f"primero para indexar el corpus legal."
            ) from exc
    return _collection


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        config = _get_config()
        _anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _anthropic_client


# === Formato de citas (skills/legal-rag.md §2) ===

def _format_thesis_number(raw: str) -> str:
    """
    Convierte el identificador codificado en el filename al formato SCJN.

    Filename usa guiones medios (legal en sistema de archivos) en lugar de
    los puntos/barras del formato oficial:
        2a-J-45-2019   → 2a./J. 45/2019   (caso más común)
        XVII-2o-PA-3-L → XVII.2o.PA.3 L   (best-effort)

    Args:
        raw: Identificador como aparece en el filename / metadatos.

    Returns:
        Identificador en el formato esperado por la SCJN; raw original si
        no se reconoce el patrón.
    """
    if not raw:
        return ""
    parts = raw.split("-")
    if len(parts) == 4 and parts[-1].isdigit() and len(parts[-1]) == 4:
        return f"{parts[0]}./{parts[1]}. {parts[2]}/{parts[3]}"
    return raw.replace("-", ".")


def format_citation(meta: dict[str, Any]) -> str:
    """
    Construye la cita canónica de un chunk según skills/legal-rag.md §2.

    Ejemplos de salida:
        [LFT Art. 48]
        [CPEUM Art. 123]
        [SCJN J: 2a./J. 45/2019 (10a.)]
        [SCJN T: XVII.2o.PA.3 L (10a.)]
        [IMSS Criterio 01-2023-NOM]

    Args:
        meta: Diccionario de metadatos del chunk (ver ingest.py).

    Returns:
        Cita lista para insertarse en una respuesta legal.
    """
    source = meta.get("source", "")
    if not source:
        return "[fuente desconocida]"

    article = meta.get("article", "")
    thesis_type = meta.get("thesis_type", "")
    thesis_number = meta.get("thesis_number", "")
    epoca = meta.get("epoca", "")

    if source == "SCJN" and thesis_type in {"J", "T"}:
        formatted = _format_thesis_number(thesis_number) if thesis_number else "?"
        epoca_str = f" ({epoca})" if epoca else ""
        return f"[SCJN {thesis_type}: {formatted}{epoca_str}]"

    if source == "IMSS":
        if thesis_number:
            return f"[IMSS Criterio {thesis_number}]"
        return "[IMSS]"

    if source in {"LFT", "LSS", "LFTSE", "LINFONAVIT", "CPEUM"}:
        return f"[{source} Art. {article}]" if article else f"[{source}]"

    return f"[{source} Art. {article}]" if article else f"[{source}]"


# === Expansión de consulta (skills/legal-rag.md §query-expansion) ===

def expand_query(query: str) -> list[str]:
    """
    Genera reformulaciones técnicas de la consulta con Haiku.

    Antes de buscar en ChromaDB, le pedimos al LLM 3 versiones más técnicas
    que usen la terminología exacta de la LFT y de las tesis SCJN. Estas
    reformulaciones se suman a la consulta original para ampliar la
    cobertura de recuperación en consultas vagas o coloquiales.

    Si la llamada al modelo falla (timeout, JSON mal formado, número
    incorrecto de strings), devuelve [] y registra un warning en español.
    La consulta original se sigue usando aguas arriba, así que un fallo
    aquí degrada gracefully a la recuperación con un único query.

    Args:
        query: Consulta original del abogado en lenguaje natural.

    Returns:
        Lista con EXPAND_QUERY_VERSIONS reformulaciones, o [] si falla.
    """
    logger.info(
        "Expandiendo consulta en %d versiones técnicas...", EXPAND_QUERY_VERSIONS,
    )

    client = _get_anthropic_client()
    config = _get_config()

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=EXPAND_QUERY_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=_EXPAND_QUERY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Consulta: {query}"}],
        )
    except anthropic.APIError as exc:
        logger.warning(
            "La expansión de consulta falló (%s). Se usará solo la consulta original.",
            exc,
        )
        return []

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    if not raw_text:
        logger.warning(
            "La expansión de consulta falló (respuesta vacía del modelo). "
            "Se usará solo la consulta original."
        )
        return []

    match = _JSON_ARRAY_PATTERN.search(raw_text)
    if not match:
        logger.warning(
            "La expansión de consulta falló (no se encontró JSON en la respuesta). "
            "Se usará solo la consulta original."
        )
        return []

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning(
            "La expansión de consulta falló (JSON inválido: %s). "
            "Se usará solo la consulta original.",
            exc,
        )
        return []

    if not isinstance(parsed, list) or len(parsed) != EXPAND_QUERY_VERSIONS:
        logger.warning(
            "La expansión de consulta falló (se esperaban %d strings, se recibieron %s). "
            "Se usará solo la consulta original.",
            EXPAND_QUERY_VERSIONS,
            len(parsed) if isinstance(parsed, list) else type(parsed).__name__,
        )
        return []

    expansions: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            logger.warning(
                "La expansión de consulta falló (elemento no-string o vacío). "
                "Se usará solo la consulta original."
            )
            return []
        expansions.append(item.strip())

    logger.info(
        "Expansión completada: %d reformulaciones generadas", len(expansions),
    )
    return expansions


# === Recuperación ===

def _merge_chroma_results(
    results: dict[str, Any], best_by_id: dict[str, RetrievedChunk],
) -> tuple[int, int]:
    """
    Fusiona los resultados de `collection.query()` en `best_by_id`,
    deduplicando por chunk_id y conservando la menor distancia coseno
    observada (best-evidence si el chunk aparece desde varias queries).

    Args:
        results: Diccionario devuelto por `Collection.query()`.
        best_by_id: Acumulador mutado en sitio. La función NO lo limpia.

    Returns:
        Tupla (nuevos_añadidos, total_crudos): cuántos chunk_ids
        no existían antes en `best_by_id`, y cuántos pares (query, chunk)
        crudos venían en `results` antes de deduplicar.
    """
    ids_per_query = results.get("ids", [])
    docs_per_query = results.get("documents", [])
    metas_per_query = results.get("metadatas", [])
    dists_per_query = results.get("distances", [])

    added = 0
    total_raw = 0
    for ids, docs, metas, dists in zip(
        ids_per_query, docs_per_query, metas_per_query, dists_per_query,
    ):
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            total_raw += 1
            distance = float(dist)
            existing = best_by_id.get(chunk_id)
            if existing is None:
                best_by_id[chunk_id] = RetrievedChunk(
                    text=doc,
                    metadata=meta or {},
                    distance=distance,
                )
                added += 1
            elif distance < existing.distance:
                best_by_id[chunk_id] = RetrievedChunk(
                    text=doc,
                    metadata=meta or {},
                    distance=distance,
                )
    return added, total_raw


def retrieve(
    query: str | list[str],
    top_k: int = DEFAULT_RETRIEVE_K,
    law_lane_k: int = DEFAULT_RETRIEVE_K_LAW_LANE,
) -> list[RetrievedChunk]:
    """
    Recupera candidatos en dos carriles y los fusiona.

    Carril 1 — semántico (clásico): top-K por similitud coseno sobre todo
    el corpus, sin filtros. Captura tesis SCJN cuyas rúbricas se parecen
    a la consulta y artículos de ley que bge-m3 acerca semánticamente.

    Carril 2 — normativo (nuevo): top-`law_lane_k` adicionales por query,
    forzando `where={"type": {"$in": LAW_TYPES}}`. Garantiza que el pool
    SIEMPRE contenga candidatos de LFT/CPEUM, incluso cuando bge-m3 los
    descarta por similitud frente a tesis (problema diagnosticado en
    tests/output/qa_comparativo_quota.md). El reranker decide después si
    son relevantes; al menos tiene la oportunidad de verlos.

    Ambos carriles se fusionan deduplicando por chunk_id y conservando
    la menor distancia observada (best-evidence cuando un mismo chunk
    aparece desde varias queries o carriles).

    Args:
        query: Consulta en lenguaje natural, o lista de reformulaciones.
        top_k: Candidatos por query del carril semántico.
        law_lane_k: Candidatos por query del carril normativo. Usa 0 para
            desactivarlo (degrada al comportamiento clásico de un carril).

    Returns:
        Lista de RetrievedChunk con texto, metadatos y distancia coseno
        (la menor entre las queries/carriles que recuperaron ese chunk).
    """
    queries: list[str] = [query] if isinstance(query, str) else list(query)

    embed_model = _get_embed_model()
    collection = _get_collection()

    embeddings = [embed_model.get_query_embedding(q) for q in queries]

    best_by_id: dict[str, RetrievedChunk] = {}

    # Carril 1 — semántico (sin filtro de tipo)
    semantic_results = collection.query(
        query_embeddings=embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    sem_added, sem_raw = _merge_chroma_results(semantic_results, best_by_id)
    semantic_unique = len(best_by_id)

    # Carril 2 — normativo (forzar candidatos tipo ley primaria)
    normative_added = 0
    if law_lane_k > 0:
        normative_results = collection.query(
            query_embeddings=embeddings,
            n_results=law_lane_k,
            where={"type": {"$in": list(LAW_TYPES)}},
            include=["documents", "metadatas", "distances"],
        )
        normative_added, _ = _merge_chroma_results(
            normative_results, best_by_id,
        )
        logger.info(
            "Carril normativo: %d chunks de ley añadidos al pool",
            normative_added,
        )

    chunks = sorted(best_by_id.values(), key=lambda c: c.distance)

    if len(queries) == 1:
        logger.info(
            "Pool final: %d candidatos únicos (semántico + normativo)",
            len(chunks),
        )
    else:
        logger.info(
            "Pool final: %d candidatos únicos (semántico: %d, "
            "normativo añadidos: %d, %d queries)",
            len(chunks), semantic_unique, normative_added, len(queries),
        )
    return chunks


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = DEFAULT_TOP_K,
    law_quota: int = LAW_QUOTA,
    law_score_floor: float = LAW_SCORE_FLOOR,
) -> list[RetrievedChunk]:
    """
    Reordena con cross-encoder bge-reranker-v2-m3 y devuelve los top-K.

    Aplica una cuota suave: si en el top-K natural hay menos de `law_quota`
    chunks de tipo ley primaria (LAW_TYPES), promueve los mejores chunks
    ley del pool restante cuyo rerank_score supere `law_score_floor`,
    desplazando los no-ley con menor score. Esto corrige el sesgo del
    cross-encoder a favor de tesis SCJN cuando el rubro coincide
    léxicamente con la consulta — sin ampliar top_k ni inyectar ruido
    (chunks bajo el piso no se promueven).

    Args:
        query: Misma consulta usada en `retrieve`.
        chunks: Candidatos de `retrieve`.
        top_k: Número de chunks que llegan al LLM (típicamente 5).
        law_quota: Mínimo de chunks tipo ley_federal/constitucion en el
            top-K. Usa 0 para desactivar la cuota.
        law_score_floor: Score mínimo (logit) que debe tener un chunk ley
            para ser promovido. Evita inyectar artículos irrelevantes.

    Returns:
        Top-K chunks ordenados por rerank_score descendente.
    """
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    all_sorted = sorted(
        chunks, key=lambda c: c.rerank_score or 0.0, reverse=True,
    )
    natural_top = all_sorted[:top_k]

    if law_quota <= 0:
        logger.info(
            "Reranking (sin cuota): %d → top %d (mejor score: %.3f)",
            len(chunks), len(natural_top),
            natural_top[0].rerank_score if natural_top else 0.0,
        )
        return natural_top

    law_in_top = sum(
        1 for c in natural_top if c.metadata.get("type") in LAW_TYPES
    )
    if law_in_top >= law_quota:
        logger.info(
            "Reranking: %d → top %d (mejor score: %.3f, cuota ya cumplida: "
            "%d/%d chunks ley)",
            len(chunks), len(natural_top),
            natural_top[0].rerank_score if natural_top else 0.0,
            law_in_top, law_quota,
        )
        return natural_top

    law_candidates = [
        c for c in all_sorted[top_k:]
        if c.metadata.get("type") in LAW_TYPES
        and (c.rerank_score or -999.0) >= law_score_floor
    ]
    needed = law_quota - law_in_top
    promoted = law_candidates[:needed]
    if not promoted:
        logger.info(
            "Reranking: %d → top %d (cuota incumplible: ningún chunk ley "
            "con score >= %.2f en el pool)",
            len(chunks), len(natural_top), law_score_floor,
        )
        return natural_top

    final = list(natural_top)
    non_law_asc = sorted(
        [c for c in final if c.metadata.get("type") not in LAW_TYPES],
        key=lambda c: c.rerank_score or 0.0,
    )
    for c in non_law_asc[: len(promoted)]:
        final.remove(c)
    final.extend(promoted)
    final_sorted = sorted(
        final, key=lambda c: c.rerank_score or 0.0, reverse=True,
    )

    logger.info(
        "Reranking con cuota de ley: %d → top %d (mejor score: %.3f, "
        "%d→%d chunks ley, promovidos %d con score >= %.2f)",
        len(chunks), len(final_sorted),
        final_sorted[0].rerank_score if final_sorted else 0.0,
        law_in_top, law_in_top + len(promoted),
        len(promoted), law_score_floor,
    )
    return final_sorted


# === Formato del contexto para el LLM ===

def _describe_character(meta: dict[str, Any]) -> str:
    """Describe el carácter jurídico del chunk para que el LLM aplique §3."""
    doc_type = meta.get("type", "")
    mandatory = meta.get("mandatory", False)

    if doc_type == "jurisprudencia":
        return "jurisprudencia OBLIGATORIA para tribunales (5+ casos)"
    if doc_type == "tesis":
        return "tesis aislada — ORIENTADORA, NO obligatoria"
    if doc_type == "constitucion":
        return "Constitución (máxima jerarquía)"
    if doc_type == "ley_federal":
        return "ley federal vigente, obligatoria"
    if doc_type == "criterio_imss":
        return "criterio normativo IMSS"
    if doc_type == "laudo":
        return "laudo (precedente con fuerza limitada)"
    if mandatory:
        return "fuente obligatoria"
    if doc_type:
        return doc_type.replace("_", " ")
    return "carácter no determinado"


def _describe_relevance(rerank_score: float | None) -> str:
    """
    Traduce el logit del reranker a una etiqueta cualitativa para el LLM.

    bge-reranker-v2-m3 devuelve logits no calibrados. Umbrales elegidos
    empíricamente (ver tests/output/qa_pre_mejoras.md, fase A de QA):
        > 1.0   alta — el chunk responde directamente al supuesto
        > -1.0  media — relacionado pero no central
        ≤ -1.0  baja — tangencial; probablemente no aplica
    """
    if rerank_score is None:
        return "desconocida"
    if rerank_score > 1.0:
        return "alta"
    if rerank_score > -1.0:
        return "media"
    return "baja"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Formatea los chunks recuperados para el LLM, exponiendo la cita exacta
    que debe usar, el carácter jurídico (J vs T, etc.), la fecha y una
    señal cualitativa de relevancia derivada del rerank_score.

    Args:
        chunks: Top-K chunks tras reranking.

    Returns:
        Bloque de texto listo para inyectar como contexto al LLM.
    """
    if not chunks:
        return "(No se encontraron documentos relevantes en el corpus indexado.)"

    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        citation = format_citation(chunk.metadata)
        page = chunk.metadata.get("page", "?")
        date = chunk.metadata.get("date", "")
        character = _describe_character(chunk.metadata)
        relevance = _describe_relevance(chunk.rerank_score)
        date_str = f", fecha: {date}" if date else ""

        blocks.append(
            f"=== Fuente {i} ===\n"
            f"Cita EXACTA a usar: {citation}\n"
            f"Carácter: {character}\n"
            f"Relevancia estimada: {relevance}\n"
            f"Ubicación: página {page}{date_str}\n\n"
            f"Contenido:\n{chunk.text}\n"
        )

    return "\n".join(blocks)


# === Generación con Haiku ===

def _call_llm(query: str, context: str) -> str:
    """
    Llama a claude-haiku-4-5 con SYSTEM_PROMPT_BASE y devuelve el texto.

    Aplica prompt caching ephemeral al system prompt para amortizar costo
    cuando hay varias consultas en menos de 5 minutos.
    """
    client = _get_anthropic_client()
    config = _get_config()

    user_message = (
        "Contexto recuperado del corpus legal:\n\n"
        f"{context}\n"
        "---\n\n"
        f"Consulta del abogado:\n{query}"
    )

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=MAX_TOKENS_RESPONSE,
            temperature=LLM_TEMPERATURE,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT_BASE,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise RuntimeError(
            f"Error al llamar al modelo {config.llm_model}: {exc}. "
            f"Verifica tu ANTHROPIC_API_KEY y la conexión a internet."
        ) from exc

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text

    raise RuntimeError("La respuesta del LLM no contenía texto.")


def _estimate_confidence(chunks: list[RetrievedChunk]) -> float:
    """
    Estima confianza en [0, 1] a partir del logit del mejor chunk reranqueado.

    bge-reranker-v2-m3 devuelve scores no acotados; sigmoid los normaliza.
    No es una probabilidad calibrada — solo una señal relativa para la UI.
    """
    if not chunks:
        return 0.0
    top_score = chunks[0].rerank_score or 0.0
    try:
        normalized = 1.0 / (1.0 + math.exp(-top_score))
    except OverflowError:
        normalized = 0.0 if top_score < 0 else 1.0
    return round(normalized, 2)


# === Función pública ===

def answer_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    retrieve_k: int = DEFAULT_RETRIEVE_K_PER_QUERY,
) -> dict[str, Any]:
    """
    Responde una consulta legal con citas verificables del corpus indexado.

    Antes de buscar, expande la consulta en EXPAND_QUERY_VERSIONS
    reformulaciones técnicas con Haiku, recupera con todas (original +
    reformulaciones) y deduplica por chunk_id antes del reranker. Si la
    expansión falla, degrada a la consulta original sin afectar el flujo.

    Args:
        query: Consulta en lenguaje natural del abogado.
        top_k: Número de chunks finales que llegan al LLM (default 5).
        retrieve_k: Candidatos por consulta antes del reranker (default
            10). Con la consulta original más 3 reformulaciones se traen
            ~40 brutos, que tras deduplicar quedan en ~25-35 únicos.

    Returns:
        dict con claves:
            - 'answer': str en Markdown con la estructura de legal-rag.md §4
            - 'sources': list[dict] con citation, source, doc_id, page,
              rerank_score, mandatory, type — para mostrar al abogado
            - 'confidence': float en [0, 1] estimando la fuerza del fundamento

    Raises:
        ValueError: Si la consulta está vacía.
        RuntimeError: Si el corpus no está indexado o el LLM falla.
    """
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    logger.info("Consulta recibida: %s", query[:120])

    expansions = expand_query(query)
    queries: list[str] = [query, *expansions]
    candidates = retrieve(queries, top_k=retrieve_k)
    if not candidates:
        raise RuntimeError(
            "El corpus no devolvió ningún resultado. Verifica que "
            "`python src/ingest.py` haya indexado al menos un PDF."
        )

    top_chunks = rerank(query, candidates, top_k=top_k)

    # Umbral defensivo: si todos los chunks son tangenciales, evita pedirle
    # al LLM que invente — devuelve la frase canónica directamente.
    if top_chunks and all(
        (c.rerank_score or 0.0) < DEFENSIVE_RELEVANCE_THRESHOLD
        for c in top_chunks
    ):
        logger.warning(
            "Todos los top chunks tienen score < %.2f. "
            "Devolviendo respuesta canónica de 'no encontré'.",
            DEFENSIVE_RELEVANCE_THRESHOLD,
        )
        answer_text = NO_ENCONTRE_CANONICAL
    else:
        context = format_context(top_chunks)
        answer_text = _call_llm(query, context)

    sources = [
        {
            "citation": format_citation(c.metadata),
            "source": c.metadata.get("source", ""),
            "doc_id": c.metadata.get("doc_id", ""),
            "page": c.metadata.get("page", 0),
            "rerank_score": round(c.rerank_score or 0.0, 3),
            "mandatory": bool(c.metadata.get("mandatory", False)),
            "type": c.metadata.get("type", ""),
        }
        for c in top_chunks
    ]

    confidence = _estimate_confidence(top_chunks)
    logger.info("Respuesta generada (confianza estimada: %.2f)", confidence)

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
    }


def get_corpus_stats() -> dict[str, Any]:
    """
    Lee estadísticas agregadas del corpus indexado en ChromaDB.

    Consumida por src/api.py (GET /api/stats) para poblar el sidebar de
    la SPA. No carga modelos pesados — solo abre el cliente ChromaDB.

    Returns:
        dict con claves:
            - 'chunks': total de chunks indexados
            - 'documents': número de documentos únicos (por doc_id)
            - 'juris': documentos cuyo type == 'jurisprudencia'
            - 'tesis': documentos cuyo type == 'tesis'
            - 'sources': lista ordenada de códigos de fuente presentes
              (LFT, SCJN, IMSS, etc.)
        Si ChromaDB no está inicializado o la colección está vacía,
        devuelve la misma estructura con ceros y lista vacía.
    """
    try:
        collection = _get_collection()
    except RuntimeError:
        return {"chunks": 0, "documents": 0, "juris": 0, "tesis": 0, "sources": []}

    chunks_total = collection.count()
    if chunks_total == 0:
        return {"chunks": 0, "documents": 0, "juris": 0, "tesis": 0, "sources": []}

    data = collection.get(include=["metadatas"])
    metas = data.get("metadatas", []) or []

    doc_ids: set[str] = set()
    juris_docs: set[str] = set()
    tesis_docs: set[str] = set()
    sources: set[str] = set()
    for m in metas:
        if not m:
            continue
        did = m.get("doc_id", "")
        if did:
            doc_ids.add(did)
            doc_type = m.get("type", "")
            if doc_type == "jurisprudencia":
                juris_docs.add(did)
            elif doc_type == "tesis":
                tesis_docs.add(did)
        src = m.get("source", "")
        if src:
            sources.add(src)

    return {
        "chunks": chunks_total,
        "documents": len(doc_ids),
        "juris": len(juris_docs),
        "tesis": len(tesis_docs),
        "sources": sorted(sources),
    }


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]).strip()
    if not user_query:
        user_query = input("Consulta legal: ").strip()

    if not user_query:
        logger.error("No se recibió ninguna consulta.")
        sys.exit(1)

    try:
        result = answer_query(user_query)
    except (ValueError, RuntimeError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)

    print(result["answer"])
    print("\n" + "=" * 60)
    print("Fuentes utilizadas:")
    for src in result["sources"]:
        marker = "[obligatoria]" if src["mandatory"] else "[orientadora]"
        print(
            f"  {src['citation']:<45} "
            f"score={src['rerank_score']:+.3f} {marker} "
            f"(pág {src['page']})"
        )
    print(f"\nConfianza estimada: {result['confidence']}")
