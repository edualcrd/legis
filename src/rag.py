"""
Pipeline RAG de Legis: recuperación, reranking y generación de respuestas legales.

Flujo (alineado con skills/legal-rag.md):
    consulta del abogado
        → Voyage voyage-3 (embedding de la query, vía API)
        → ChromaDB top-20 (similitud coseno sobre `legis_corpus`)
        → Voyage rerank-2.5 top-5 (reranker vía API)
        → claude-haiku-4-5 con SYSTEM_PROMPT_BASE (prompts.py)
        → respuesta en Markdown con la estructura de legal-rag.md §4

Reglas no negociables aplicadas (skills/legal-rag.md §5):
    - El LLM nunca ve los chunks sin cita asociada (format_context les antepone
      la cita exacta a usar).
    - El system prompt obliga a usar la frase "No encontré información..."
      cuando no hay respaldo en el contexto.
    - Distinción J vs T propagada al LLM vía el campo "Carácter" del contexto.

Embeddings y reranking corren por API (Voyage), no con modelos locales: la
imagen de despliegue se mantiene ligera (sin torch) y cabe en Render. El
reranker de Voyage devuelve relevance_score en [0,1]; `_relevance_to_logit`
lo convierte a la escala logit que esperan los boosts y umbrales calibrados
(ver `rerank()` y `_relevance_to_logit`). El cliente Voyage y el de Anthropic
se inicializan una sola vez por proceso vía singletons lazy.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import anthropic
import chromadb
import voyageai
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from src.prompts import EXPAND_QUERY_SYSTEM_PROMPT, SYSTEM_PROMPT_BASE
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

# Carril BM25: top-k léxico por keyword exacta. Rescata consultas coloquiales
# que la similitud semántica de bge-m3 no acerca a los términos técnicos
# (ej. "cuánto me toca" → "indemnización", "liquidación" en Arts. 48/50 LFT).
DEFAULT_RETRIEVE_K_BM25_LANE: int = 10

# Tokenización para BM25 en español legal: solo letras Unicode + dígitos,
# sin stemming (preserva "patrón" vs "patronal") y sin stopwords (preserva
# "no", "sin", "con" que cambian sentido jurídico). MISMA función para
# corpus indexado y para la query — invariante crítico.
_BM25_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[\wáéíóúñü]+", re.UNICODE,
)

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

# Boost aditivo post-reranker para imponer la jerarquía legal mexicana
# (Constitución > Ley federal > Jurisprudencia obligatoria > Tesis aislada).
# El cross-encoder bge-reranker-v2-m3 no conoce esa jerarquía y tiende a
# premiar tesis SCJN cuyos rúbricas coinciden léxicamente con la consulta,
# desplazando artículos LFT/CPEUM al fondo del top-K. Sumamos una constante
# al rerank_score (NO multiplicamos: los scores son logits no acotados y
# multiplicar negativos los hunde más, agravando el bug). Valores
# calibrados contra fase A de QA: +0.8 mueve un Art. LFT con score -0.4
# a +0.4, superando a una tesis tangencial con score +0.3. La cuota
# LAW_QUOTA + LAW_SCORE_FLOOR sigue operando sobre el score boosteado,
# rescatando ley marginalmente bajo el piso (consistente con la intención
# original de la cuota: dar piso semántico a la promoción).
LAW_RERANK_BOOST: float = 1.2
MANDATORY_JURISPRUDENCIA_BOOST: float = 0.3

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
# Captura el primer arreglo JSON en la respuesta — tolera prólogo o ```json fences.
_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)

# === Singletons lazy ===

_voyage_client: voyageai.Client | None = None
_collection: Collection | None = None
_anthropic_client: anthropic.Anthropic | None = None
_config: Settings | None = None

# Singletons del carril BM25. Se construyen una sola vez por proceso en
# `_get_bm25_index()` leyendo el corpus completo desde ChromaDB. Mismo
# contrato que `_collection`: si ingest.py modifica el corpus, hay que
# reiniciar el servidor para que BM25 vea los cambios.
#
# Solo persisten el índice (estadísticas BM25, sin texto crudo) y los
# chunk_ids alineados por posición. El texto y los metadatos del corpus NO
# se mantienen residentes: en `retrieve()` se re-piden por id a ChromaDB solo
# para los ~10 top hits del carril, evitando decenas de MB de strings
# residentes para siempre en el free tier de Render (512 MB).
_bm25_index: BM25Okapi | None = None
_bm25_chunk_ids: list[str] | None = None


@dataclass
class RetrievedChunk:
    """
    Chunk recuperado de ChromaDB, opcionalmente reranqueado.

    Campos:
        text: Texto del chunk tal como se indexó.
        metadata: Metadatos verificables del chunk (source, doc_id,
            article, page, type, mandatory, etc. — ver ingest.py).
        distance: Distancia coseno al embedding de la query (menor =
            más similar). Vale `float("inf")` cuando el chunk solo
            vino del carril BM25 (sin pasar por similitud vectorial);
            en ese caso el orden final lo resuelve `rerank_score`.
        rerank_score: Score EFECTIVO usado para ordenar (logit del
            cross-encoder + boost de jerarquía legal aplicado en
            `rerank()`). None mientras el chunk está en el pool crudo.
            Es el score que ven los umbrales (DEFENSIVE_RELEVANCE_THRESHOLD,
            LAW_SCORE_FLOOR), la sigmoid de confianza y `_describe_relevance`.
        raw_rerank_score: Logit crudo del cross-encoder bge-reranker-v2-m3
            antes del boost. None mientras el chunk está en el pool crudo.
            Útil para trazabilidad/QA: permite distinguir si un chunk subió
            por mérito propio o por la jerarquía legal.
        bm25_score: Score BM25 (sólo presente si el chunk apareció en el
            carril léxico). Sirve para trazabilidad y depuración de QA;
            no afecta el orden final.
    """

    text: str
    metadata: dict[str, Any]
    distance: float
    rerank_score: float | None = None
    raw_rerank_score: float | None = None
    bm25_score: float | None = None


def _get_config() -> Settings:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_voyage_client() -> voyageai.Client:
    """
    Devuelve el cliente de Voyage AI (embeddings + rerank), inicializado una
    sola vez por proceso. Reemplaza a los antiguos singletons de bge-m3 y
    bge-reranker-v2-m3 locales: ya no se cargan modelos pesados en memoria.
    """
    global _voyage_client
    if _voyage_client is None:
        config = _get_config()
        logger.info(
            "Inicializando cliente Voyage (embeddings: %s, rerank: %s)...",
            config.embed_model, config.rerank_model,
        )
        _voyage_client = voyageai.Client(api_key=config.voyage_api_key)
        logger.info("Cliente Voyage listo")
    return _voyage_client


def _relevance_to_logit(relevance_score: float) -> float:
    """
    Convierte el relevance_score de Voyage (p ∈ [0,1]) a un logit sin acotar.

    El reranker local previo (bge-reranker-v2-m3) devolvía logits sin acotar,
    y todas las constantes del pipeline (LAW_RERANK_BOOST,
    MANDATORY_JURISPRUDENCIA_BOOST, los umbrales DEFENSIVE_RELEVANCE_THRESHOLD /
    LAW_SCORE_FLOOR / _describe_relevance y la sigmoid de _estimate_confidence)
    están calibradas en esa escala. Voyage devuelve probabilidades
    normalizadas, así que aplicamos el inverso de la sigmoid
    (logit = ln(p / (1 - p))) con clamp en los extremos para evitar ±inf. Así
    toda la lógica calibrada sigue siendo válida sin reescribir constantes.

    Args:
        relevance_score: Score de relevancia de Voyage en [0, 1].

    Returns:
        Logit equivalente (float sin acotar).
    """
    p = min(max(float(relevance_score), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        config = _get_config()
        # anonymized_telemetry=False impide que ChromaDB envíe telemetría a
        # posthog: deseable en un producto legal (no se filtra nada hacia
        # afuera). No reduce RAM ni silencia el log benigno "Failed to send
        # telemetry event ClientStartEvent" (bug conocido de chromadb/posthog).
        client = chromadb.PersistentClient(
            path=str(config.chroma_persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
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


# === Tokenización e índice BM25 (carril léxico) ===

def _tokenize_es(text: str) -> list[str]:
    """
    Tokeniza texto en español para indexación/consulta BM25.

    Lowercase + extracción de tokens Unicode alfanuméricos. Sin stemming
    (preserva "patrón" vs "patronal") y sin stopwords (preserva "no",
    "sin", "con", que cambian sentido jurídico). Debe usarse la MISMA
    función para el corpus indexado y para la query — invariante crítico.

    Args:
        text: Texto crudo (chunk del corpus o consulta del abogado).

    Returns:
        Lista de tokens en minúsculas; lista vacía si el texto no contiene
        caracteres alfanuméricos.
    """
    return _BM25_TOKEN_PATTERN.findall(text.lower())


def _get_bm25_index() -> tuple[BM25Okapi | None, list[str]]:
    """
    Construye y cachea el índice BM25 sobre todos los chunks del corpus.

    Carga única por proceso (mismo contrato que `_get_collection`): si
    `ingest.py` modifica el corpus tras el arranque, hay que reiniciar
    el servidor para que el carril BM25 vea los cambios.

    Solo retiene en memoria el índice (estadísticas BM25) y los chunk_ids
    alineados por posición; NO conserva el texto ni los metadatos del corpus
    (los re-pide `retrieve()` por id solo para los top hits del carril). Por
    eso el `collection.get` de aquí pide únicamente `documents`: lo justo para
    tokenizar y construir el índice, sin materializar también los metadatos.

    Si la colección está vacía (corpus no ingestado), devuelve `(None, [])` y
    registra un warning — el carril BM25 se salta sin romper `retrieve()`.
    Cuando el primer chunk sea indexado y el servidor se reinicie, el índice
    se construye en la primera consulta.

    Returns:
        Tupla `(índice, chunk_ids)`, con los chunk_ids alineados por posición
        con las puntuaciones de `índice.get_scores()`. `índice` es `None` si
        el corpus está vacío.
    """
    global _bm25_index, _bm25_chunk_ids

    if _bm25_chunk_ids is not None:
        return _bm25_index, _bm25_chunk_ids

    collection = _get_collection()
    data = collection.get(include=["documents"])
    ids = data.get("ids", []) or []
    documents = data.get("documents", []) or []

    if not ids:
        logger.warning(
            "Corpus vacío al construir índice BM25; carril léxico deshabilitado "
            "hasta el próximo reinicio tras ingestar.",
        )
        _bm25_index = None
        _bm25_chunk_ids = []
        return _bm25_index, _bm25_chunk_ids

    t0 = time.perf_counter()
    tokenized_corpus = [_tokenize_es(doc) for doc in documents]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunk_ids = list(ids)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Índice BM25 construido: %d chunks tokenizados en %.2fs",
        len(tokenized_corpus), elapsed,
    )
    return _bm25_index, _bm25_chunk_ids


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
            system=EXPAND_QUERY_SYSTEM_PROMPT,
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


def _merge_bm25_results(
    top_ids: list[str],
    top_scores: list[float],
    texts_by_id: dict[str, str],
    metas_by_id: dict[str, dict[str, Any]],
    best_by_id: dict[str, RetrievedChunk],
) -> int:
    """
    Fusiona el top-N del carril BM25 en `best_by_id`.

    Los scores BM25 no son comparables con las distancias coseno de los
    carriles vectoriales (uno es similitud creciente, la otra es
    distancia decreciente), así que esta función NO compite por "menor
    distancia": solo añade chunks nuevos al pool y anota `bm25_score`
    en los que ya estaban (trazabilidad para QA). El orden final lo
    resuelve `rerank()` con cross-encoder scores comparables entre los
    tres carriles.

    Los chunks añadidos llevan `distance=float("inf")` como placeholder
    no-comparable: si el reranker se desactivara, quedarían al final del
    `sorted` por distancia. Con el reranker activado (caso normal), el
    `rerank_score` sobrescribe el orden y `distance=inf` es irrelevante.

    Args:
        top_ids: chunk_ids del top-N BM25, en orden de score descendente.
        top_scores: Scores BM25 paralelos a `top_ids`.
        texts_by_id: Mapa chunk_id → texto del chunk (de los singletons).
        metas_by_id: Mapa chunk_id → metadatos del chunk.
        best_by_id: Acumulador mutado en sitio (compartido con
            `_merge_chroma_results`).

    Returns:
        Número de chunks NUEVOS añadidos al pool (no contabiliza los que
        solo recibieron anotación de `bm25_score`).
    """
    added = 0
    for chunk_id, score in zip(top_ids, top_scores):
        existing = best_by_id.get(chunk_id)
        if existing is None:
            best_by_id[chunk_id] = RetrievedChunk(
                text=texts_by_id.get(chunk_id, ""),
                metadata=metas_by_id.get(chunk_id, {}),
                distance=float("inf"),
                bm25_score=score,
            )
            added += 1
        else:
            existing.bm25_score = score
    return added


def retrieve(
    query: str | list[str],
    top_k: int = DEFAULT_RETRIEVE_K,
    law_lane_k: int = DEFAULT_RETRIEVE_K_LAW_LANE,
    bm25_lane_k: int = DEFAULT_RETRIEVE_K_BM25_LANE,
) -> list[RetrievedChunk]:
    """
    Recupera candidatos en tres carriles y los fusiona.

    Carril 1 — semántico (clásico): top-K por similitud coseno sobre todo
    el corpus, sin filtros. Captura tesis SCJN cuyas rúbricas se parecen
    a la consulta y artículos de ley que bge-m3 acerca semánticamente.

    Carril 2 — normativo: top-`law_lane_k` adicionales por query,
    forzando `where={"type": {"$in": LAW_TYPES}}`. Garantiza que el pool
    SIEMPRE contenga candidatos de LFT/CPEUM, incluso cuando bge-m3 los
    descarta por similitud frente a tesis (problema diagnosticado en
    tests/output/qa_comparativo_quota.md). El reranker decide después si
    son relevantes; al menos tiene la oportunidad de verlos.

    Carril 3 — BM25 léxico: top-`bm25_lane_k` por keyword exacto sobre la
    query ORIGINAL únicamente (no las expansiones, que ya inyectan
    números de artículo y producirían falsos positivos al match
    literalmente "Art" y "48"). Rescata consultas coloquiales como
    "cuánto me toca" → matchea literalmente "indemnización"/"liquidación"
    en Arts. 48/50 LFT, donde bge-m3 no acerca lo suficiente.

    Los carriles 1 y 2 se fusionan por menor distancia coseno (mejor
    evidencia). El carril 3 se fusiona aditivamente: solo añade chunks
    nuevos al pool (con `distance=inf` como placeholder no-comparable) y
    anota `bm25_score` en los que ya estaban. El reranker resuelve el
    orden final con cross-encoder scores comparables entre los 3.

    Args:
        query: Consulta en lenguaje natural, o lista de reformulaciones.
        top_k: Candidatos por query del carril semántico.
        law_lane_k: Candidatos por query del carril normativo. Usa 0 para
            desactivarlo.
        bm25_lane_k: Top-N del carril BM25 sobre la query original. Usa 0
            para desactivarlo (degrada al pipeline previo de dos carriles).

    Returns:
        Lista de RetrievedChunk con texto, metadatos, distancia coseno y
        opcionalmente bm25_score. Ordenada por distancia ascendente (los
        chunks que solo vinieron de BM25 quedan al final con
        `distance=inf`; el reranker reordena después por rerank_score).
    """
    queries: list[str] = [query] if isinstance(query, str) else list(query)

    voyage = _get_voyage_client()
    collection = _get_collection()
    config = _get_config()

    embeddings = voyage.embed(
        queries, model=config.embed_model, input_type="query",
    ).embeddings

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

    # Carril 3 — BM25 léxico (solo sobre la query ORIGINAL).
    bm25_added = 0
    if bm25_lane_k > 0:
        bm25, bm25_ids = _get_bm25_index()
        original_query = queries[0]
        tokenized_query = _tokenize_es(original_query)
        if bm25 is None:
            logger.warning("Carril BM25: corpus no indexado, omitido")
        elif not tokenized_query:
            logger.warning(
                "Carril BM25: query vacía tras tokenizar (%r), omitido",
                original_query,
            )
        else:
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:bm25_lane_k]
            top_ids = [bm25_ids[i] for i in top_indices]
            top_scores = [float(scores[i]) for i in top_indices]
            # Texto y metadatos SOLO de los top hits: se re-piden por id a
            # ChromaDB en vez de mantener todo el corpus residente en memoria
            # (clave para caber en Render free tier). `get(ids=...)` puede
            # devolver los ids en orden arbitrario, así que se mapean por id.
            fetched = collection.get(
                ids=top_ids, include=["documents", "metadatas"],
            )
            fetched_ids = fetched.get("ids", []) or []
            fetched_docs = fetched.get("documents", []) or []
            fetched_metas = fetched.get("metadatas", []) or []
            texts_by_id = {
                cid: doc for cid, doc in zip(fetched_ids, fetched_docs)
            }
            metas_by_id = {
                cid: (m or {}) for cid, m in zip(fetched_ids, fetched_metas)
            }
            bm25_added = _merge_bm25_results(
                top_ids, top_scores, texts_by_id, metas_by_id, best_by_id,
            )
            logger.info(
                "Carril BM25: %d chunks únicos añadidos al pool "
                "(top-%d sobre query original)",
                bm25_added, bm25_lane_k,
            )

    chunks = sorted(best_by_id.values(), key=lambda c: c.distance)

    logger.info(
        "Pool final: %d candidatos únicos (semántico: %d, "
        "normativo añadidos: %d, BM25 añadidos: %d, %d %s)",
        len(chunks), semantic_unique, normative_added, bm25_added,
        len(queries), "query" if len(queries) == 1 else "queries",
    )
    return chunks


def _hierarchy_boost(metadata: dict[str, Any]) -> float:
    """
    Devuelve el boost aditivo a sumar al logit del cross-encoder según la
    jerarquía legal mexicana (Constitución > Ley federal > Jurisprudencia
    obligatoria > Tesis aislada).

    Aditivo y no multiplicativo: los logits del reranker pueden ser
    negativos, y multiplicar un negativo por >1 lo hunde más — invirtiendo
    el efecto deseado. La suma siempre desplaza hacia arriba.

    Args:
        metadata: Metadatos del chunk (claves `type` y `mandatory`).

    Returns:
        Float a sumar al rerank_score crudo. 0.0 cuando no aplica boost.
    """
    chunk_type = metadata.get("type")
    if chunk_type in LAW_TYPES:
        return LAW_RERANK_BOOST
    if chunk_type == "jurisprudencia" and metadata.get("mandatory"):
        return MANDATORY_JURISPRUDENCIA_BOOST
    return 0.0


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = DEFAULT_TOP_K,
    law_quota: int = LAW_QUOTA,
    law_score_floor: float = LAW_SCORE_FLOOR,
) -> list[RetrievedChunk]:
    """
    Reordena con el reranker de Voyage (rerank-2.5) y devuelve los top-K.

    Voyage devuelve relevance_score en [0,1]; `_relevance_to_logit` lo convierte
    a la escala logit del antiguo cross-encoder local, de modo que los boosts y
    umbrales calibrados (abajo) siguen siendo válidos sin tocarlos.

    Aplica DOS mecanismos para imponer la jerarquía legal mexicana:

    1. Boost aditivo de jerarquía (NUEVO):
        - ley_federal / constitucion → rerank_score + LAW_RERANK_BOOST (+0.8)
        - jurisprudencia con mandatory=True → + MANDATORY_JURISPRUDENCIA_BOOST (+0.3)
        - tesis (T) y demás tipos → sin cambio
       El logit crudo del cross-encoder se preserva en `raw_rerank_score`
       para trazabilidad; el `rerank_score` es el score EFECTIVO post-boost
       que el resto del pipeline usa (sort, umbrales, sigmoid de confianza,
       descripción cualitativa). Aditivo, no multiplicativo: los logits
       pueden ser negativos y un multiplicador los hunde más.

    2. Cuota suave (existente): si tras el boost siguen faltando chunks
       ley primaria en el top-K natural (`law_in_top < law_quota`),
       promueve los mejores chunks ley del pool restante cuyo rerank_score
       boosteado supere `law_score_floor`, desplazando los no-ley con
       menor score. Sin ampliar top_k ni inyectar ruido.

    Args:
        query: Misma consulta usada en `retrieve`.
        chunks: Candidatos de `retrieve`.
        top_k: Número de chunks que llegan al LLM (típicamente 5).
        law_quota: Mínimo de chunks tipo ley_federal/constitucion en el
            top-K. Usa 0 para desactivar la cuota.
        law_score_floor: Score mínimo (logit BOOSTEADO) que debe tener un
            chunk ley para ser promovido. Evita inyectar artículos
            irrelevantes.

    Returns:
        Top-K chunks ordenados por rerank_score (post-boost) descendente.
    """
    if not chunks:
        return []

    voyage = _get_voyage_client()
    config = _get_config()
    documents = [c.text for c in chunks]
    reranking = voyage.rerank(
        query, documents, model=config.rerank_model, top_k=len(documents),
    )
    # Voyage devuelve los resultados ordenados por relevancia, cada uno con su
    # índice original (`.index`) y `relevance_score` en [0,1]. Reconstruimos un
    # array alineado con `chunks` y lo convertimos a logit para preservar la
    # escala que esperan los boosts y umbrales calibrados.
    scores = [0.0] * len(chunks)
    for result in reranking.results:
        scores[result.index] = _relevance_to_logit(result.relevance_score)

    law_boosted = 0
    juris_boosted = 0
    for chunk, score in zip(chunks, scores):
        raw = float(score)
        chunk.raw_rerank_score = raw
        boost = _hierarchy_boost(chunk.metadata)
        chunk.rerank_score = raw + boost
        if boost == LAW_RERANK_BOOST:
            law_boosted += 1
        elif boost == MANDATORY_JURISPRUDENCIA_BOOST:
            juris_boosted += 1

    logger.info(
        "Boost jerarquía: %d chunks ley primaria (+%.2f), %d jurisprudencia "
        "obligatoria (+%.2f)",
        law_boosted, LAW_RERANK_BOOST,
        juris_boosted, MANDATORY_JURISPRUDENCIA_BOOST,
    )

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
