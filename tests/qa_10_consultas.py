"""
QA legal: 10 consultas coloquiales/en primera persona sobre el RAG de Legis.

Reutiliza la maquinaria instrumentada de `tests/qa_consultas_coloquiales.py`
(que generó los reportes históricos de `tests/output/`) sin tocar su set de
regresión de 5 consultas. Aporta dos cosas nuevas:

1. Las 10 consultas que un abogado laboralista teclearía en lenguaje natural.
2. Fidelidad con `src.rag.answer_query`: aplica el corte defensivo de
   `DEFENSIVE_RELEVANCE_THRESHOLD` (que `correr_consulta` no aplica) para que
   el reporte refleje el comportamiento real de producción.
3. Ranking de las 3 consultas con peor resultado, con recomendación de mejora.

Uso:
    python tests/qa_10_consultas.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Aseguramos que `src.*` y `tests.*` resuelvan desde la raíz del repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (  # noqa: E402
    DEFENSIVE_RELEVANCE_THRESHOLD,
    NO_ENCONTRE_CANONICAL,
)
from tests.qa_consultas_coloquiales import (  # noqa: E402
    FRASE_NO_ENCONTRE,
    correr_consulta,
    formatear_reporte,
)

# Consultas exactas pedidas por el rol QA legal: coloquiales y en primera
# persona, tal como las tipearía un abogado describiendo el caso de su cliente.
CONSULTAS_10: list[str] = [
    "me corrieron y no me pagaron el finiquito",
    "cuánto le toca de liquidación a mi cliente con 8 años",
    "puede el patrón despedir a una embarazada",
    "mi cliente lleva 3 meses sin cobrar su quincena",
    "el patrón quiere cambiarle el horario sin avisar",
    "procede la reinstalación en este caso",
    "cuántas horas extra puede pedir el patrón",
    "el contrato dice que es freelancer pero trabaja en oficina",
    "el patrón cerró la empresa y no pagó a nadie",
    "mi cliente fue despedido por estar en el sindicato",
]

OUTPUT_PATH = PROJECT_ROOT / "tests" / "output" / "qa_10_consultas.md"


def aplicar_corte_defensivo(resultado: dict[str, Any]) -> dict[str, Any]:
    """
    Replica el corte defensivo de `answer_query` (src/rag.py:1055) sobre el
    resultado de `correr_consulta`, que por diseño NO lo aplica.

    Si todos los top chunks tienen rerank_score < DEFENSIVE_RELEVANCE_THRESHOLD,
    producción devuelve la frase canónica sin invocar al LLM. Aquí sobrescribimos
    la respuesta y los flags para que el reporte refleje ese comportamiento real,
    usando los scores que `correr_consulta` ya devolvió (sin inferencia extra).

    Args:
        resultado: Dict devuelto por `correr_consulta`.

    Returns:
        El mismo dict, mutado si aplicó el corte defensivo.
    """
    top_chunks = resultado.get("top_chunks", [])
    if not top_chunks:
        return resultado

    todos_tangenciales = all(
        c["rerank_score"] < DEFENSIVE_RELEVANCE_THRESHOLD for c in top_chunks
    )
    if todos_tangenciales:
        resultado["answer"] = NO_ENCONTRE_CANONICAL
        resultado["dijo_no_encontre"] = bool(
            FRASE_NO_ENCONTRE.search(NO_ENCONTRE_CANONICAL)
        )
        resultado["citas_lft"] = []
        resultado["citas_scjn"] = []
        resultado["citas_cpeum"] = []
        resultado["corte_defensivo"] = True
    return resultado


def _sin_cita_relevante(r: dict[str, Any]) -> bool:
    """True si la respuesta no citó ninguna fuente verificable (LFT/CPEUM/SCJN)."""
    return not (r["citas_lft"] or r["citas_cpeum"] or r["citas_scjn"])


def analizar_peores(
    resultados: list[dict[str, Any]], n: int = 3
) -> list[tuple[int, dict[str, Any]]]:
    """
    Ordena las consultas de peor a mejor y devuelve las `n` peores.

    Criterio (peor primero), documentado en el reporte:
        1º  dijo "no encontré información" (no resolvió la consulta)
        2º  sin ninguna cita relevante (LFT ∧ CPEUM ∧ SCJN vacías)
        3º  menor confianza estimada

    Args:
        resultados: Lista de dicts de `correr_consulta` (ya con corte defensivo).
        n: Número de peores a devolver.

    Returns:
        Lista de tuplas (índice_1based, resultado), de peor a mejor.
    """
    indexados = list(enumerate(resultados, start=1))
    indexados.sort(
        key=lambda par: (
            not par[1]["dijo_no_encontre"],   # False (dijo no encontré) primero
            not _sin_cita_relevante(par[1]),  # False (sin citas) primero
            par[1]["confidence"],             # menor confianza primero
        )
    )
    return indexados[:n]


# Recomendación de mejora por modo de fallo detectado. La clave es la consulta
# exacta; el valor explica qué ajustar (corpus, diccionario de expand_query en
# prompts.py, o cuota/boost en rag.py). Se rellena en formatear_peores según el
# modo de fallo observado en tiempo de ejecución.
def _recomendacion(r: dict[str, Any]) -> str:
    """Sugerencia concreta de mejora según el modo de fallo del resultado."""
    if r["dijo_no_encontre"]:
        return (
            "Devolvió la frase canónica de 'no encontré'. Revisar si el artículo "
            "predecible para el supuesto está indexado y, si lo está, reforzar la "
            "entrada coloquial→técnica en EXPAND_QUERY_SYSTEM_PROMPT (src/prompts.py) "
            "para que la expansión inyecte el número de artículo correcto."
        )
    if _sin_cita_relevante(r):
        return (
            "Respondió sin citar ninguna fuente verificable (LFT/CPEUM/SCJN). "
            "Riesgo de respuesta vaga: revisar el carril normativo y la cuota de "
            "ley (LAW_QUOTA / LAW_RERANK_BOOST en src/rag.py) para forzar al menos "
            "un artículo aplicable al top-K."
        )
    return (
        "Confianza baja pese a citar fuentes: el mejor chunk quedó con rerank_score "
        "bajo. Revisar si la consulta necesita una reformulación técnica adicional "
        "en el diccionario de expand_query, o si el artículo central del supuesto "
        "falta en el corpus."
    )


def formatear_peores(peores: list[tuple[int, dict[str, Any]]]) -> str:
    """Genera la sección Markdown 'Las 3 consultas con peor resultado'."""
    lines: list[str] = [
        "## Las 3 consultas con peor resultado",
        "",
        "**Criterio de orden (peor primero):** 1º dijo \"no encontré información\"; "
        "2º sin ninguna cita relevante (LFT ∧ CPEUM ∧ SCJN vacías); "
        "3º menor confianza estimada.",
        "",
        "| Rank | # | Consulta | Conf. | Modo de fallo |",
        "|---|---|---|---|---|",
    ]
    for rank, (idx, r) in enumerate(peores, start=1):
        if r["dijo_no_encontre"]:
            modo = "dijo \"no encontré\""
        elif _sin_cita_relevante(r):
            modo = "sin cita verificable"
        else:
            modo = "confianza baja"
        lines.append(
            f"| {rank} | {idx} | {r['query']} | {r['confidence']:.2f} | {modo} |"
        )
    lines.append("")
    for rank, (idx, r) in enumerate(peores, start=1):
        lines.append(f"### Peor #{rank} — Consulta #{idx}: {r['query']}")
        lines.append("")
        lines.append(f"- **Confianza:** {r['confidence']:.2f}")
        lines.append(
            f"- **LFT:** {', '.join(r['citas_lft']) if r['citas_lft'] else '—'} "
            f"| **SCJN:** {', '.join(r['citas_scjn']) if r['citas_scjn'] else '—'} "
            f"| **CPEUM:** {', '.join(r['citas_cpeum']) if r['citas_cpeum'] else '—'}"
        )
        lines.append(
            f"- **¿Dijo \"no encontré\"?** {'sí' if r['dijo_no_encontre'] else 'no'}"
        )
        lines.append(f"- **Recomendación:** {_recomendacion(r)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Ejecuta las 10 consultas, aplica corte defensivo y escribe el reporte."""
    fase = "10 consultas coloquiales"
    print(f"=== QA Legis: ejecutando {len(CONSULTAS_10)} consultas ===")

    resultados: list[dict[str, Any]] = []
    for i, query in enumerate(CONSULTAS_10, start=1):
        print(f"\n[{i}/{len(CONSULTAS_10)}] {query}")
        try:
            r = aplicar_corte_defensivo(correr_consulta(query))
            print(
                f"  Confianza: {r['confidence']:.2f} | LFT: {r['citas_lft']} | "
                f"SCJN: {len(r['citas_scjn'])} citas | "
                f"'no encontré': {r['dijo_no_encontre']}"
            )
            resultados.append(r)
        except Exception as exc:  # noqa: BLE001 — el reporte debe sobrevivir 1 fallo
            print(f"  ERROR: {exc}")
            resultados.append({
                "query": query,
                "expansions": [],
                "n_candidates": 0,
                "top_chunks": [],
                "answer": f"ERROR DE EJECUCIÓN: {exc}",
                "confidence": 0.0,
                "citas_lft": [],
                "citas_scjn": [],
                "citas_cpeum": [],
                "dijo_no_encontre": False,
            })

    cuerpo = formatear_reporte(resultados, fase)
    peores = analizar_peores(resultados, n=3)
    reporte = cuerpo + "\n" + formatear_peores(peores)
    OUTPUT_PATH.write_text(reporte, encoding="utf-8")

    conf_avg = sum(r["confidence"] for r in resultados) / len(resultados)
    print(f"\n=== Confianza promedio: {conf_avg:.2f} ===")
    print("=== 3 peores consultas ===")
    for rank, (idx, r) in enumerate(peores, start=1):
        print(f"  {rank}. #{idx} ({r['confidence']:.2f}) {r['query']}")
    print(f"\n=== Reporte guardado en: {OUTPUT_PATH} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
