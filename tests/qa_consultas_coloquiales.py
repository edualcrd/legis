"""
QA instrumentado de 5 consultas coloquiales sobre el RAG de Legis.

Este script ejecuta el pipeline completo de src/rag.py (expansion → retrieve →
rerank → LLM) pero reportando cada paso para diagnosticar fallas en
consultas que un abogado tiparía en lenguaje natural ("freelancer",
"embarazada", "cuánto le toca de liquidación").

No es pytest: se ejecuta directamente y escribe un reporte Markdown en
`tests/output/qa_<fase>_mejoras.md`. La fase (pre o post) se determina por
el primer argumento de línea de comando.

Uso:
    python tests/qa_consultas_coloquiales.py pre
    python tests/qa_consultas_coloquiales.py post
    python tests/qa_consultas_coloquiales.py post_quota
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Aseguramos que `src.*` resuelva cuando se ejecuta desde la raíz del repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (  # noqa: E402
    DEFAULT_RETRIEVE_K_PER_QUERY,
    DEFAULT_TOP_K,
    _call_llm,
    _estimate_confidence,
    expand_query,
    format_citation,
    format_context,
    rerank,
    retrieve,
)

CONSULTAS: list[str] = [
    "freelancer relación laboral empresa",
    "elementos relación laboral prestación servicios independientes",
    "abandono de trabajo despido",
    "cuánto le toca de liquidación trabajador 5 años",
    "puede el patrón despedir a una embarazada",
]

FRASE_NO_ENCONTRE = re.compile(
    r"no\s+encontr[ée]\s+informaci[óo]n\s+suficiente",
    re.IGNORECASE,
)
PATRON_LFT = re.compile(r"\[LFT\s+Art\.\s+([\w\d/]+)", re.IGNORECASE)
PATRON_SCJN = re.compile(r"\[SCJN\s+(J|T):\s+([^\]]+)\]", re.IGNORECASE)
PATRON_CPEUM = re.compile(r"\[CPEUM\s+Art\.\s+([\w\d]+)", re.IGNORECASE)


def correr_consulta(query: str) -> dict[str, Any]:
    """
    Ejecuta el pipeline RAG completo capturando cada artefacto intermedio.

    Replica el flujo de `answer_query` pero exponiendo expansiones, candidates
    crudos y top-5 reranqueados con sus scores. No modifica producción.

    Args:
        query: Consulta del abogado en lenguaje natural.

    Returns:
        Dict con keys: query, expansions, top_chunks (con citation, source,
        type, rerank_score, snippet), answer, confidence, citas_lft, citas_scjn,
        citas_cpeum, dijo_no_encontre.
    """
    expansions = expand_query(query)
    queries: list[str] = [query, *expansions]
    candidates = retrieve(queries, top_k=DEFAULT_RETRIEVE_K_PER_QUERY)
    top = rerank(query, candidates, top_k=DEFAULT_TOP_K)
    context = format_context(top)
    answer = _call_llm(query, context)
    confidence = _estimate_confidence(top)

    top_chunks_summary = [
        {
            "citation": format_citation(c.metadata),
            "source": c.metadata.get("source", ""),
            "type": c.metadata.get("type", ""),
            "rerank_score": round(c.rerank_score or 0.0, 3),
            "snippet": c.text[:200].replace("\n", " "),
        }
        for c in top
    ]

    citas_lft = sorted(set(PATRON_LFT.findall(answer)))
    citas_scjn = [f"{t}: {n}" for t, n in PATRON_SCJN.findall(answer)]
    citas_cpeum = sorted(set(PATRON_CPEUM.findall(answer)))
    dijo_no_encontre = bool(FRASE_NO_ENCONTRE.search(answer))

    return {
        "query": query,
        "expansions": expansions,
        "n_candidates": len(candidates),
        "top_chunks": top_chunks_summary,
        "answer": answer,
        "confidence": confidence,
        "citas_lft": citas_lft,
        "citas_scjn": citas_scjn,
        "citas_cpeum": citas_cpeum,
        "dijo_no_encontre": dijo_no_encontre,
    }


def formatear_reporte(resultados: list[dict[str, Any]], fase: str) -> str:
    """Genera el reporte Markdown completo a partir de los resultados."""
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        f"# Reporte QA — {fase} mejoras",
        f"",
        f"Generado: {fecha}",
        f"",
        f"## Resumen ejecutivo",
        f"",
        f"| # | Consulta | Conf. | LFT | SCJN | CPEUM | \"No encontré\" |",
        f"|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(resultados, start=1):
        lft = ", ".join(r["citas_lft"]) or "—"
        scjn = ", ".join(r["citas_scjn"]) or "—"
        cpeum = ", ".join(r["citas_cpeum"]) or "—"
        no_enc = "sí" if r["dijo_no_encontre"] else "no"
        q_short = r["query"][:50]
        lines.append(
            f"| {i} | {q_short} | {r['confidence']:.2f} | {lft} | {scjn} | {cpeum} | {no_enc} |"
        )

    conf_avg = sum(r["confidence"] for r in resultados) / len(resultados)
    lines.append("")
    lines.append(f"**Confianza promedio:** {conf_avg:.2f}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(resultados, start=1):
        lines.append(f"## Consulta #{i}: {r['query']}")
        lines.append("")
        lines.append(f"**Confianza:** {r['confidence']:.2f} | **Candidatos recuperados:** {r['n_candidates']}")
        lines.append("")
        lines.append("### Expansiones generadas")
        lines.append("")
        if r["expansions"]:
            for j, exp in enumerate(r["expansions"], start=1):
                lines.append(f"{j}. {exp}")
        else:
            lines.append("_(la expansión falló o devolvió vacío — se usó solo la consulta original)_")
        lines.append("")
        lines.append("### Top-5 chunks reranqueados")
        lines.append("")
        lines.append("| # | Cita | Source | Type | Score | Snippet |")
        lines.append("|---|---|---|---|---|---|")
        for j, c in enumerate(r["top_chunks"], start=1):
            snippet = c["snippet"].replace("|", "\\|")
            lines.append(
                f"| {j} | {c['citation']} | {c['source']} | {c['type']} | "
                f"{c['rerank_score']:+.3f} | {snippet[:120]}... |"
            )
        lines.append("")
        lines.append("### Citas extraídas de la respuesta")
        lines.append("")
        lines.append(f"- **LFT:** {', '.join(r['citas_lft']) if r['citas_lft'] else '—'}")
        lines.append(f"- **SCJN:** {', '.join(r['citas_scjn']) if r['citas_scjn'] else '—'}")
        lines.append(f"- **CPEUM:** {', '.join(r['citas_cpeum']) if r['citas_cpeum'] else '—'}")
        lines.append(f"- **¿Dijo \"no encontré información\"?** {'sí' if r['dijo_no_encontre'] else 'no'}")
        lines.append("")
        lines.append("### Respuesta del LLM")
        lines.append("")
        lines.append("```markdown")
        lines.append(r["answer"])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


PHASE_TO_FILENAME: dict[str, str] = {
    "pre": "qa_pre_mejoras.md",
    "post": "qa_post_mejoras.md",
    "post_quota": "qa_post_quota.md",
    "post_dual": "qa_post_dual.md",
}


def main() -> int:
    """Punto de entrada: ejecuta las 5 consultas y escribe el reporte."""
    fase = sys.argv[1] if len(sys.argv) > 1 else "pre"
    if fase not in PHASE_TO_FILENAME:
        opciones = ", ".join(sorted(PHASE_TO_FILENAME.keys()))
        print(f"Fase inválida: '{fase}'. Usa una de: {opciones}.", file=sys.stderr)
        return 1

    output_path = PROJECT_ROOT / "tests" / "output" / PHASE_TO_FILENAME[fase]

    print(f"=== QA Legis: ejecutando {len(CONSULTAS)} consultas (fase={fase}) ===")
    resultados: list[dict[str, Any]] = []
    for i, query in enumerate(CONSULTAS, start=1):
        print(f"\n[{i}/{len(CONSULTAS)}] {query}")
        try:
            r = correr_consulta(query)
            print(f"  Confianza: {r['confidence']:.2f} | LFT: {r['citas_lft']} | SCJN: {len(r['citas_scjn'])} citas | 'no encontré': {r['dijo_no_encontre']}")
            resultados.append(r)
        except Exception as exc:
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

    reporte = formatear_reporte(resultados, fase)
    output_path.write_text(reporte, encoding="utf-8")
    print(f"\n=== Reporte guardado en: {output_path} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
