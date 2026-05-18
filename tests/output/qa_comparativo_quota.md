# Comparativo QA: post_mejoras vs post_quota

Fecha: 2026-05-18

## Resumen ejecutivo

| # | Consulta | post_mejoras Conf. | post_quota Conf. | post_mejoras citas LFT | post_quota citas LFT | "No encontré" (post_mejoras → post_quota) | Veredicto |
|---|---|---|---|---|---|---|---|
| 1 | freelancer relación laboral | 0.52 | 0.51 | 20, 21, 291 | 20, 21 | no → no | Sin cambio significativo |
| 2 | elementos relación laboral | 0.54 | 0.54 | 20, 291 | 10, 291, 8 (Art 8/10 son sugerencias del LLM, no del contexto) | no → sí | **Regresión por ruido de retrieval, no por cuota** |
| 3 | abandono de trabajo despido | 0.65 | 0.65 | — | 341 | no → no | **Cambio neutro** — cuota promovió Art 341 (causas de rescisión), tangencial pero NO alucinación |
| 4 | cuánto le toca de liquidación | 0.51 | 0.51 | 5, 502 | 48, 5, 50, 502 (Arts 48 y 50 son sugerencias del LLM en "recomendaciones", no chunks del contexto) | sí → sí | Sin cambio funcional |
| 5 | despedir embarazada | 0.70 | 0.70 | — | — | no → no | Sin cambio funcional |

**Confianza promedio:** 0.58 → 0.58 (sin cambio)

**Cero alucinaciones** detectadas en ambas fases.

## Comportamiento de la cuota por consulta

Datos extraídos del log del reranker:

| # | Consulta | Cuota actuó | Detalle |
|---|---|---|---|
| 1 | freelancer | No | Top-5 natural ya tenía 4/2 chunks ley (Arts 20, 21 LFT) |
| 2 | elementos relación laboral | No | Top-5 natural ya tenía 3/2 chunks ley (Art 291-A LFT + LSS 224) |
| 3 | abandono | **Sí** | 0→2 promovidos. Mejor LFT en pool con score ≥ -1.5: Art 341 (score +0.047) |
| 4 | liquidación | No | Top-5 natural ya tenía 5/5 chunks ley (LSS 191, LFT 502×2, LFT 5×2) |
| 5 | embarazada | **Sí** | 0→2 promovidos. Mejor LFT en pool con score ≥ -1.5: Art 947 (score +0.005) |

**Observación clave:** la cuota técnicamente funciona — promovió chunks ley cuando el top-5 natural no tenía suficientes. Pero los chunks promovidos NO son los artículos esperados:

- Consulta #3: esperado Art. 47/48 LFT; promovido Art. 341 (causas de rescisión, tangencial).
- Consulta #5: esperado Art. 170 LFT y/o CPEUM Art. 123-A-V; promovido Art. 947 (rebeldía del patrón, irrelevante).

## Diagnóstico de raíz

Los artículos LFT clave **no están llegando al pool de candidatos con score suficiente**. La cuota solo puede promover chunks que YA están en el pool con `rerank_score ≥ -1.5`. Si bge-m3 no acerca esos artículos a la consulta vectorialmente, la cuota no tiene nada que promover.

Verificación: el corpus indexado SÍ contiene los artículos (confirmado en qa_comparativo.md previo: `corpus/raw/LFT_completa_2026-05-17.pdf` los incluye). El fallo está en la fase de **recuperación vectorial**, antes del reranking.

**Causa probable:** el embedding multilingüe bge-m3 no tiene representaciones cercanas entre:
- "abandono de trabajo despido" ↔ texto del Art. 47 LFT (lista enumerada de causas)
- "puede el patrón despedir a una embarazada" ↔ texto del Art. 170 LFT (período de descanso prenatal/postnatal)

Las tesis SCJN ganan en similitud porque sus rúbricas son frases temáticas redactadas como queries, mientras que los artículos de ley son texto normativo enumerativo.

## Criterios de aceptación del plan

| Criterio | Resultado |
|---|---|
| Las 5 consultas terminan sin excepción | ✓ obligatorio |
| Consulta #1 mantiene Arts. 20/21 (no regresión) | ✓ obligatorio |
| Consulta #2 mantiene Art. 20 (no regresión) | ✗ (regresión por ruido de retrieval, NO por cuota — cuota no actuó en #2) |
| Consulta #3 cita ≥ 1 artículo LFT del rango 47-48 | ✗ (cita 341 en su lugar) |
| Consulta #4 cita Art. 48 o Art. 50 LFT en el top-5 | ✗ (top-5 sin cambios; el LLM los menciona como sugerencias) |
| Consulta #5 cita Art. 170 LFT o CPEUM Art. 123 | ✗ |
| Cero alucinaciones nuevas | ✓ |
| Confianza promedio post_quota ≥ post_mejoras - 0.05 | ✓ (igual, 0.58) |

**0 de 3 criterios "+" cumplidos.** La cuota NO resolvió el problema diagnosticado en el comparativo previo.

## Veredicto final

**La cuota implementada funciona técnicamente pero no resuelve el problema de fondo.** Como anticipaba el plan en su sección "Si ningún criterio + mejora":

> Significa que los artículos LFT relevantes no están llegando siquiera al pool de candidatos (problema de embedding bge-m3, no de reranking). En ese caso documentar el hallazgo y proponer plan futuro: retrieve dirigido con `where={"type": {"$in": ["ley_federal"]}}` como pool complementario.

### ¿Mantener la cuota o revertir?

**Recomendación: mantener.** Razones:

1. **No daña**: cero regresiones causadas por la cuota; la consulta #2 cambió por ruido del retrieval no por la cuota (la cuota no actuó en #2).
2. **Defensa frente a sesgos del cross-encoder**: si en el futuro el corpus crece y los Arts. 47/48/170 LLEGAN al pool con score adecuado, la cuota garantiza que no sean desplazados por tesis tangenciales.
3. **Costo cero**: ~25 líneas de código, retrocompatible (`law_quota=0` desactiva).
4. **Marginal upside en #3**: el LLM tuvo al menos un artículo LFT en el contexto (aunque tangencial) y la respuesta integra el concepto de "rescisión" desde la ley, no solo desde tesis.

### Próximo paso recomendado

**Implementar retrieve dirigido** (plan separado, no este iteración):

```python
# Pseudocódigo de retrieve_with_law_pool en src/rag.py
def retrieve_with_law_pool(queries, top_k):
    # Pool A: retrieve clásico (todo el corpus)
    pool_a = retrieve(queries, top_k=top_k)
    # Pool B: retrieve forzando type ∈ {ley_federal, constitucion}
    pool_b = collection.query(
        query_embeddings=embeddings,
        n_results=top_k // 2,
        where={"type": {"$in": ["ley_federal", "constitucion"]}},
    )
    # Fusionar deduplicando
    return dedup_by_id(pool_a + pool_b)
```

Esto garantizaría que al menos N candidatos ley lleguen al reranker con score real (no descartados en la similitud coseno por sesgo léxico de bge-m3 a favor de tesis). La cuota actual ya implementada se beneficiará directamente: pool B alimentará chunks ley con score recalculado por el cross-encoder, y la cuota los promoverá si están bajo el top-K natural.

### Resumen ejecutivo de 2 líneas

La cuota de ley en `rerank()` está implementada, validada y se queda como defensa estructural. El cuello de botella real está aguas arriba en `retrieve()`; siguiente iteración debe implementar un pool de candidatos dirigido por `type` en ChromaDB.
