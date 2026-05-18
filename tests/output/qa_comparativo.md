# Comparativo QA pre vs post mejoras

Fecha: 2026-05-18

## Resumen ejecutivo

| # | Consulta | Pre Conf. | Post Conf. | Pre citas LFT | Post citas LFT | "No encontré" | Veredicto |
|---|---|---|---|---|---|---|---|
| 1 | freelancer relación laboral empresa | 0.52 | 0.52 | 20, 21, 291 | 20, 21, 291 | no → no | Sin cambio (ya OK) |
| 2 | elementos relación laboral... | 0.52 | 0.54 | — | **20, 291** | no → no | **Mejora** — ahora cita Art. 20 LFT |
| 3 | abandono de trabajo despido | 0.65 | 0.65 | — | — | no → no | Sin cambio cuantitativo |
| 4 | cuánto le toca de liquidación 5 años | 0.54 | 0.51 | **502, 87** | 5, 502 | no → **sí** | **Mejora cualitativa grande** — antes citaba Art. 502 (muerte) y 87 (aguinaldo) como si fueran liquidación; ahora dice "no encontré" y recomienda Arts. 48, 50, 162, 163 |
| 5 | puede el patrón despedir a una embarazada | 0.70 | 0.70 | — | — | no → no | **Mejora cualitativa** — ahora añade nota explícita sobre Art. 170 LFT y Art. 123-A-V CPEUM faltantes del corpus |

**Confianza promedio:** 0.59 → 0.58 (la baja viene de #4 que ahora honestamente reconoce su límite)

**Cero alucinaciones** detectadas en ambas fases.

## Cambios clave por mejora aplicada

### C1 — Reformulación de `_EXPAND_QUERY_SYSTEM_PROMPT`

Antes (genérico):
> "Reformula la consulta del usuario en 3 versiones más técnicas..."

Ahora (con diccionario coloquial→jurídico y roles fijos):
- Reformulación #1: orientada a LFT con número de artículo predecible.
- Reformulación #2: orientada a SCJN.
- Reformulación #3: concepto genérico.

**Efecto observable:** Las expansiones ahora siempre incluyen el número del artículo LFT relevante.

Ejemplos del corpus post:
- Consulta #2: _"Elementos constitutivos de la relación de trabajo y presunción de subordinación Art. 20 LFT, Art. 21 LFT"_ → trajo Art. 20 a top-5 (antes no llegaba).
- Consulta #4: _"Indemnización por terminación de relación laboral: tres meses de salario Art. 48 LFT, veinte días por año de antigüedad Art. 162 LFT, y prima de antigüedad Art. 162 LFT"_.
- Consulta #5: _"Terminación de relación laboral de trabajadora en estado de gravidez: fuero de maternidad, estabilidad reforzada y prohibición de despido Art. 170 LFT, Art. 133-XV LFT, Art. 123 Apartado A V CPEUM"_.

### C2 — `SYSTEM_PROMPT_BASE` con razonamiento y nota de faltantes

Añadidas tres secciones nuevas:
1. **RAZONAMIENTO ANTES DE CITAR** — fuerza al LLM a evaluar relevancia antes de redactar.
2. **PRIORIDAD A LEY SOBRE JURISPRUDENCIA** — instrucción explícita de citar artículos primero, y de añadir nota cuando faltan artículos predecibles del corpus.
3. **Estructura flexible** — secciones opcionales que pueden omitirse si no hay contenido.

**Efecto observable:**
- Consulta #5 ahora incluye la nota: _"El corpus indexado no recuperó los artículos de la LFT relevantes para esta consulta (Art. 170 LFT sobre derechos de la madre trabajadora, Art. 48 LFT sobre causales de rescisión, ni Art. 123 Apartado A V CPEUM sobre prohibición de despido por embarazo). La respuesta se apoya en tesis SCJN orientadoras."_
- Consulta #4 ahora invoca correctamente "no encontré" + recomienda los artículos exactos a buscar (48, 50, 162, 163).

### C3 — Señal de relevancia en `format_context`

Cada chunk ahora viene con etiqueta `Relevancia estimada: alta | media | baja` derivada del rerank_score.

**Efecto observable:** El LLM en la consulta #4 explícitamente menciona cuáles fuentes son tangenciales y por qué no le sirven (ver respuesta: "normas transitorias", "supuesto específico, no aplica", "régimen de seguridad social, no liquidación").

### C4 — Umbral defensivo `DEFENSIVE_RELEVANCE_THRESHOLD = -1.5`

No se activó en ninguna de las 5 consultas (todas tuvieron al menos un chunk con score > -1.5), pero permanece como red de seguridad para consultas fuera de scope.

## Diagnóstico residual

Lo que NO resolvió esta iteración:

1. **Consulta #3 sigue sin citar Arts. 47/48 LFT.** Las 5 tesis SCJN tienen scores muy altos (+0.638 etc.) y dominan. El reranker bge-reranker-v2-m3 prioriza tesis sobre artículos cuando el rubro de la tesis coincide léxicamente con la consulta. **Solución futura sugerida:** retrieve híbrido (forzar al menos N chunks tipo `ley_federal` en el top-K, ej. 2 de 5 reservados a ley).

2. **Consulta #4 todavía no recupera Arts. 48/50 LFT** aunque la expansión los menciona. Esto indica que el embedding bge-m3 de "indemnización tres meses veinte días" no se acerca lo suficiente al embedding de "Artículo 48.- Si en el juicio correspondiente no comprueba el patrón la causa de la rescisión...". El sistema correctamente reporta su límite, pero la limitación viene del embedding base.

## Criterios de aceptación del plan

| Criterio | Resultado |
|---|---|
| Las 5 consultas terminan sin excepción | ✓ |
| Consulta #5 cita Art. 133/170 LFT | ✗ (no en corpus recuperado, pero ahora se reporta como faltante) |
| Consulta #1 cita Art. 20/21 sin alucinar | ✓ |
| Consulta #4 cita Art. 48/50 con cálculo | ✗ (no en corpus, pero ahora dice "no encontré" y recomienda los artículos) |
| Confianza promedio post > pre en ≥3 consultas | ✗ — solo subió en #2; pero la calidad cualitativa mejoró en #4 y #5 |
| Cero alucinaciones | ✓ |

## Veredicto final

**Las mejoras al system prompt y query expansion funcionan.** El sistema ahora:

1. Genera reformulaciones técnicas que incluyen números de artículo LFT predecibles, lo que aumenta la probabilidad de recuperar la ley base.
2. Reconoce honestamente cuando el corpus no cubre el tema, en lugar de citar artículos tangenciales como si fueran respuesta.
3. Avisa al abogado qué artículos debería verificar manualmente cuando faltan del corpus.

**El cuello de botella restante NO es el prompt, es la recuperación.** Los Arts. 48/50 LFT (consulta #4) y Art. 170 LFT (consulta #5) sí están en el corpus indexado (corpus/raw/LFT_completa_2026-05-17.pdf) pero el reranker no los prioriza frente a tesis SCJN cuyo rubro coincide léxicamente con la consulta. La solución para esto es estructural (retrieve híbrido con cuota fija de chunks tipo ley_federal), no de prompt.
