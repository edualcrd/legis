# Legis — Asistente Laboral con IA para Despachos en México

## Contexto del negocio

SaaS de IA para despachos laboralistas pequeños en México (2-10 abogados).
El producto resuelve consultas de jurisprudencia de la SCJN, Ley Federal del
Trabajo y criterios del IMSS en segundos — trabajo que hoy le toma al abogado
horas de búsqueda manual.

**Cliente ideal:** Despacho laboral de 2-5 abogados en CDMX, Monterrey o
Guadalajara. Factura $30,000-$150,000 MXN/mes. No tiene herramientas de IA.
Pierde 8-15 horas semanales en búsqueda de jurisprudencia.

**Precio:** $500 MXN/mes por usuario. Plan Estudio (hasta 5 usuarios): $1,500 MXN/mes.

**Propuesta de valor:** "El precedente exacto, en segundos."

**Dominio:** legis.mx / legis.ai

---

## Stack técnico

| Capa | Tecnología | Razón |
|---|---|---|
| Embeddings | Voyage `voyage-3` (API) | Sin modelos locales → imagen ligera que cabe en Render (migrado desde bge-m3 local, 2026-05-23) |
| Vectores | ChromaDB (local, horneado en la imagen Docker) | Gratis, sin latencia de red; corpus estático de solo lectura |
| Reranker | Voyage `rerank-2.5` (API) | top-20 → top-5 (migrado desde bge-reranker-v2-m3 local, 2026-05-23) |
| Léxico | BM25 (rank-bm25, en memoria) | Carril de keyword exacto para consultas coloquiales |
| LLM | claude-haiku-4-5 | Costo mínimo (~$5/mes en validación), rápido |
| Interfaz | FastAPI + React (SPA single-file vía CDN, sin build step) | Migrado desde Streamlit en fase de scaffolding |
| Auth | Magic link (Supabase + Resend + JWT) | Acceso por invitación al beta; `/api/query` protegido |
| Hosting | Render (free tier, Docker) | Bajo costo en validación; posible solo tras quitar torch/modelos locales |

---

## Estructura del proyecto

```
legal-ai-mexico/
├── CLAUDE.md                  # Este archivo — contexto permanente
├── skills/
│   ├── legal-rag.md           # Cómo manejar contenido legal mexicano
│   ├── corpus-ingest.md       # Cómo procesar PDFs legales
│   ├── legal-prompts.md       # Biblioteca de prompts por caso de uso
│   └── qa-legal.md            # Casos de prueba con respuesta esperada
├── corpus/
│   ├── raw/                   # PDFs y .txt originales descargados
│   └── processed/             # archivos limpios listos para indexar
├── src/
│   ├── api.py                 # FastAPI: /api/query, /api/stats, /api/auth/*, /auth, /health + sirve frontend
│   ├── auth.py                # Magic link: Supabase + Resend + JWT (lógica; rutas en api.py)
│   ├── ingest.py              # Indexación del corpus (embeddings vía Voyage)
│   ├── rag.py                 # Recuperación + rerank (Voyage) + respuesta
│   ├── prompts.py             # System prompts centralizados
│   └── utils.py               # Config tipada (.env) y logger
├── frontend/
│   └── index.html             # SPA React single-file (CDN, sin build step); login + app
├── scripts/
│   ├── descargar_leyes.py     # LFT, CPEUM, LSS, LFTSE, criterios IMSS
│   └── listar_tesis_sjf.py    # Tesis SCJN laborales del API público SJF
├── tests/
│   └── casos_reales.py        # Pruebas con consultas reales de abogados
├── supabase_schema.sql        # Esquema de las tablas usuarios + magic_tokens
├── Dockerfile                 # Imagen de despliegue (python:3.11-slim)
├── render.yaml                # Blueprint de Render (servicio web Docker)
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

---

## Fuentes del corpus legal (fase 1)

| Fuente | URL | Contenido |
|---|---|---|
| SCJN — Semanario Judicial | sjf.scjn.gob.mx | Tesis y jurisprudencia laboral |
| Cámara de Diputados | diputados.gob.mx/LeyesBiblio | LFT completa actualizada |
| IMSS | imss.gob.mx | Criterios normativos vigentes |
| STPS | stps.gob.mx | Laudos y normativa laboral |

**Meta corpus fase 1:** 500+ documentos indexados antes del primer demo.

---

## Casos de uso prioritarios (fase 1)

Estos 5 casos resuelven el 80% de las consultas de un despacho laboral:

1. **Cálculo de liquidación** — partes proporcionales, 3 meses, 20 días por año
2. **Búsqueda de jurisprudencia** — tesis SCJN por tema o supuesto específico
3. **Redacción de contratos** — contrato individual de trabajo por tipo de relación
4. **Criterios IMSS** — outsourcing, bases de cotización, multas
5. **Procedencia de reinstalación** — análisis con precedentes aplicables

---

## Convenciones de código

- **Idioma:** código en inglés, comentarios y mensajes de error en español
- **Errores:** siempre en español claro — los mensajes los ve el abogado, no un dev
- **Docstrings:** obligatorios en todas las funciones
- **Typing:** usar type hints en Python en toda función nueva
- **Logs:** usar `logging`, nunca `print()` en producción

```python
# ✅ Correcto
def search_jurisprudencia(query: str, top_k: int = 5) -> list[dict]:
    """
    Busca tesis relevantes en el corpus indexado.
    
    Args:
        query: Consulta en lenguaje natural del abogado
        top_k: Número de documentos a recuperar
    
    Returns:
        Lista de chunks con contenido y metadatos de fuente
    """

# ❌ Incorrecto
def search(q):
    print("buscando...")
```

---

## Reglas críticas del dominio legal

Estas reglas son NO negociables. Un error legal destruye la confianza del cliente:

1. **NUNCA inventar artículos.** Si el corpus no tiene la respuesta, decirlo
   explícitamente: *"No encontré información suficiente en los documentos
   disponibles para responder esta consulta."*

2. **SIEMPRE citar la fuente exacta:** número de tesis, artículo de ley,
   fecha de criterio. Formato: `[LFT Art. 123]` o `[SCJN Tesis: 2a./J. 45/2019]`

3. **NUNCA dar consejo legal definitivo.** El asistente informa y facilita —
   el criterio jurídico final es siempre del abogado.

4. **Distinguir jurisprudencia obligatoria de tesis aislada.** Una tesis
   aislada no es jurisprudencia. Indicarlo siempre en la respuesta.

5. **Verificar vigencia.** Si un artículo fue reformado, indicar la fecha
   de la última reforma disponible en el corpus.

---

## Definition of Done

Una funcionalidad está completa cuando:

- [ ] Pasa los casos de prueba en `tests/casos_reales.py`
- [ ] Cita fuente exacta en cada respuesta
- [ ] No alucina información legal en 20 pruebas consecutivas
- [ ] Responde en menos de 5 segundos
- [ ] El mensaje de error (si aplica) está en español claro
- [ ] Tiene docstring completo

---

## Agentes y sus responsabilidades

Cuando trabajes en este proyecto, adopta el rol indicado:

| Rol | Cuándo usarlo | Responsabilidad |
|---|---|---|
| **Arquitecto** | Inicio de cada módulo nuevo | Diseña antes de codificar |
| **Ing. de datos** | Todo lo relacionado con corpus | ingest.py, limpieza de PDFs |
| **Backend dev** | RAG, API, lógica de negocio | rag.py, prompts.py, api.py |
| **Frontend dev** | Interfaz de usuario | frontend/index.html (SPA React vía CDN) |
| **QA legal** | Antes de cualquier demo | Prueba con casos reales, detecta alucinaciones |

**Cómo invocar un agente:**
> "Actúa como [rol]. Lee el CLAUDE.md y [instrucción específica]."

---

## Estado actual del proyecto

| Fase | Estado | Descripción |
|---|---|---|
| Corpus | 🟡 En curso | Scripts listos (descargar_leyes.py, listar_tesis_sjf.py); falta ejecutar |
| Ingestión | ✅ Completo | src/ingest.py soporta PDF (PyMuPDF) y TXT; embeddings vía Voyage API |
| RAG core | ✅ Completo | src/rag.py: voyage-3 → ChromaDB → rerank-2.5 → Haiku 4.5 (migrado de bge-* locales el 2026-05-23) |
| Interfaz v2 | ✅ Completo | src/api.py + frontend/index.html (React vía CDN, sin build) |
| Auth beta | ✅ Código completo | Magic link (Supabase + Resend + JWT); `/api/query` protegido. Falta: crear tablas, env vars, verificar dominio en Resend |
| Re-indexado Voyage | ⬜ Pendiente | Correr `python src/ingest.py` con `VOYAGE_API_KEY` antes de desplegar (el chroma_db actual es bge-m3, incompatible) |
| Despliegue Render | 🟡 Archivos listos | Dockerfile + render.yaml + .dockerignore; falta `docker build` y configurar el servicio |
| QA fase 1 | ⬜ Pendiente | 20 casos reales — stubs en tests/; RE-CORRER tras el cambio de reranker |
| Primer demo | ⬜ Pendiente | Primer usuario beta: Pedro (abogado laboralista, CDMX) |

Actualiza este bloque al completar cada fase.

---

## Decisiones técnicas tomadas (log)

_Registra aquí cada decisión importante para no repetir la discusión._

| Fecha | Decisión | Razón |
|---|---|---|
| Inicio | ChromaDB local sobre Pinecone | Costo $0 en validación |
| Inicio | Haiku sobre Sonnet | 10x más barato, suficiente para MVP |
| Inicio | Streamlit sobre React | Lanzar en días, no semanas |
| 2026-05-23 | Embeddings + rerank a Voyage API (voyage-3 + rerank-2.5); se elimina torch/sentence-transformers/transformers/llama-index | El stack local (~4-6 GB RAM) no cabía en Render free tier (512 MB); la imagen baja a ~200 MB |
| 2026-05-23 | Convertir `relevance_score [0,1]` de Voyage a logit (inverse-sigmoid) en `_relevance_to_logit` | Preservar las constantes calibradas del reranker (LAW_RERANK_BOOST, umbrales, sigmoid de confianza) sin reescribirlas |
| 2026-05-23 | Auth por magic link: Supabase + Resend + JWT | Acceso por invitación al beta sin construir login con contraseñas |
| 2026-05-23 | Proteger `/api/query` en backend (no solo gating de pantalla) | Que nadie con la URL pública gaste la cuota de Haiku |
| 2026-05-23 | Despliegue en Render con Docker; índice ChromaDB horneado en la imagen | Imagen reproducible; corpus estático, sin disco persistente de pago |
