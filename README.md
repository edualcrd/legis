# Legis — Asistente Laboral con IA

> El precedente exacto, en segundos.

SaaS de IA para despachos laboralistas en México. Resuelve consultas sobre
jurisprudencia SCJN, Ley Federal del Trabajo y criterios IMSS en segundos
— trabajo que hoy le toma al abogado horas de búsqueda manual.

Lee [`CLAUDE.md`](./CLAUDE.md) para el contexto completo del negocio y
[`skills/legal-rag.md`](./skills/legal-rag.md) para las reglas de manejo
de contenido legal.

---

## Quickstart

Requisitos: Python 3.11, ~3 GB libres (modelos HF locales).

```bash
# 1. Crear entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Instalar dependencias (primera vez descarga torch ~700 MB)
pip install -r requirements.txt

# 3. Configurar credenciales
cp .env.example .env
# Editar .env y poner tu ANTHROPIC_API_KEY

# 4. Descargar corpus base (LFT, CPEUM, LSS, LFTSE, criterios IMSS)
python scripts/descargar_leyes.py
# Opcional: descargar metadatos de tesis SCJN laborales
python scripts/listar_tesis_sjf.py --max-pages 200

# 5. Indexar todo el corpus en ChromaDB local
python src/ingest.py

# 6. Lanzar la API + SPA (FastAPI sirve el frontend React en /)
uvicorn src.api:app --port 8000
# Abrir http://localhost:8000/ en el navegador
# Docs OpenAPI: http://localhost:8000/api/docs
```

---

## Arquitectura

```
Ingesta:
  corpus/raw/*.pdf
    → PyMuPDF (extracción texto)
    → chunking inteligente que respeta artículos (legal-rag.md §6)
    → BAAI/bge-m3 (embeddings)
    → ChromaDB local (persistido en ./chroma_db)

Consulta:
  pregunta del abogado
    → BAAI/bge-m3 (embedding de la query)
    → ChromaDB top-20 por similitud
    → BAAI/bge-reranker-v2-m3 top-5 (cross-encoder)
    → claude-haiku-4-5 con prompt de legal-rag.md §4
    → respuesta con citas verificables en formato [LFT Art. 48]
```

**Decisiones técnicas:**
- ChromaDB local (no Pinecone): $0 en validación, sin latencia de red
- Haiku (no Sonnet): 10× más barato, suficiente para MVP
- bge-m3 (no OpenAI): gratis, fuerte en español jurídico
- Reranker cross-encoder: reduce alucinaciones de cita (crítico por
  regla §5 de `legal-rag.md`)

---

## Estructura del proyecto

```
legis/
├── CLAUDE.md                  contexto permanente del proyecto
├── README.md
├── requirements.txt
├── .env.example
├── skills/
│   ├── legal-rag.md           reglas de dominio legal mexicano
│   └── corpus-ingest.md       reglas de procesamiento de PDFs
├── corpus/
│   ├── raw/                   PDFs y .txt originales (fuera de git)
│   └── processed/             archivos normalizados (regenerable)
├── src/
│   ├── api.py                 FastAPI: endpoints /api/* + sirve frontend
│   ├── ingest.py              indexación del corpus en ChromaDB
│   ├── rag.py                 lógica de recuperación y respuesta
│   ├── prompts.py             system prompts centralizados
│   └── utils.py               logger y configuración
├── frontend/
│   └── index.html             SPA React (single-file, sin build step)
├── scripts/
│   ├── descargar_leyes.py     descarga LFT, CPEUM, LSS, IMSS
│   └── listar_tesis_sjf.py    enumera tesis SCJN laborales
└── tests/
    └── casos_reales.py        5 casos laborales de prueba
```

---

## Reglas no negociables

Antes de tocar código que devuelva contenido legal al usuario, lee
[`skills/legal-rag.md`](./skills/legal-rag.md). Resumen:

1. **Nunca inventar** artículos de ley ni números de tesis
2. **Siempre citar** la fuente exacta: `[LFT Art. 48]`, `[SCJN J: 2a./J. 45/2019 (10a.)]`
3. **Distinguir** jurisprudencia obligatoria (J) de tesis aislada (T)
4. Si el corpus no tiene la respuesta, decirlo explícitamente
5. Mensajes al usuario siempre en español claro

---

## Estado del proyecto

Ver tabla "Estado actual del proyecto" en [`CLAUDE.md`](./CLAUDE.md).
