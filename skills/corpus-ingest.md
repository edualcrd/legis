# Skill: corpus-ingest
# Cómo procesar PDFs legales mexicanos para indexarlos en ChromaDB

---

## Propósito

Esta skill define el procedimiento para añadir, normalizar e indexar
documentos legales en el corpus de Legis. Garantiza que cada PDF se
parsea correctamente, conserva metadatos verificables y respeta las
reglas de chunking de [`legal-rag.md`](./legal-rag.md) §6.

---

## 1. Fuentes oficiales y dónde descargarlas

| Fuente | URL | Tipo de documento |
|---|---|---|
| SCJN — Semanario Judicial | sjf.scjn.gob.mx | Tesis y jurisprudencia |
| Cámara de Diputados | diputados.gob.mx/LeyesBiblio | LFT, LSS, LINFONAVIT, CPEUM |
| IMSS | imss.gob.mx | Criterios normativos |
| STPS | stps.gob.mx | Laudos y NOMs laborales |

Solo se descargan **PDFs digitales**, no escaneos. Si una fuente solo está
disponible escaneada, queda fuera de alcance hasta que se añada OCR
(ver "Fuera de alcance" abajo).

---

## 2. Convención de nombres de archivo

El nombre del PDF lleva los metadatos clave para que `ingest.py` pueda
parsearlos sin ambigüedad. Formato:

```
{FUENTE}_{IDENTIFICADOR}_{FECHA}.pdf
```

### Ejemplos

| Tipo | Nombre de archivo |
|---|---|
| Ley Federal del Trabajo, última reforma | `LFT_completa_2024-01-15.pdf` |
| LFT, capítulo específico | `LFT_cap-VI-trabajadores-confianza_2024-01-15.pdf` |
| Jurisprudencia obligatoria SCJN | `SCJN_J_2a-J-45-2019_10a.pdf` |
| Tesis aislada SCJN | `SCJN_T_XVII-2o-PA-3-L_10a.pdf` |
| Criterio normativo IMSS | `IMSS_Criterio_01-2023-NOM_2023-03-10.pdf` |
| Constitución | `CPEUM_completa_2024-09-15.pdf` |

**Reglas:**
- Sin espacios. Usar guiones medios (`-`) dentro de un campo, guiones
  bajos (`_`) entre campos.
- Fecha en formato ISO `YYYY-MM-DD` (fecha de publicación o última reforma).
- Para tesis: el segundo campo es `J` (jurisprudencia obligatoria) o `T`
  (tesis aislada) seguido del número de tesis.
- La época SCJN va al final: `_10a.pdf` o `_11a.pdf`.

---

## 3. Metadatos obligatorios por chunk

Cada chunk indexado en ChromaDB debe llevar **todos** estos metadatos.
Sin ellos, las citas no son verificables y violamos la regla §5 de
`legal-rag.md`.

```python
metadata = {
    "source": "LFT",                # LFT | CPEUM | LSS | SCJN | IMSS | STPS
    "doc_id": "LFT_completa_2024-01-15",   # nombre del archivo sin .pdf
    "article": "48",                # número de artículo si aplica, "" si no
    "thesis_number": "",            # "2a./J. 45/2019" para tesis SCJN, "" si no
    "thesis_type": "",              # "J" (obligatoria) | "T" (aislada) | "" si no aplica
    "epoca": "",                    # "10a." | "11a." | "" si no aplica
    "date": "2024-01-15",           # ISO date del documento
    "type": "ley_federal",          # ley_federal | constitucion | jurisprudencia | tesis | criterio_imss | laudo
    "mandatory": True,              # True si es ley o jurisprudencia obligatoria
    "page": 12,                     # página del PDF de origen
}
```

**Por qué cada campo importa:**
- `source` + `article` → cita en formato `[LFT Art. 48]`
- `thesis_type` + `thesis_number` + `epoca` → cita `[SCJN J: 2a./J. 45/2019 (10a.)]`
- `mandatory` → permite al RAG distinguir jurisprudencia obligatoria de tesis aislada
- `date` → genera la sección "Vigencia" de la respuesta (legal-rag.md §4)
- `page` → para depurar manualmente cuando el modelo cita mal

---

## 4. Reglas de chunking (extraídas de legal-rag.md §6)

### Regla de oro
**Nunca cortar un artículo a la mitad.** El chunker debe detectar los
límites de artículo (`Artículo NN.`) y respetarlos.

### Tamaños por tipo de documento

| Tipo | Chunk size | Overlap |
|---|---|---|
| LFT, LSS, CPEUM (artículos cortos) | 512 tokens | 50 tokens |
| Tesis SCJN (texto denso) | 768 tokens | 100 tokens |
| Criterios IMSS | 512 tokens | 75 tokens |
| Laudos (documentos largos) | 1024 tokens | 150 tokens |

### Implementación recomendada
- Usar `SentenceSplitter` de LlamaIndex como base
- Sobrescribir el `chunking_tokenizer_fn` para que detecte `\nArtículo \d+\.`
  como separador prioritario antes de aplicar el límite de tokens

---

## 5. Procedimiento manual antes de indexar

Antes de añadir un PDF nuevo al corpus, **el ingeniero de datos** debe:

1. **Verificar la fuente:** descargar solo del sitio oficial. Anotar la URL
   en un comentario del commit que añade el PDF.
2. **Inspeccionar el PDF en visor:** abrir el PDF y confirmar que el
   texto es seleccionable (no es un escaneo). Si no se puede seleccionar,
   está fuera de alcance hasta tener OCR.
3. **Renombrar siguiendo la convención** (§2 arriba).
4. **Extracción de prueba:** correr una extracción rápida con PyMuPDF y
   verificar manualmente las primeras 3 páginas:
   ```python
   import fitz
   doc = fitz.open("corpus/raw/LFT_completa_2024-01-15.pdf")
   print(doc[0].get_text())
   ```
   Si el texto sale con columnas mezcladas, caracteres raros o artículos
   cortados → ese PDF requiere preprocesamiento manual (ponerlo en
   `corpus/processed/` con las correcciones).
5. **Indexar:** una vez verificado, correr `python src/ingest.py`. El
   script registra cada documento procesado en el log.

---

## 6. Cómo añadir una fuente nueva al corpus

1. Descargar el PDF de la fuente oficial.
2. Aplicar el procedimiento manual de §5.
3. Si el tipo de documento es nuevo (p. ej. NOM en lugar de tesis),
   añadir su tipo al enum `type` de metadatos (§3) y documentar el formato
   de cita correspondiente en [`legal-rag.md`](./legal-rag.md) §2.
4. Reindexar todo el corpus (`python src/ingest.py`) para que los
   nuevos metadatos sean consistentes.
5. Añadir al menos un caso de prueba en `tests/casos_reales.py` que use
   la nueva fuente.

---

## 7. Fuera de alcance (fase 1)

Estas son decisiones explícitas para no añadir complejidad antes del MVP:

- **OCR:** PDFs escaneados (algunos criterios IMSS viejos, laudos
  estatales) quedan excluidos. Reevaluar cuando haya 3+ clientes pagando.
- **Scraping automático:** la descarga de PDFs es manual. No vale la pena
  automatizar antes de validar que los abogados pagan por el producto.
- **Versionado automático de reformas:** cuando la LFT se reforme, se
  reemplaza manualmente el PDF y se reindexa. La fecha en el nombre
  del archivo y en `metadata["date"]` mantiene la trazabilidad.

---

## 8. Checklist de QA antes de cerrar la fase de ingestión

- [ ] Todos los PDFs de `corpus/raw/` siguen la convención de nombres
- [ ] Cada chunk tiene los metadatos completos de §3
- [ ] Inspección manual de 10 chunks aleatorios: ningún artículo cortado
- [ ] El número total de chunks indexados está en el log
- [ ] `chroma_db/` está poblado y `python -c "import chromadb; ..."`
      puede consultarlo
- [ ] Al menos 1 PDF de cada tipo (`ley_federal`, `jurisprudencia`,
      `tesis`, `criterio_imss`) está indexado
