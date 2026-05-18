# Skill: legal-rag
# Cómo manejar contenido legal mexicano en Legis

---

## Propósito

Esta skill define cómo Claude Code debe construir, consultar y responder
usando el corpus legal mexicano de Legis. Su objetivo es garantizar que
cada respuesta sea jurídicamente precisa, cite fuentes verificables y
nunca alucine información legal.

---

## 1. Jerarquía de fuentes legales en México

Claude debe conocer y respetar esta jerarquía al construir respuestas.
Una fuente de mayor jerarquía siempre prevalece sobre una inferior:

```
1. Constitución Política de los Estados Unidos Mexicanos (CPEUM)
2. Leyes federales (LFT, LFTSE, LSS, LINFONAVIT)
3. Jurisprudencia obligatoria de la SCJN
4. Tesis aisladas de la SCJN
5. Criterios normativos del IMSS / INFONAVIT / STPS
6. Laudos de los Tribunales Laborales
```

**Regla:** Si hay conflicto entre fuentes, siempre cita la de mayor
jerarquía y menciona explícitamente que prevalece sobre las inferiores.

---

## 2. Formato obligatorio de citas legales

Cada respuesta que cite una fuente debe usar exactamente este formato:

### Ley o código
```
[LFT Art. 123] → Ley Federal del Trabajo, Artículo 123
[CPEUM Art. 123 Apartado A] → Constitución, Artículo 123 Apartado A
[LSS Art. 15] → Ley del Seguro Social, Artículo 15
```

### Jurisprudencia SCJN
```
[SCJN J: 2a./J. 45/2019 (10a.)] → Jurisprudencia obligatoria
[SCJN T: XVII.2o.P.A. 3 L (10a.)] → Tesis aislada
```
- **J** = Jurisprudencia obligatoria
- **T** = Tesis aislada (no obligatoria, solo orientadora)
- Siempre incluir época: (10a.) = Décima Época, (11a.) = Undécima Época

### Criterio normativo IMSS
```
[IMSS Criterio 01/2023/NOM] → Criterio normativo con número y año
```

---

## 3. Distinción crítica: jurisprudencia vs tesis aislada

Esta distinción es **no negociable** en cada respuesta:

| Tipo | Carácter | Cómo indicarlo |
|---|---|---|
| Jurisprudencia (5+ casos) | **Obligatoria** para tribunales | "Conforme a jurisprudencia obligatoria..." |
| Tesis aislada (1-4 casos) | **Orientadora**, no obligatoria | "Existe una tesis aislada que orienta en el sentido de..." |

**Nunca** presentar una tesis aislada como si fuera jurisprudencia
obligatoria. Es uno de los errores más graves en práctica legal mexicana.

---

## 4. Estructura de respuesta

Toda respuesta del RAG debe seguir esta estructura:

```
## Respuesta

[Respuesta directa a la consulta en 2-3 párrafos]

## Fundamento legal

- [Cita 1 con formato correcto]
- [Cita 2 con formato correcto]

## Criterio jurisprudencial aplicable

[Si existe jurisprudencia relevante, citarla con distinción J/T]

## Vigencia

Información basada en legislación vigente al [fecha del documento más 
reciente en el corpus]. Verificar posibles reformas posteriores.

## Advertencia

Esta respuesta es orientativa. El criterio jurídico final corresponde
al abogado responsable del caso.
```

---

## 5. Reglas de alucinación — las más importantes del proyecto

Estas reglas son absolutas. Violarlas destruye la confianza del cliente:

### ❌ NUNCA hacer
- Inventar números de artículos que no están en el corpus
- Citar una tesis con número incorrecto
- Presentar un criterio como "vigente" sin verificar la fecha
- Combinar dos artículos distintos en una sola cita falsa
- Afirmar que "la jurisprudencia es unánime" sin verificarlo

### ✅ SIEMPRE hacer
- Si el corpus no tiene la respuesta: decirlo explícitamente
- Si hay duda sobre vigencia: indicarlo con "verificar reforma posterior"
- Si hay criterios contradictorios: presentar ambos con sus fuentes
- Si la consulta está fuera del corpus: sugerir fuente oficial donde buscar

### Frase exacta cuando no hay información suficiente:
> *"No encontré información suficiente en el corpus disponible para
> responder esta consulta con certeza. Te recomiendo consultar
> directamente el Semanario Judicial de la Federación en sjf.scjn.gob.mx
> o la versión vigente de la LFT en diputados.gob.mx"*

---

## 6. Chunking inteligente de documentos legales

Al procesar PDFs del corpus, respetar estas reglas de división:

### Regla de oro: nunca cortar un artículo a la mitad
```python
# ✅ Correcto: chunk termina al final del artículo
chunk = "Artículo 48. El trabajador podrá solicitar ante la Junta..."
# El chunk incluye el artículo completo

# ❌ Incorrecto: artículo cortado
chunk = "...podrá solicitar ante la Junta de Concil"
# El artículo queda truncado
```

### Tamaños de chunk por tipo de documento
| Documento | Chunk size | Overlap |
|---|---|---|
| LFT (artículos cortos) | 512 tokens | 50 tokens |
| Tesis SCJN (texto denso) | 768 tokens | 100 tokens |
| Criterios IMSS | 512 tokens | 75 tokens |
| Laudos (documentos largos) | 1024 tokens | 150 tokens |

### Metadatos obligatorios por chunk
```python
metadata = {
    "source": "LFT",           # Nombre de la fuente
    "article": "48",           # Número de artículo si aplica
    "date": "2024-01-01",      # Fecha de la versión del documento
    "type": "ley_federal",     # Tipo: ley_federal, jurisprudencia, criterio
    "mandatory": True          # True si es jurisprudencia obligatoria
}
```

---

## 7. Los 5 casos de uso más frecuentes

Prompts optimizados para cada caso:

### Caso 1: Cálculo de liquidación
```
Contexto recuperado: [chunks de LFT Arts. 48, 50, 89, 162]
Prompt: "Calcula la liquidación para un trabajador con [X] años de 
servicio, salario de [Y] pesos diarios, despedido sin causa justificada.
Muestra el desglose: 3 meses, 20 días por año, partes proporcionales.
Cita el artículo exacto de la LFT para cada concepto."
```

### Caso 2: Búsqueda de jurisprudencia
```
Contexto recuperado: [chunks de tesis SCJN relevantes]
Prompt: "Encuentra jurisprudencia o tesis sobre [tema]. Indica si es
obligatoria (J) o aislada (T), el número exacto y la época. Distingue
claramente entre criterios contradictorios si los hay."
```

### Caso 3: Redacción de contrato
```
Contexto recuperado: [chunks de LFT Arts. 24-28, 35-44]
Prompt: "Redacta un contrato individual de trabajo para [tipo de relación].
Incluye todas las cláusulas obligatorias según la LFT. Señala qué
artículo fundamenta cada cláusula."
```

### Caso 4: Criterios IMSS outsourcing
```
Contexto recuperado: [chunks de LSS, LFT reforma 2021, criterios IMSS]
Prompt: "Explica el régimen actual de subcontratación laboral en México
tras la reforma de 2021. Cita la LFT y los criterios normativos del IMSS
vigentes. Distingue entre subcontratación permitida y prohibida."
```

### Caso 5: Procedencia de reinstalación
```
Contexto recuperado: [chunks de LFT Arts. 48, 49, 123 CPEUM]
Prompt: "Analiza si procede la reinstalación en este caso: [descripción].
Considera: tipo de trabajador, antigüedad, causa de despido. Cita
jurisprudencia aplicable y señala si el patrón puede optar por
indemnización en lugar de reinstalar."
```

---

## 8. Control de calidad — checklist antes de devolver respuesta

Antes de entregar cualquier respuesta al usuario, verificar:

- [ ] ¿Cada afirmación legal tiene su cita con formato correcto?
- [ ] ¿Se distinguió claramente jurisprudencia J vs tesis T?
- [ ] ¿Se indicó la vigencia o fecha del documento fuente?
- [ ] ¿Se incluyó la advertencia de que es orientativo?
- [ ] ¿Ningún artículo fue inventado o combinado incorrectamente?
- [ ] ¿Si no había información suficiente, se dijo explícitamente?

Si algún punto falla → reescribir la respuesta antes de enviarla.
