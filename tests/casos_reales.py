"""
Casos de prueba reales con consultas que un abogado laboralista haría a Legis.

Cubre los 5 casos de uso prioritarios definidos en CLAUDE.md y desarrollados
en skills/legal-rag.md §7. Cada caso documenta:
- La consulta exacta del abogado
- Las citas que la respuesta DEBE incluir
- Errores comunes que NO deben aparecer (anti-alucinación)

Los asserts quedan pendientes hasta que el RAG esté implementado. El rol
QA legal los completa y ejecuta con `pytest tests/casos_reales.py`.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Pendiente: requiere RAG implementado y corpus indexado")
def test_caso_1_calculo_liquidacion() -> None:
    """
    Caso 1: Cálculo de liquidación por despido injustificado.

    Consulta esperada:
        "Calcula la liquidación de un trabajador con 5 años de antigüedad,
         salario diario de $800 MXN, despedido sin causa justificada."

    La respuesta debe citar:
        - [LFT Art. 48] (indemnización constitucional / reinstalación)
        - [LFT Art. 50] (3 meses + 20 días por año)
        - [LFT Art. 89] (integración del salario)
        - [LFT Art. 162] (prima de antigüedad)

    Debe incluir el desglose: 3 meses, 20 días por año, partes
    proporcionales de aguinaldo y vacaciones.
    """
    raise NotImplementedError("Implementar cuando RAG esté listo")


@pytest.mark.skip(reason="Pendiente: requiere RAG implementado y corpus indexado")
def test_caso_2_busqueda_jurisprudencia() -> None:
    """
    Caso 2: Búsqueda de jurisprudencia sobre rescisión de contrato.

    Consulta esperada:
        "¿Qué jurisprudencia existe sobre rescisión laboral por faltas
         injustificadas del trabajador?"

    La respuesta debe:
        - Distinguir explícitamente J (jurisprudencia obligatoria) de T (tesis aislada)
        - Incluir número exacto y época (10a. / 11a.)
        - Si hay criterios contradictorios, presentarlos por separado
    """
    raise NotImplementedError("Implementar cuando RAG esté listo")


@pytest.mark.skip(reason="Pendiente: requiere RAG implementado y corpus indexado")
def test_caso_3_redaccion_contrato() -> None:
    """
    Caso 3: Redacción de contrato individual de trabajo.

    Consulta esperada:
        "Redacta un contrato individual de trabajo por tiempo indeterminado
         para un puesto administrativo."

    La respuesta debe citar:
        - [LFT Art. 24] (forma escrita)
        - [LFT Art. 25] (contenido obligatorio del contrato)
        - [LFT Arts. 35-40] (modalidades de contrato)

    Cada cláusula del contrato debe señalar el artículo que la fundamenta.
    """
    raise NotImplementedError("Implementar cuando RAG esté listo")


@pytest.mark.skip(reason="Pendiente: requiere RAG implementado y corpus indexado")
def test_caso_4_imss_outsourcing() -> None:
    """
    Caso 4: Régimen de subcontratación tras la reforma 2021.

    Consulta esperada:
        "¿Puede una empresa contratar personal vía outsourcing en México hoy?"

    La respuesta debe:
        - Citar la LFT reformada en 2021 (Arts. 12-15)
        - Citar criterios IMSS vigentes sobre REPSE
        - Distinguir entre subcontratación permitida (servicios especializados)
          y prohibida (suministro de personal)
        - NO afirmar que el outsourcing está "totalmente prohibido"
          (alucinación frecuente)
    """
    raise NotImplementedError("Implementar cuando RAG esté listo")


@pytest.mark.skip(reason="Pendiente: requiere RAG implementado y corpus indexado")
def test_caso_5_procedencia_reinstalacion() -> None:
    """
    Caso 5: Procedencia de reinstalación tras despido.

    Consulta esperada:
        "Trabajador de confianza con 8 años de antigüedad fue despedido
         sin causa. ¿Procede reinstalación o el patrón puede pagar?"

    La respuesta debe:
        - Citar [LFT Art. 48] (derecho a elegir reinstalación o indemnización)
        - Citar [LFT Art. 49] (casos en que el patrón puede negarse a reinstalar:
          trabajador de confianza, doméstico, eventual, < 1 año, contacto
          directo con el patrón)
        - Citar jurisprudencia aplicable sobre trabajadores de confianza
        - Concluir si en este caso el patrón puede optar por indemnización
    """
    raise NotImplementedError("Implementar cuando RAG esté listo")


def test_anti_alucinacion_corpus_vacio() -> None:
    """
    Verifica que cuando no hay información en el corpus, el modelo responde
    con la frase exacta definida en skills/legal-rag.md §5, no inventa.

    Este test se puede ejecutar incluso antes de tener corpus real, usando
    una consulta intencionalmente fuera de dominio.
    """
    pytest.skip("Pendiente: requiere RAG implementado")
