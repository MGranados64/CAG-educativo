"""
evals.py — El "profesor que corrige el examen" (LLM-as-a-Judge)
=================================================================

🎓 LA ANALOGÍA
--------------
En cag_core.py un estudiante (el modelo) memorizó un libro y contesta
preguntas. Pero ¿cómo sabemos si contestó bien? Aquí entra un PROFESOR
(el mismo modelo local, con instrucciones distintas) que corrige cada
respuesta con una rúbrica, igual que en un examen. A esto se le llama
"LLM-as-a-Judge": usar un modelo de lenguaje como juez.

Tenemos DOS jueces, cada uno con una sola pregunta en mente
(un juez con una sola tarea se equivoca menos que uno que evalúa todo):

  ⚖️ Juez de VERACIDAD (Faithfulness)
     "¿Todo lo que dijo el estudiante aparece en el libro, o inventó algo?"
     Inventar datos que suenan creíbles se llama ALUCINACIÓN.

  🎯 Juez de RELEVANCIA (Answer Relevancy)
     "¿Contestó lo que se le preguntó, o se fue por las ramas?"

📐 INSPIRACIÓN: LAS MÉTRICAS DE RAGAS (Es et al., EACL 2024)
-----------------------------------------------------------
  • Faithfulness: se divide la respuesta en afirmaciones atómicas (frases con
    un solo hecho) y se verifica cada una contra el contexto.
        puntaje = afirmaciones respaldadas / afirmaciones totales
    Aquí hacemos exactamente eso, y además pedimos al juez la CITA TEXTUAL
    del documento que respalda cada afirmación (transparencia total).

  • Answer Relevancy: RAGas genera preguntas a partir de la respuesta
    ("preguntas inversas") y mide con embeddings qué tan parecidas son a la
    pregunta original. Aquí el juez genera esas preguntas inversas como
    EVIDENCIA visible y luego califica con una rúbrica de 1 a 5.

⚠️ UNA ADVERTENCIA HONESTA
--------------------------
El juez es el MISMO modelo que respondió. Los estudios sobre LLM-as-a-Judge
(Zheng et al., NeurIPS 2023) muestran que un modelo tiende a ser indulgente
consigo mismo ("sesgo de autopromoción"). Por eso cada veredicto muestra su
evidencia y su razonamiento: la última palabra la tienes tú.

🔒 SALIDA ESTRUCTURADA
----------------------
Para que el juez no responda "con palabras bonitas" sino con datos que el
programa pueda leer, le pasamos a Ollama un "esquema" JSON en el parámetro
`format`: el modelo solo puede escribir texto que encaje en ese molde.

Pruébalo desde la terminal (con Ollama encendido):

    python evals.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from cag_core import MedicionDeTiempos, MotorCAG

# ════════════════════════════════════════════════════════════════════════════
# ⚖️ JUEZ DE VERACIDAD
# ════════════════════════════════════════════════════════════════════════════

# Este juez recibe el MISMO conocimiento que el chat, al inicio del prompt: así
# reutiliza el "libro" ya memorizado en la caché en vez de leerlo de nuevo.
INSTRUCCIONES_JUEZ_VERACIDAD = """\
Eres un auditor de veracidad. Tu ÚNICA fuente de verdad es la biblioteca de \
conocimiento de arriba (<biblioteca_de_conocimiento>). No uses tu conocimiento \
general: aunque un dato sea cierto en el mundo real, si no está en los \
documentos, NO está respaldado.

Recibirás una pregunta y la respuesta que dio un asistente. Haz lo siguiente:
1. Divide la respuesta en afirmaciones atómicas: frases cortas que afirman un \
solo hecho verificable. Usa SOLO lo que dice la respuesta; no agregues \
afirmaciones que no estén en ella. Ignora saludos y cortesías. Si la respuesta \
dice que algo NO aparece en los documentos, esa también es una afirmación: está \
respaldada si efectivamente no aparece.
2. Para cada afirmación, busca en los documentos la cita textual que la \
respalda (copia el fragmento exacto, máximo dos frases). Si no existe, deja la \
evidencia vacía.
3. Explica en una frase tu razonamiento y después da el veredicto: \
respaldada = true solo si la evidencia la confirma por completo (se permiten \
cálculos sencillos con datos del documento, como sumar días). Si contradice el \
documento o agrega datos que no están, respaldada = false.
4. Escribe un resumen de una o dos frases que entienda una persona sin \
conocimientos técnicos.
Responde en español."""

# El "molde" exacto que debe llenar el juez. Fíjate en el ORDEN de los campos:
# primero la evidencia, luego el razonamiento y AL FINAL el veredicto. Así lo
# obligamos a buscar la prueba antes de decidir, no a justificar una decisión
# ya tomada.
ESQUEMA_VERACIDAD = {
    "type": "object",
    "properties": {
        "afirmaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "afirmacion": {"type": "string"},
                    "evidencia": {"type": "string"},
                    "razonamiento": {"type": "string"},
                    "respaldada": {"type": "boolean"},
                },
                "required": ["afirmacion", "evidencia", "razonamiento", "respaldada"],
            },
        },
        "resumen": {"type": "string"},
    },
    "required": ["afirmaciones", "resumen"],
}

# ════════════════════════════════════════════════════════════════════════════
# 🎯 JUEZ DE RELEVANCIA
# ════════════════════════════════════════════════════════════════════════════

# Este juez NO ve el documento, a propósito: no le importa si la respuesta es
# verdadera (de eso se encarga el otro juez), solo si contesta lo preguntado.
INSTRUCCIONES_JUEZ_RELEVANCIA = """\
Eres un auditor de relevancia. NO juzgas si la respuesta es verdadera (de eso \
se encarga otro auditor); solo juzgas si responde DIRECTAMENTE a lo que se \
preguntó.

Recibirás una pregunta y la respuesta que dio un asistente. Haz lo siguiente:
1. Mirando SOLO la respuesta, escribe exactamente 3 preguntas que esa \
respuesta estaría contestando ("preguntas inversas"). Si se parecen a la \
pregunta original, la respuesta es relevante.
2. Enumera las partes de la respuesta que no aportan nada a la pregunta \
(relleno, temas ajenos). Deja la lista vacía si no hay.
3. Razona brevemente y asigna una puntuación con esta rúbrica:
   5 = responde directa y completamente, sin relleno.
   4 = responde, pero con algo de información innecesaria o le falta un \
detalle menor.
   3 = responde solo en parte, o reconoce con honestidad que no tiene la \
información (no inventa, pero tampoco entrega lo pedido).
   2 = toca el tema, pero no contesta lo que se preguntó.
   1 = no responde, cambia de tema o es incomprensible.
Responde en español."""

ESQUEMA_RELEVANCIA = {
    "type": "object",
    "properties": {
        "preguntas_inversas": {"type": "array", "items": {"type": "string"}},
        "partes_irrelevantes": {"type": "array", "items": {"type": "string"}},
        "razonamiento": {"type": "string"},
        "puntuacion": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["preguntas_inversas", "partes_irrelevantes", "razonamiento", "puntuacion"],
}

# ════════════════════════════════════════════════════════════════════════════
# 📦 RESULTADOS
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ResultadoVeracidad:
    puntaje: float  # 0.0 (todo inventado) a 1.0 (todo respaldado)
    afirmaciones: list[dict] = field(default_factory=list)
    resumen: str = ""
    medicion: MedicionDeTiempos | None = None
    error: str | None = None  # si el juez no pudo evaluar

    @property
    def respaldadas(self) -> int:
        return sum(1 for afirmacion in self.afirmaciones if afirmacion["respaldada"])


@dataclass
class ResultadoRelevancia:
    puntaje: float  # 0.0 (no contesta) a 1.0 (contesta directo y completo)
    puntuacion_1_a_5: int = 0
    preguntas_inversas: list[str] = field(default_factory=list)
    partes_irrelevantes: list[str] = field(default_factory=list)
    razonamiento: str = ""
    medicion: MedicionDeTiempos | None = None
    error: str | None = None


@dataclass
class InformeDeEvaluacion:
    veracidad: ResultadoVeracidad
    relevancia: ResultadoRelevancia
    segundos: float


def semaforo(puntaje: float) -> str:
    """Convierte un puntaje de 0 a 1 en una etiqueta fácil de leer."""
    if puntaje >= 0.9:
        return "🟢 Excelente"
    if puntaje >= 0.6:
        return "🟡 Mejorable"
    return "🔴 Problema"


# ════════════════════════════════════════════════════════════════════════════
# 🧑‍⚖️ LLAMAR AL JUEZ
# ════════════════════════════════════════════════════════════════════════════


class JuezSinVeredicto(RuntimeError):
    """El juez no devolvió una evaluación utilizable."""


def _llamar_al_juez(
    motor: MotorCAG, system: str, pregunta: str, respuesta: str, esquema: dict
) -> tuple[dict, MedicionDeTiempos]:
    """
    Envía la pregunta y la respuesta a auditar, y exige que el juez conteste
    rellenando el `esquema` JSON. Devuelve los datos ya convertidos a un
    diccionario de Python, junto con sus tiempos.
    """
    resultado = motor.cliente.chat(
        **motor.parametros_base(),  # mismo modelo, mesa y razonamiento que el chat
        format=esquema,  # 👈 "salida estructurada": el JSON DEBE encajar en el molde
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"<pregunta>\n{pregunta}\n</pregunta>\n\n"
                    f"<respuesta_a_auditar>\n{respuesta}\n</respuesta_a_auditar>"
                ),
            },
        ],
    )
    if resultado.done_reason == "length":
        raise JuezSinVeredicto("La evaluación no cupo en la mesa del modelo y quedó cortada.")
    try:
        datos = json.loads(resultado.message.content)
    except json.JSONDecodeError as error:
        raise JuezSinVeredicto(f"El juez no devolvió un JSON válido ({error}).") from error

    faltantes = [campo for campo in esquema["required"] if campo not in datos]
    if faltantes:
        raise JuezSinVeredicto(f"Al veredicto del juez le faltan campos: {', '.join(faltantes)}.")
    return datos, MedicionDeTiempos.desde_respuesta(resultado)


def evaluar_veracidad(motor: MotorCAG, pregunta: str, respuesta: str) -> ResultadoVeracidad:
    """⚖️ ¿Cada afirmación de la respuesta está respaldada por el documento?"""
    # El conocimiento va primero (vía prompt_de_sistema) para reutilizar la caché.
    system = motor.prompt_de_sistema(INSTRUCCIONES_JUEZ_VERACIDAD)
    try:
        datos, medicion = _llamar_al_juez(motor, system, pregunta, respuesta, ESQUEMA_VERACIDAD)
    except JuezSinVeredicto as error:
        return ResultadoVeracidad(puntaje=0.0, error=str(error))

    afirmaciones = datos["afirmaciones"]
    if afirmaciones:
        # 🧮 La fórmula de RAGas, calculada por Python (no por el modelo), para
        # que el número sea exacto y reproducible.
        respaldadas = sum(1 for afirmacion in afirmaciones if afirmacion["respaldada"])
        puntaje = respaldadas / len(afirmaciones)
    else:
        # Una respuesta sin afirmaciones verificables no puede alucinar.
        puntaje = 1.0

    return ResultadoVeracidad(
        puntaje=puntaje, afirmaciones=afirmaciones, resumen=datos["resumen"], medicion=medicion
    )


def evaluar_relevancia(motor: MotorCAG, pregunta: str, respuesta: str) -> ResultadoRelevancia:
    """🎯 ¿La respuesta contesta directamente lo que se preguntó?"""
    try:
        datos, medicion = _llamar_al_juez(
            motor, INSTRUCCIONES_JUEZ_RELEVANCIA, pregunta, respuesta, ESQUEMA_RELEVANCIA
        )
    except JuezSinVeredicto as error:
        return ResultadoRelevancia(puntaje=0.0, error=str(error))

    puntuacion = datos["puntuacion"]
    return ResultadoRelevancia(
        # Convertimos la escala 1–5 a 0–1:  1→0.00, 2→0.25, 3→0.50, 4→0.75, 5→1.00
        puntaje=(puntuacion - 1) / 4,
        puntuacion_1_a_5=puntuacion,
        preguntas_inversas=datos["preguntas_inversas"],
        partes_irrelevantes=datos["partes_irrelevantes"],
        razonamiento=datos["razonamiento"],
        medicion=medicion,
    )


def evaluar_respuesta(motor: MotorCAG, pregunta: str, respuesta: str) -> InformeDeEvaluacion:
    """
    Corre los dos jueces, uno detrás del otro. En tu computadora hay una sola
    tarjeta gráfica: dos profesores corrigiendo a la vez solo se estorbarían.
    """
    inicio = time.perf_counter()
    veracidad = evaluar_veracidad(motor, pregunta, respuesta)
    relevancia = evaluar_relevancia(motor, pregunta, respuesta)
    return InformeDeEvaluacion(veracidad, relevancia, segundos=time.perf_counter() - inicio)


# ════════════════════════════════════════════════════════════════════════════
# 🧪 ¿QUIÉN EVALÚA AL EVALUADOR?   python evals.py
#
# Antes de confiar en un juez, hay que probarlo con respuestas cuya
# calificación ya conocemos. Si el juez aprueba una alucinación evidente,
# el juez es el que falla.
# ════════════════════════════════════════════════════════════════════════════

CASOS_DE_PRUEBA = [
    {
        "nombre": "Respuesta correcta",
        "pregunta": "¿Cuántas horas fermenta el café?",
        "respuesta": "El café fermenta 36 horas en tanques de concreto.",
        "se_espera": "veracidad alta y relevancia alta",
    },
    {
        "nombre": "Alucinación",
        "pregunta": "¿Cuántas horas fermenta el café?",
        "respuesta": (
            "El café fermenta 72 horas en tanques de acero importados de Italia, "
            "según la norma ISO-9001 de la cooperativa."
        ),
        "se_espera": "veracidad baja (datos inventados) y relevancia alta",
    },
    {
        "nombre": "Respuesta que se va por las ramas",
        "pregunta": "¿Cuántos días de vacaciones tiene un empleado en su primer año?",
        "respuesta": "La tienda La Taza de Ernestina cierra los martes.",
        "se_espera": "veracidad alta (el dato es cierto) pero relevancia baja",
    },
]

if __name__ == "__main__":
    from cag_core import Documento

    ruta_ejemplo = Path(__file__).parent / "datos" / "manual_cooperativa_nube_alta.md"
    motor = MotorCAG()
    motor.cargar_conocimiento([Documento(ruta_ejemplo.name, ruta_ejemplo.read_text("utf-8"))])

    for caso in CASOS_DE_PRUEBA:
        informe = evaluar_respuesta(motor, caso["pregunta"], caso["respuesta"])
        print(f"\n🧪 {caso['nombre']}  (se espera: {caso['se_espera']})")
        if informe.veracidad.error:
            print(f"   ⚖️ Veracidad: sin veredicto ({informe.veracidad.error})")
        else:
            print(f"   ⚖️ Veracidad:  {informe.veracidad.puntaje:.0%}  {semaforo(informe.veracidad.puntaje)}")
            for afirmacion in informe.veracidad.afirmaciones:
                icono = "✅" if afirmacion["respaldada"] else "❌"
                print(f"      {icono} {afirmacion['afirmacion']}")
        if informe.relevancia.error:
            print(f"   🎯 Relevancia: sin veredicto ({informe.relevancia.error})")
        else:
            print(f"   🎯 Relevancia: {informe.relevancia.puntaje:.0%}  {semaforo(informe.relevancia.puntaje)}")
            print(f"      {informe.relevancia.razonamiento}")
        print(f"   ⏱️ {informe.segundos:.1f} s")
