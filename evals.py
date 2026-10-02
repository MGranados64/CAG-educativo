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

Y dos jueces más para el banco de pruebas CAG vs RAG (comparacion.py), que
usan preguntas cuya respuesta correcta ya conocemos:

  ✅ Juez de EXACTITUD (Answer Correctness)
     "¿Coincide con la respuesta de referencia?"

  🔎 Juez de RECUPERACIÓN (Context Recall), solo para RAG
     "¿El buscador trajo los fragmentos necesarios para responder?"

📐 INSPIRACIÓN: LAS MÉTRICAS DE RAGAS (Es et al., EACL 2024)
-----------------------------------------------------------
  • Faithfulness: se divide la respuesta en afirmaciones atómicas (frases con
    un solo hecho) y se verifica cada una contra el contexto.
        puntaje = afirmaciones respaldadas / afirmaciones totales
    Aquí hacemos exactamente eso, y además pedimos al juez la CITA TEXTUAL
    del documento que respalda cada afirmación, y Python comprueba que esa
    cita exista de verdad (ver cita_verificada).

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
import re
import time
from dataclasses import dataclass, field

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
afirmaciones que no estén en ella. Ignora saludos, cortesías y las menciones de \
la fuente (como "según la sección 4 del manual").
2. Clasifica cada afirmación: tipo = "ausencia" si dice que algo NO aparece en \
los documentos (está respaldada si efectivamente no aparece); tipo = "hecho" en \
cualquier otro caso.
3. Para cada hecho, copia de los documentos la cita textual que lo respalda: el \
fragmento exacto (máximo dos frases) que contiene el dato concreto (la cifra, el \
nombre o el suceso). Nunca copies etiquetas como <documento>. Si no existe, deja \
la evidencia vacía.
4. Explica en una frase tu razonamiento y después da el veredicto: \
respaldada = true solo si la evidencia confirma el hecho por completo (se \
permiten cálculos sencillos con datos del documento, como sumar días). Si tu \
razonamiento dice que el documento no menciona el dato, respaldada = false.
5. Escribe un resumen de una o dos frases que entienda una persona sin \
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
                    "tipo": {"type": "string", "enum": ["hecho", "ausencia"]},
                    "evidencia": {"type": "string"},
                    "razonamiento": {"type": "string"},
                    "respaldada": {"type": "boolean"},
                },
                "required": ["afirmacion", "tipo", "evidencia", "razonamiento", "respaldada"],
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
# ✅ JUEZ DE EXACTITUD (para el banco de pruebas "cara a cara")
# ════════════════════════════════════════════════════════════════════════════

# Cuando SÍ conocemos la respuesta correcta (las preguntas de referencia de
# datos/*/preguntas.json), podemos ir más allá de la veracidad: ¿acertó?
# Equivale a la métrica "answer correctness" de RAGas.
INSTRUCCIONES_JUEZ_EXACTITUD = """\
Eres un profesor que compara la respuesta de un asistente con la respuesta \
correcta (la "respuesta de referencia").

Reglas:
- Compara el SIGNIFICADO, no las palabras exactas ni el idioma.
- Las cifras, fechas y nombres deben coincidir con la referencia.
- "correcta": contiene todo lo esencial de la referencia y no la contradice.
- "parcial": acierta solo una parte o le falta algo importante.
- "incorrecta": contradice la referencia, no contesta o inventa la respuesta.
- La información ADICIONAL que no contradice la referencia NO es un error: no \
la penalices (si es inventada, de eso se encarga otro auditor, el de veracidad).
- Si la referencia dice que la información no aparece en los documentos, la \
respuesta es correcta solo si también dice que no tiene esa información; si \
inventa un dato, es incorrecta.
Explica brevemente tu razonamiento antes del veredicto. Responde en español."""

ESQUEMA_EXACTITUD = {
    "type": "object",
    "properties": {
        "razonamiento": {"type": "string"},
        "veredicto": {"type": "string", "enum": ["correcta", "parcial", "incorrecta"]},
    },
    "required": ["razonamiento", "veredicto"],
}
PUNTAJE_DE_EXACTITUD = {"correcta": 1.0, "parcial": 0.5, "incorrecta": 0.0}

# ════════════════════════════════════════════════════════════════════════════
# 🔎 JUEZ DE RECUPERACIÓN (solo para RAG)
# ════════════════════════════════════════════════════════════════════════════

# RAG puede fallar por dos motivos muy distintos: el BUSCADOR no trajo el
# fragmento correcto, o el MODELO lo tenía y aun así respondió mal. Este juez
# separa ambos casos. Equivale a la métrica "context recall" de RAGas.
INSTRUCCIONES_JUEZ_RECUPERACION = """\
Eres un auditor del buscador de un sistema RAG. Recibirás una pregunta, su \
respuesta correcta y los fragmentos de texto que el buscador recuperó.

Decide si esos fragmentos contienen la información necesaria para llegar a la \
respuesta correcta:
- "completa": con esos fragmentos se puede responder todo.
- "parcial": contienen solo una parte de la información necesaria.
- "nada": no contienen la información necesaria.
Explica qué dato falta, si falta alguno, antes del veredicto. Responde en español."""

ESQUEMA_RECUPERACION = {
    "type": "object",
    "properties": {
        "razonamiento": {"type": "string"},
        "veredicto": {"type": "string", "enum": ["completa", "parcial", "nada"]},
    },
    "required": ["razonamiento", "veredicto"],
}
PUNTAJE_DE_RECUPERACION = {"completa": 1.0, "parcial": 0.5, "nada": 0.0}

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
class ResultadoConVeredicto:
    """Resultado de los jueces de exactitud y de recuperación."""

    puntaje: float  # 1.0, 0.5 o 0.0
    veredicto: str = ""
    razonamiento: str = ""
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


def _llamar_al_juez(motor: MotorCAG, system: str, contenido: str, esquema: dict) -> tuple[dict, MedicionDeTiempos]:
    """
    Envía al juez el material a revisar (`contenido`) y exige que conteste
    rellenando el `esquema` JSON. Devuelve los datos ya convertidos a un
    diccionario de Python, junto con sus tiempos.
    """
    resultado = motor.cliente.chat(
        **motor.parametros_base(),  # mismo modelo, mesa y razonamiento que el chat
        format=esquema,  # 👈 "salida estructurada": el JSON DEBE encajar en el molde
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": contenido},
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


def _pregunta_y_respuesta(pregunta: str, respuesta: str) -> str:
    """El material que reciben los jueces de veracidad y relevancia."""
    return (
        f"<pregunta>\n{pregunta}\n</pregunta>\n\n"
        f"<respuesta_a_auditar>\n{respuesta}\n</respuesta_a_auditar>"
    )


# Para comprobar las citas: qué fracción de los pares de palabras de la cita
# debe existir en los documentos, y palabras demasiado comunes para decir si
# una cita "tiene que ver" con una afirmación.
COINCIDENCIA_MINIMA_DE_LA_CITA = 0.7
PALABRAS_COMUNES = {
    "sobre", "entre", "desde", "hasta", "según", "donde", "cuando", "tiene", "tienen",
    "puede", "pueden", "debe", "deben", "están", "estos", "estas", "otros", "otras",
    "cada", "todos", "todas", "documento", "documentos", "información", "respuesta",
    "about", "which", "there", "their", "these", "those", "where", "would", "could",
}


def _palabras(texto: str) -> list[str]:
    """Las palabras y cifras de un texto, en minúsculas y sin formato ni puntuación."""
    return re.findall(r"[^\W_]+", texto.lower())


def cita_verificada(evidencia: str, conocimiento: str, afirmacion: str = "") -> bool:
    """
    🔍 No le creemos al juez sin pruebas: Python comprueba su cita.

    Los modelos rara vez copian letra por letra: unen renglones de una lista o
    de una tabla, quitan negritas, cambian comillas... Por eso no exigimos una
    copia exacta, sino tres cosas que una cita inventada no cumple:

      1. EXISTE: al menos el 70 % de sus pares de palabras seguidas aparecen
         en los documentos ("fermentación de 72 horas" no aparece).
      2. SUS CIFRAS EXISTEN: todos sus números están en los documentos.
      3. TIENE QUE VER: comparte al menos una palabra importante (o una cifra)
         con la afirmación. Copiar un texto real que no habla del tema no vale.

    Tampoco vale copiar etiquetas (<documento ...>) en lugar de contenido.
    """
    if not evidencia.strip() or "<" in evidencia:
        return False
    palabras_cita = _palabras(evidencia)
    palabras_documentos = _palabras(conocimiento)
    if len(palabras_cita) < 2:
        return False

    pares_documentos = set(zip(palabras_documentos, palabras_documentos[1:]))
    pares_cita = list(zip(palabras_cita, palabras_cita[1:]))
    coincidencia = sum(par in pares_documentos for par in pares_cita) / len(pares_cita)
    if coincidencia < COINCIDENCIA_MINIMA_DE_LA_CITA:
        return False

    cifras_documentos = {palabra for palabra in palabras_documentos if palabra.isdigit()}
    if any(palabra.isdigit() and palabra not in cifras_documentos for palabra in palabras_cita):
        return False

    importantes = {
        palabra
        for palabra in _palabras(afirmacion)
        if palabra.isdigit() or (len(palabra) >= 5 and palabra not in PALABRAS_COMUNES)
    }
    return not importantes or bool(importantes & set(palabras_cita))


def evaluar_veracidad(motor: MotorCAG, pregunta: str, respuesta: str) -> ResultadoVeracidad:
    """⚖️ ¿Cada afirmación de la respuesta está respaldada por el documento?"""
    # El conocimiento va primero (vía prompt_de_sistema) para reutilizar la caché.
    system = motor.prompt_de_sistema(INSTRUCCIONES_JUEZ_VERACIDAD)
    try:
        datos, medicion = _llamar_al_juez(
            motor, system, _pregunta_y_respuesta(pregunta, respuesta), ESQUEMA_VERACIDAD
        )
    except JuezSinVeredicto as error:
        return ResultadoVeracidad(puntaje=0.0, error=str(error))

    afirmaciones = datos["afirmaciones"]
    for afirmacion in afirmaciones:
        # Un HECHO solo cuenta como respaldado si su cita existe de verdad. (Las
        # afirmaciones de AUSENCIA, "no aparece en los documentos", no llevan cita.)
        if afirmacion["tipo"] == "hecho" and afirmacion["respaldada"]:
            if not cita_verificada(afirmacion["evidencia"], motor.conocimiento_en_cache, afirmacion["afirmacion"]):
                afirmacion["respaldada"] = False
                afirmacion["razonamiento"] += (
                    " ⚠️ Python no encontró esa cita textual en los documentos, así que no cuenta como respaldo."
                )
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
            motor, INSTRUCCIONES_JUEZ_RELEVANCIA, _pregunta_y_respuesta(pregunta, respuesta), ESQUEMA_RELEVANCIA
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


def evaluar_exactitud(
    motor: MotorCAG, pregunta: str, respuesta: str, respuesta_esperada: str
) -> ResultadoConVeredicto:
    """✅ ¿La respuesta coincide con la respuesta de referencia?"""
    contenido = (
        f"<pregunta>\n{pregunta}\n</pregunta>\n\n"
        f"<respuesta_de_referencia>\n{respuesta_esperada}\n</respuesta_de_referencia>\n\n"
        f"<respuesta_a_calificar>\n{respuesta}\n</respuesta_a_calificar>"
    )
    try:
        datos, _ = _llamar_al_juez(motor, INSTRUCCIONES_JUEZ_EXACTITUD, contenido, ESQUEMA_EXACTITUD)
    except JuezSinVeredicto as error:
        return ResultadoConVeredicto(puntaje=0.0, error=str(error))
    return ResultadoConVeredicto(
        puntaje=PUNTAJE_DE_EXACTITUD[datos["veredicto"]],
        veredicto=datos["veredicto"],
        razonamiento=datos["razonamiento"],
    )


def evaluar_recuperacion(
    motor: MotorCAG, pregunta: str, respuesta_esperada: str, fragmentos: list[str]
) -> ResultadoConVeredicto:
    """🔎 ¿El buscador de RAG trajo los fragmentos necesarios para responder?"""
    fragmentos_numerados = "\n\n".join(
        f'<fragmento numero="{numero}">\n{texto}\n</fragmento>' for numero, texto in enumerate(fragmentos, start=1)
    )
    contenido = (
        f"<pregunta>\n{pregunta}\n</pregunta>\n\n"
        f"<respuesta_correcta>\n{respuesta_esperada}\n</respuesta_correcta>\n\n"
        f"<fragmentos_recuperados>\n{fragmentos_numerados}\n</fragmentos_recuperados>"
    )
    try:
        datos, _ = _llamar_al_juez(motor, INSTRUCCIONES_JUEZ_RECUPERACION, contenido, ESQUEMA_RECUPERACION)
    except JuezSinVeredicto as error:
        return ResultadoConVeredicto(puntaje=0.0, error=str(error))
    return ResultadoConVeredicto(
        puntaje=PUNTAJE_DE_RECUPERACION[datos["veredicto"]],
        veredicto=datos["veredicto"],
        razonamiento=datos["razonamiento"],
    )


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
    from cag_core import CARPETA_CORPUS_DE_EJEMPLO, cargar_documentos_de_carpeta

    motor = MotorCAG()
    motor.cargar_conocimiento(cargar_documentos_de_carpeta(CARPETA_CORPUS_DE_EJEMPLO))

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
