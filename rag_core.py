"""
rag_core.py — El rival: RAG (Retrieval-Augmented Generation)
=============================================================

🎓 LA ANALOGÍA
--------------
Volvamos al examen. En RAG el estudiante NO memoriza el libro. Antes del examen
alguien recorta el libro en fichas (fragmentos) y las guarda en un fichero. En
cada pregunta, un bibliotecario (el buscador) busca las fichas que PARECEN más
relacionadas y solo esas se le entregan al estudiante.

  ✅ Ventaja: el estudiante lee poquito en cada pregunta, y el fichero puede ser
     enorme (millones de fichas).
  ⚠️ Riesgo: si el bibliotecario trae las fichas equivocadas, el estudiante no
     puede responder bien, por muy listo que sea.

🧭 CUATRO BIBLIOTECARIOS (ESTRATEGIAS DE BÚSQUEDA)
-------------------------------------------------
  🔎 Clásico     Busca por SIGNIFICADO (embeddings) y entrega las k fichas más
                 parecidas a la pregunta.
  🔀 Híbrido     Busca por significado Y por PALABRAS EXACTAS (BM25), junta las
                 dos listas (RRF) y le pide al modelo que elija las k fichas más
                 útiles (reranking).
  👪 Padre-hijo  Busca fichas pequeñas (precisas), pero entrega a su "padre": el
                 documento completo, o solo su sección si el documento es grande.
  🤖 Agéntico    El modelo divide la pregunta en búsquedas, busca, revisa si le
                 falta algo y, si hace falta, busca otra vez antes de responder.

Las cuatro comparten el mismo fichero (las mismas fichas y embeddings): se puede
cambiar de bibliotecario sin volver a indexar.

🧩 CÓMO FLUYE LA INFORMACIÓN EN ESTE ARCHIVO
--------------------------------------------
  1. MotorRAG.indexar()   → trocea los documentos en fichas, calcula el
     "embedding" de cada una (una lista de números que representa su
     significado) y prepara el índice de palabras (BM25).
  2. MotorRAG.buscar()    → aplica la estrategia elegida y devuelve las fichas,
     cuánto tardó y los PASOS que siguió (para que veas qué hizo).
  3. MotorRAG.preguntar_en_vivo() → manda al MISMO modelo de chat que usa CAG
     solo esas fichas y la pregunta.

⚖️ UNA PELEA JUSTA CONTRA CAG
-----------------------------
Para comparar con CAG (comparacion.py) usamos el mismo modelo de chat, la misma
mesa (num_ctx), el mismo razonamiento, temperatura 0 y reglas equivalentes. La
ÚNICA diferencia: CAG ve todos los documentos; RAG, solo lo que trae el buscador.

Pruébalo desde la terminal (con Ollama encendido):

    python rag_core.py
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import ollama

from cag_core import (
    CARPETA_CORPUS_DE_EJEMPLO,
    DURACIONES_DE_CACHE,
    MAXIMO_DE_TOKENS_POR_RESPUESTA,
    MODELO_RECOMENDADO,
    TEMPERATURA,
    Documento,
    RespuestaDelModelo,
    RespuestaEnVivo,
    calcular_tamano_de_mesa,
    cargar_documentos_de_carpeta,
    describir_modelo,
    nivel_de_razonamiento,
)

# ════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

# El modelo que convierte textos en embeddings. Es pequeño (unos 270 MB) y muy
# rápido, pero fue entrenado sobre todo en inglés: con textos en español busca
# peor. Un modelo multilingüe puede mejorar la búsqueda en español.
MODELO_DE_EMBEDDINGS = os.environ.get("CAG_MODELO_EMBEDDINGS", "nomic-embed-text")

# Tamaño de cada ficha. Fichas pequeñas = búsquedas más precisas, pero con
# menos contexto alrededor; fichas grandes = más contexto, pero más "ruido".
CARACTERES_POR_FRAGMENTO = 1_200

# Cuando un párrafo es más largo que una ficha, lo partimos en trozos que se
# enciman un poco, para no cortar una idea justo a la mitad.
SOLAPAMIENTO_EN_CARACTERES = 200

# Cuántas fichas trae el bibliotecario en cada pregunta (el famoso "top-k").
FRAGMENTOS_A_RECUPERAR = 4

# Híbrido: cuántas candidatas de la fusión le enseñamos al modelo para que elija.
CANDIDATAS_PARA_REORDENAR = 10

# Padre-hijo: un documento más largo que esto no se entrega completo, solo la
# sección de la ficha encontrada (así la estrategia sirve en bibliotecas grandes).
TAMANO_MAXIMO_DEL_PADRE = 6_000

# Agéntico: cuántas búsquedas puede planear al inicio, cuántas veces puede
# revisar "¿me falta algo?" y cuántas fichas puede entregar como máximo.
BUSQUEDAS_INICIALES_DEL_AGENTE = 3
REVISIONES_DEL_AGENTE = 1
FICHAS_MAXIMAS_DEL_AGENTE = 8


@dataclass(frozen=True)
class Estrategia:
    clave: str
    nombre: str
    descripcion: str


ESTRATEGIAS = {
    estrategia.clave: estrategia
    for estrategia in [
        Estrategia(
            "clasico",
            "🔎 Clásico",
            "Busca por significado (embeddings) y entrega las k fichas más parecidas a la pregunta.",
        ),
        Estrategia(
            "hibrido",
            "🔀 Híbrido",
            "Busca por significado y por palabras exactas (BM25), junta las dos listas (RRF) "
            "y el modelo elige las k fichas más útiles (reranking).",
        ),
        Estrategia(
            "padre_hijo",
            "👪 Padre-hijo",
            "Busca fichas pequeñas, pero entrega su documento completo (o solo su sección, "
            "si el documento es grande).",
        ),
        Estrategia(
            "agentico",
            "🤖 Agéntico",
            "El modelo divide la pregunta en búsquedas, busca (híbrido), revisa si le falta "
            "algo y, si hace falta, busca otra vez.",
        ),
    ]
}

INSTRUCCIONES_RAG = """\
Eres un asistente que responde preguntas usando EXCLUSIVAMENTE los fragmentos \
de documentos que aparecen dentro de <fragmentos_recuperados> en el mensaje del \
usuario. Un buscador los eligió por ser los más parecidos a la pregunta.

Reglas:
- Responde solo con información que esté en esos fragmentos. No uses tu \
conocimiento general ni supongas datos que no aparecen.
- Si la respuesta no está en los fragmentos, dilo con claridad (por ejemplo: \
"Esa información no aparece en los documentos cargados").
- Cuando sea útil, indica de qué documento o sección sale el dato.
- Responde en el mismo idioma de la pregunta, de forma clara y breve."""

# ── Instrucciones para las tareas de apoyo (reordenar, planear, revisar) ────

INSTRUCCIONES_REORDENAR = """\
Eres el ayudante de un buscador. Recibirás una pregunta y unas fichas \
numeradas. Elige las {k} fichas MÁS útiles para responderla. Si la pregunta \
necesita varios datos, elige fichas que los cubran todos. Responde solo con los \
números de las fichas elegidas, en el campo "elegidas"."""

ESQUEMA_REORDENAR = {
    "type": "object",
    "properties": {"elegidas": {"type": "array", "items": {"type": "integer"}}},
    "required": ["elegidas"],
}

INSTRUCCIONES_PLANEAR = """\
Eres un agente que busca información en unos documentos para responder una \
pregunta. Divide la pregunta en 1 a {maximo} búsquedas cortas e independientes, \
una por cada dato que haga falta encontrar. Usa palabras que probablemente \
aparezcan en los documentos y escribe las búsquedas en el mismo idioma de la \
pregunta. Si la pregunta pide una lista completa de cosas dispersas, haz una \
búsqueda por cada tipo de cosa."""

ESQUEMA_PLAN = {
    "type": "object",
    "properties": {"busquedas": {"type": "array", "items": {"type": "string"}}},
    "required": ["busquedas"],
}

INSTRUCCIONES_REVISAR = """\
Eres un agente que revisa si ya tiene la información necesaria. Recibirás una \
pregunta y las fichas encontradas hasta ahora. Decide si con ellas se puede \
responder TODA la pregunta. Si falta algo, di qué falta y propone UNA búsqueda \
nueva, con palabras distintas a las ya usadas y en el idioma de la pregunta. Si \
la pregunta pide una lista completa, piensa si podría haber más elementos en \
otras partes de los documentos."""

ESQUEMA_REVISION = {
    "type": "object",
    "properties": {
        "falta": {"type": "string"},
        "suficiente": {"type": "boolean"},
        "nueva_busqueda": {"type": "string"},
    },
    "required": ["falta", "suficiente", "nueva_busqueda"],
}


def prefijos_para(modelo_de_embeddings: str) -> tuple[str, str]:
    """
    nomic-embed-text fue entrenado con prefijos que le dicen el "papel" de cada
    texto: "search_document: " para las fichas y "search_query: " para las
    preguntas. Con ellos busca mejor. Otros modelos no los necesitan.
    """
    if modelo_de_embeddings.startswith("nomic-embed-text"):
        return "search_document: ", "search_query: "
    return "", ""


# ════════════════════════════════════════════════════════════════════════════
# 📦 ESTRUCTURAS DE DATOS
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Fragmento:
    """Una "ficha": un pedazo de un documento, con su origen."""

    documento: str  # de qué archivo salió
    seccion: str  # bajo qué título estaba (ayuda a buscar y a citar)
    texto: str
    numero: int  # su posición en el fichero (-1 si es un "padre" armado al vuelo)

    @property
    def texto_con_contexto(self) -> str:
        """
        El texto que se convierte en embedding. Le anteponemos el documento y la
        sección: así una ficha que solo dice "14 días hábiles" también "sabe"
        que habla de vacaciones.
        """
        return f"[{self.documento} › {self.seccion}]\n{self.texto}"


@dataclass
class FragmentoRecuperado:
    """Una ficha que trajo el buscador, con qué tan parecida era a la pregunta."""

    fragmento: Fragmento
    similitud: float  # de -1 a 1: cuanto más alta, más parecida (por significado)


@dataclass
class ResultadoDeBusqueda:
    """Lo que entrega el bibliotecario: las fichas, el tiempo y lo que hizo."""

    recuperados: list[FragmentoRecuperado]
    segundos: float
    pasos: list[str] = field(default_factory=list)  # bitácora legible de la búsqueda


@dataclass
class InformeDeIndexado:
    """Lo que pasó al preparar el fichero."""

    documentos: list[str]
    numero_de_fragmentos: int
    caracteres_promedio: int
    dimensiones: int  # cuántos números tiene cada embedding
    segundos_indexando: float
    modelo_de_embeddings: str


# ════════════════════════════════════════════════════════════════════════════
# ✂️ TROCEAR LOS DOCUMENTOS
# ════════════════════════════════════════════════════════════════════════════


def partir_texto_largo(texto: str, tamano: int, solapamiento: int) -> list[str]:
    """Parte un texto largo en trozos de `tamano` caracteres que se enciman un poco."""
    trozos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            # Mejor cortar en un espacio que a mitad de una palabra.
            espacio = texto.rfind(" ", inicio + tamano // 2, fin)
            if espacio != -1:
                fin = espacio
        trozos.append(texto[inicio:fin].strip())
        if fin == len(texto):
            break
        inicio = max(fin - solapamiento, inicio + 1)
    return trozos


def trocear_documento(
    documento: Documento,
    tamano: int = CARACTERES_POR_FRAGMENTO,
    solapamiento: int = SOLAPAMIENTO_EN_CARACTERES,
) -> list[Fragmento]:
    """
    Corta un documento en fichas, respetando su estructura:
      • nunca mezcla dos secciones (un título "# ..." empieza una ficha nueva);
      • junta párrafos seguidos de la misma sección hasta llenar la ficha;
      • parte los párrafos gigantes en trozos que se enciman.
    """
    fragmentos: list[Fragmento] = []
    seccion = "Inicio"
    parrafos_de_la_ficha: list[str] = []

    def cerrar_ficha():
        # Una ficha que solo tiene un título (p. ej. "## 3. Acuerdos" seguido de
        # "### Acuerdo 1") no aporta nada al buscador: la descartamos.
        solo_un_titulo = len(parrafos_de_la_ficha) == 1 and parrafos_de_la_ficha[0].startswith("#")
        if parrafos_de_la_ficha and not solo_un_titulo:
            fragmentos.append(
                Fragmento(documento.nombre, seccion, "\n\n".join(parrafos_de_la_ficha), len(fragmentos))
            )
        parrafos_de_la_ficha.clear()

    for parrafo in re.split(r"\n\s*\n", documento.texto):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        if parrafo.startswith("#"):  # un título: empieza una sección nueva
            cerrar_ficha()
            seccion = parrafo.splitlines()[0].lstrip("#").strip()
        for trozo in partir_texto_largo(parrafo, tamano, solapamiento):
            largo_actual = sum(len(p) for p in parrafos_de_la_ficha)
            if parrafos_de_la_ficha and largo_actual + len(trozo) > tamano:
                cerrar_ficha()
            parrafos_de_la_ficha.append(trozo)
    cerrar_ficha()
    return fragmentos


# ════════════════════════════════════════════════════════════════════════════
# 🔤 BÚSQUEDA POR PALABRAS EXACTAS (BM25) Y FUSIÓN DE LISTAS (RRF)
# ════════════════════════════════════════════════════════════════════════════

# Palabras tan comunes que no ayudan a encontrar nada.
PALABRAS_VACIAS = set(
    """
    de la el en y a los las del que se por un una con para su sus al lo es como mas o
    que cual cuales cuanto cuanta cuantos cuantas cuando quien quienes donde ha hay le
    me mi si no ya este esta estos estas ese esa eso son fue ser era
    the a an of to in on for and or is are was were be by with what which who how
    """.split()
)


def palabras_para_buscar(texto: str) -> list[str]:
    """Minúsculas, sin acentos ni palabras vacías: "¿Cuántas BECAS?" → ["becas"]."""
    sin_acentos = "".join(
        letra for letra in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(letra) != "Mn"
    )
    return [palabra for palabra in re.findall(r"[^\W_]+", sin_acentos) if palabra not in PALABRAS_VACIAS]


class IndiceBM25:
    """
    El buscador "por palabras" de toda la vida (lo usan los motores de búsqueda
    desde los años 90). Puntúa más alto las fichas que contienen las palabras
    de la pregunta, sobre todo las palabras RARAS: "glifosato" pesa mucho más
    que "café", porque aparece en muy pocas fichas.

    A diferencia de los embeddings, no entiende sinónimos ("me lastimo" no
    encuentra "accidente"), pero nunca falla con nombres propios, cifras y
    términos exactos. Por eso se combinan.
    """

    def __init__(self, textos: list[str], k1: float = 1.5, b: float = 0.75):
        self.conteos = [Counter(palabras_para_buscar(texto)) for texto in textos]
        self.largos = [sum(conteo.values()) for conteo in self.conteos]
        self.largo_promedio = sum(self.largos) / max(len(self.largos), 1)
        fichas_con_la_palabra = Counter(palabra for conteo in self.conteos for palabra in conteo)
        total = len(self.conteos)
        # IDF: cuanto menos fichas tienen una palabra, más "informativa" es.
        self.idf = {
            palabra: math.log(1 + (total - n + 0.5) / (n + 0.5)) for palabra, n in fichas_con_la_palabra.items()
        }
        self.k1, self.b = k1, b

    def puntajes(self, consulta: str) -> np.ndarray:
        """Un puntaje por ficha: cuanto más alto, más palabras (raras) de la consulta contiene."""
        palabras = palabras_para_buscar(consulta)
        resultado = np.zeros(len(self.conteos), dtype=np.float32)
        for indice, (conteo, largo) in enumerate(zip(self.conteos, self.largos)):
            for palabra in palabras:
                frecuencia = conteo.get(palabra, 0)
                if frecuencia:
                    # La fórmula BM25: premia la frecuencia, pero con rendimientos
                    # decrecientes (k1), y castiga un poco las fichas muy largas (b).
                    resultado[indice] += self.idf[palabra] * frecuencia * (self.k1 + 1) / (
                        frecuencia + self.k1 * (1 - self.b + self.b * largo / self.largo_promedio)
                    )
        return resultado


def fusion_rrf(listas: list[list[int]], k: int = 60) -> list[int]:
    """
    Reciprocal Rank Fusion: junta varias listas ordenadas sin comparar sus
    puntajes (que están en escalas distintas). Cada ficha suma 1 / (k + puesto)
    por cada lista en la que aparece: estar arriba en las dos listas es lo mejor.
    """
    puntos: Counter = Counter()
    for lista in listas:
        for puesto, indice in enumerate(lista, start=1):
            puntos[indice] += 1 / (k + puesto)
    return [indice for indice, _ in puntos.most_common()]


# ════════════════════════════════════════════════════════════════════════════
# 📚 EL MOTOR RAG
# ════════════════════════════════════════════════════════════════════════════


class MotorRAG:
    """
    Trocea, indexa y busca; después responde con el mismo modelo que CAG.

    Uso mínimo:
        motor = MotorRAG("gpt-oss:20b", estrategia="hibrido")
        motor.indexar([Documento("manual.md", "...")])
        respuesta = motor.preguntar("¿Cuántos días de vacaciones tengo?")
        print(respuesta.texto, respuesta.pasos_de_busqueda)
    """

    def __init__(
        self,
        modelo: str = MODELO_RECOMENDADO,
        razonar: bool = True,
        modelo_de_embeddings: str = MODELO_DE_EMBEDDINGS,
        fragmentos_a_recuperar: int = FRAGMENTOS_A_RECUPERAR,
        estrategia: str = "clasico",
        cliente: ollama.Client | None = None,
    ):
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"Estrategia desconocida: {estrategia!r}. Opciones: {', '.join(ESTRATEGIAS)}")
        self.cliente = cliente or ollama.Client()
        self.modelo = describir_modelo(self.cliente, modelo)
        self.razonar = razonar
        self.modelo_de_embeddings = modelo_de_embeddings
        self.fragmentos_a_recuperar = fragmentos_a_recuperar
        self.estrategia = estrategia  # se puede cambiar en cualquier momento

        self.fragmentos: list[Fragmento] = []  # el fichero
        self.vectores: np.ndarray | None = None  # una fila de números por ficha
        self.indice_bm25: IndiceBM25 | None = None
        self.textos_de_documentos: dict[str, str] = {}  # para la estrategia padre-hijo
        self.num_ctx: int = 8192
        self.duracion_cache: str = "10m"
        self.informe_de_indexado: InformeDeIndexado | None = None

    @property
    def tiene_indice(self) -> bool:
        return self.vectores is not None

    def parametros_base(self) -> dict:
        """Los mismos ajustes de chat que usa MotorCAG (ver cag_core.py)."""
        return {
            "model": self.modelo.nombre,
            "think": nivel_de_razonamiento(self.modelo, self.razonar),
            "keep_alive": self.duracion_cache,
            "options": {"num_ctx": self.num_ctx, "temperature": TEMPERATURA},
        }

    def _a_embeddings(self, textos: list[str], prefijo: str) -> np.ndarray:
        """Convierte textos en vectores de largo 1 (así la similitud coseno es un simple producto)."""
        vectores = []
        for inicio in range(0, len(textos), 32):  # de 32 en 32 para no saturar a Ollama
            lote = [prefijo + texto for texto in textos[inicio : inicio + 32]]
            respuesta = self.cliente.embed(
                model=self.modelo_de_embeddings, input=lote, keep_alive=self.duracion_cache
            )
            vectores.extend(respuesta.embeddings)
        matriz = np.array(vectores, dtype=np.float32)
        return matriz / np.linalg.norm(matriz, axis=1, keepdims=True)

    # ── Paso 1: preparar el fichero ─────────────────────────────────────────

    def indexar(
        self, documentos: list[Documento], num_ctx: int | None = None, duracion_cache: str = "10m"
    ) -> InformeDeIndexado:
        """
        Trocea los documentos, calcula el embedding de cada ficha y prepara el
        índice de palabras (BM25). Sirve para las cuatro estrategias.

        `num_ctx`: en la comparación con CAG pasamos la MISMA mesa que usa CAG.
        No es solo por justicia: si cambia num_ctx, Ollama recarga el modelo de
        chat y CAG perdería su caché.
        """
        if not documentos:
            raise ValueError("No hay documentos que indexar.")
        if duracion_cache not in DURACIONES_DE_CACHE:
            raise ValueError(f"Duración de caché no válida: {duracion_cache!r}")
        self.duracion_cache = duracion_cache

        inicio = time.perf_counter()
        self.fragmentos = [ficha for documento in documentos for ficha in trocear_documento(documento)]
        prefijo_de_documento, _ = prefijos_para(self.modelo_de_embeddings)
        self.vectores = self._a_embeddings(
            [ficha.texto_con_contexto for ficha in self.fragmentos], prefijo_de_documento
        )
        self.indice_bm25 = IndiceBM25([ficha.texto_con_contexto for ficha in self.fragmentos])
        self.textos_de_documentos = {documento.nombre: documento.texto for documento in documentos}
        segundos = time.perf_counter() - inicio

        # Una sola mesa que sirva para la estrategia que más texto mueve (así
        # cambiar de estrategia nunca obliga a Ollama a recargar el modelo).
        caracteres_maximos = max(
            CANDIDATAS_PARA_REORDENAR * CARACTERES_POR_FRAGMENTO,  # el reranking ve 10 candidatas
            self.fragmentos_a_recuperar * TAMANO_MAXIMO_DEL_PADRE,  # padre-hijo entrega padres
            (BUSQUEDAS_INICIALES_DEL_AGENTE + REVISIONES_DEL_AGENTE)
            * self.fragmentos_a_recuperar
            * CARACTERES_POR_FRAGMENTO,  # el agente revisa todo lo que juntó
        )
        self.num_ctx = num_ctx or calcular_tamano_de_mesa(caracteres_maximos)

        self.informe_de_indexado = InformeDeIndexado(
            documentos=sorted(documento.nombre for documento in documentos),
            numero_de_fragmentos=len(self.fragmentos),
            caracteres_promedio=round(sum(len(f.texto) for f in self.fragmentos) / len(self.fragmentos)),
            dimensiones=int(self.vectores.shape[1]),
            segundos_indexando=segundos,
            modelo_de_embeddings=self.modelo_de_embeddings,
        )
        return self.informe_de_indexado

    # ── Herramientas de búsqueda ────────────────────────────────────────────

    def _similitudes(self, consulta: str) -> np.ndarray:
        """Qué tan parecida (por significado) es cada ficha a la consulta."""
        _, prefijo_de_consulta = prefijos_para(self.modelo_de_embeddings)
        vector = self._a_embeddings([consulta], prefijo_de_consulta)[0]
        return self.vectores @ vector  # similitud coseno con TODAS las fichas a la vez

    def _por_significado(self, consulta: str, cuantas: int) -> list[int]:
        return [int(i) for i in np.argsort(-self._similitudes(consulta))[:cuantas]]

    def _por_palabras(self, consulta: str, cuantas: int) -> list[int]:
        puntajes = self.indice_bm25.puntajes(consulta)
        orden = np.argsort(-puntajes)[:cuantas]
        return [int(i) for i in orden if puntajes[i] > 0]  # solo las que comparten alguna palabra

    def _hibrida_sin_reordenar(self, consulta: str, cuantas: int) -> list[int]:
        """Significado + palabras exactas, fusionadas con RRF."""
        return fusion_rrf([self._por_significado(consulta, 20), self._por_palabras(consulta, 20)])[:cuantas]

    def _pedir_json(self, instrucciones: str, contenido: str, esquema: dict) -> dict | None:
        """
        Le pide al modelo una respuesta en JSON con la forma de `esquema`. Si no
        la cumple (pasa de vez en cuando), lo intenta una vez más; si vuelve a
        fallar, devuelve None y quien llama usa un plan B. Usa la misma mesa que
        el chat (para no recargar el modelo) y el razonamiento mínimo (es rápido).
        """
        for _intento in range(2):
            respuesta = self.cliente.chat(
                model=self.modelo.nombre,
                think=nivel_de_razonamiento(self.modelo, razonar=False),
                keep_alive=self.duracion_cache,
                options={"num_ctx": self.num_ctx, "temperature": TEMPERATURA},
                format=esquema,
                messages=[{"role": "system", "content": instrucciones}, {"role": "user", "content": contenido}],
            )
            try:
                datos = json.loads(respuesta.message.content)
            except json.JSONDecodeError:
                continue
            if isinstance(datos, dict) and all(campo in datos for campo in esquema["required"]):
                return datos
        return None

    def _fichas_numeradas(self, indices: list[int]) -> str:
        return "\n\n".join(f"[{numero}] {self.fragmentos[i].texto_con_contexto}" for numero, i in enumerate(indices))

    def _reordenar(self, pregunta: str, candidatas: list[int], cuantas: int) -> list[int]:
        """
        Reranking: el modelo LEE las candidatas y elige las más útiles. Es más
        lento que comparar números, pero entiende la pregunta de verdad.
        """
        datos = self._pedir_json(
            INSTRUCCIONES_REORDENAR.format(k=cuantas),
            f"<pregunta>\n{pregunta}\n</pregunta>\n\n<fichas>\n{self._fichas_numeradas(candidatas)}\n</fichas>",
            ESQUEMA_REORDENAR,
        )
        elegidas: list[int] = []
        for numero in (datos or {}).get("elegidas", []):
            if isinstance(numero, int) and 0 <= numero < len(candidatas) and candidatas[numero] not in elegidas:
                elegidas.append(candidatas[numero])
        # Plan B: si eligió menos de las necesarias (o falló), completamos con el orden de la fusión.
        for indice in candidatas:
            if len(elegidas) >= cuantas:
                break
            if indice not in elegidas:
                elegidas.append(indice)
        return elegidas[:cuantas]

    @staticmethod
    def _describir(indices_o_fichas) -> str:
        fichas = [f.fragmento if isinstance(f, FragmentoRecuperado) else f for f in indices_o_fichas]
        return "; ".join(f"{ficha.documento} › {ficha.seccion}" for ficha in fichas)

    # ── Las cuatro estrategias ──────────────────────────────────────────────

    def _buscar_clasico(self, pregunta: str, pasos: list[str]) -> list[int]:
        elegidas = self._por_significado(pregunta, self.fragmentos_a_recuperar)
        pasos.append(f"🔎 Por significado, las {len(elegidas)} fichas más parecidas.")
        return elegidas

    def _buscar_hibrido(self, pregunta: str, pasos: list[str]) -> list[int]:
        por_significado = self._por_significado(pregunta, 20)
        por_palabras = self._por_palabras(pregunta, 20)
        candidatas = fusion_rrf([por_significado, por_palabras])[:CANDIDATAS_PARA_REORDENAR]
        pasos.append(
            f"🔀 Junté la búsqueda por significado y la de palabras exactas (BM25 encontró "
            f"{len(por_palabras)} fichas con palabras de la pregunta): {len(candidatas)} candidatas."
        )
        elegidas = self._reordenar(pregunta, candidatas, self.fragmentos_a_recuperar)
        pasos.append(f"🏅 El modelo leyó las candidatas y eligió las {len(elegidas)} más útiles.")
        return elegidas

    def _buscar_padre_hijo(self, pregunta: str, pasos: list[str]) -> list[FragmentoRecuperado]:
        similitudes = self._similitudes(pregunta)
        hijos = [int(i) for i in np.argsort(-similitudes)[: self.fragmentos_a_recuperar]]
        pasos.append(f"🔎 Encontré {len(hijos)} fichas hijas: {self._describir([self.fragmentos[i] for i in hijos])}.")
        padres: list[FragmentoRecuperado] = []
        vistos: set[tuple] = set()
        for indice in hijos:
            hija = self.fragmentos[indice]
            texto_del_documento = self.textos_de_documentos[hija.documento]
            if len(texto_del_documento) <= TAMANO_MAXIMO_DEL_PADRE:
                clave, seccion, texto = (hija.documento,), "documento completo", texto_del_documento
            else:  # documento grande: solo la sección de la ficha
                clave, seccion = (hija.documento, hija.seccion), hija.seccion
                texto = "\n\n".join(f.texto for f in self.fragmentos if (f.documento, f.seccion) == clave)
            if clave in vistos:
                continue
            vistos.add(clave)
            padres.append(FragmentoRecuperado(Fragmento(hija.documento, seccion, texto, -1), float(similitudes[indice])))
        pasos.append(
            f"👪 Entrego sus {len(padres)} padres ({sum(len(p.fragmento.texto) for p in padres):,} caracteres): "
            f"{self._describir(padres)}."
        )
        return padres

    def _buscar_agentico(self, pregunta: str, pasos: list[str]) -> list[int]:
        # 1) Planear: ¿qué datos necesito y cómo los busco?
        plan = self._pedir_json(
            INSTRUCCIONES_PLANEAR.format(maximo=BUSQUEDAS_INICIALES_DEL_AGENTE),
            pregunta,
            ESQUEMA_PLAN,
        )
        busquedas = [b for b in (plan or {}).get("busquedas", []) if isinstance(b, str) and b.strip()]
        busquedas = busquedas[:BUSQUEDAS_INICIALES_DEL_AGENTE] or [pregunta]
        pasos.append(f"🧭 Planeé {len(busquedas)} búsqueda(s): " + " · ".join(f"«{b}»" for b in busquedas))

        # 2) Buscar cada cosa (híbrido, sin reordenar para no gastar tiempo).
        juntadas: list[int] = []
        for busqueda in busquedas:
            nuevas = [i for i in self._hibrida_sin_reordenar(busqueda, self.fragmentos_a_recuperar) if i not in juntadas]
            juntadas.extend(nuevas)
            pasos.append(f"🔀 «{busqueda}» → {len(nuevas)} ficha(s) nueva(s).")

        # 3) Revisar: ¿me falta algo? Si sí, buscar otra vez.
        for _revision in range(REVISIONES_DEL_AGENTE):
            revision = self._pedir_json(
                INSTRUCCIONES_REVISAR,
                f"<pregunta>\n{pregunta}\n</pregunta>\n\n<fichas>\n{self._fichas_numeradas(juntadas)}\n</fichas>",
                ESQUEMA_REVISION,
            )
            if revision is None or revision["suficiente"] or not revision["nueva_busqueda"].strip():
                pasos.append("🤔 ¿Me falta algo? No: con esto se puede responder.")
                break
            nueva = revision["nueva_busqueda"].strip()
            nuevas = [i for i in self._hibrida_sin_reordenar(nueva, self.fragmentos_a_recuperar) if i not in juntadas]
            juntadas.extend(nuevas)
            pasos.append(f"🤔 ¿Me falta algo? Sí: {revision['falta']} → busco «{nueva}»: {len(nuevas)} ficha(s) nueva(s).")

        # 4) Si juntó demasiadas, el modelo elige las más útiles.
        if len(juntadas) > FICHAS_MAXIMAS_DEL_AGENTE:
            juntadas = self._reordenar(pregunta, juntadas, FICHAS_MAXIMAS_DEL_AGENTE)
            pasos.append(f"🏅 Juntó demasiadas: el modelo eligió las {FICHAS_MAXIMAS_DEL_AGENTE} más útiles.")
        return juntadas

    # ── Paso 2: buscar ──────────────────────────────────────────────────────

    def buscar(self, pregunta: str) -> ResultadoDeBusqueda:
        """El bibliotecario: aplica la estrategia elegida y devuelve fichas, tiempo y pasos."""
        if not self.tiene_indice:
            raise RuntimeError("Primero indexa documentos con indexar().")
        inicio = time.perf_counter()
        pasos: list[str] = []
        if self.estrategia == "padre_hijo":
            recuperados = self._buscar_padre_hijo(pregunta, pasos)
        else:
            buscar_con = {
                "clasico": self._buscar_clasico,
                "hibrido": self._buscar_hibrido,
                "agentico": self._buscar_agentico,
            }[self.estrategia]
            indices = buscar_con(pregunta, pasos)
            # La similitud con la pregunta original, para mostrarla en la app.
            similitudes = self._similitudes(pregunta)
            recuperados = [FragmentoRecuperado(self.fragmentos[i], float(similitudes[i])) for i in indices]
        return ResultadoDeBusqueda(recuperados, time.perf_counter() - inicio, pasos)

    # ── Paso 3: responder ───────────────────────────────────────────────────

    def preguntar_en_vivo(self, pregunta: str, historial: list[dict] | None = None) -> "RespuestaEnVivoRAG":
        """Busca las fichas y prepara la respuesta en streaming (como un grifo)."""
        busqueda = self.buscar(pregunta)
        fichas = "\n\n".join(
            f'<fragmento documento="{r.fragmento.documento}" seccion="{r.fragmento.seccion}">\n'
            f"{r.fragmento.texto}\n</fragmento>"
            for r in busqueda.recuperados
        )
        parametros = self.parametros_base()
        parametros_de_la_peticion = {
            **parametros,
            "options": {**parametros["options"], "num_predict": MAXIMO_DE_TOKENS_POR_RESPUESTA},
            "messages": [
                {"role": "system", "content": INSTRUCCIONES_RAG},
                *(historial or []),
                {
                    "role": "user",
                    "content": (
                        f"<fragmentos_recuperados>\n{fichas}\n</fragmentos_recuperados>\n\n"
                        f"<pregunta>\n{pregunta}\n</pregunta>"
                    ),
                },
            ],
        }
        return RespuestaEnVivoRAG(self.cliente, parametros_de_la_peticion, pregunta, busqueda)

    def preguntar(self, pregunta: str, historial: list[dict] | None = None) -> RespuestaDelModelo:
        """Versión sin streaming: espera a que termine y devuelve la respuesta completa."""
        respuesta_en_vivo = self.preguntar_en_vivo(pregunta, historial)
        for _fragmento in respuesta_en_vivo:
            pass
        return respuesta_en_vivo.resultado


class RespuestaEnVivoRAG(RespuestaEnVivo):
    """El mismo "grifo" que usa CAG, pero que además recuerda qué fichas se usaron y cómo."""

    def __init__(
        self,
        cliente: ollama.Client,
        parametros_de_la_peticion: dict,
        pregunta: str,
        busqueda: ResultadoDeBusqueda,
    ):
        super().__init__(cliente, parametros_de_la_peticion, pregunta)
        self._busqueda = busqueda

    def __iter__(self) -> Iterator[str]:
        yield from super().__iter__()
        self.resultado.fragmentos_recuperados = self._busqueda.recuperados
        self.resultado.segundos_buscando = self._busqueda.segundos
        self.resultado.pasos_de_busqueda = self._busqueda.pasos
        # Para comparar con CAG, el tiempo de RAG incluye el tiempo de búsqueda
        # (que en las estrategias híbrida y agéntica incluye llamadas al modelo).
        self.resultado.segundos_totales += self._busqueda.segundos
        if self.resultado.segundos_hasta_primer_token is not None:
            self.resultado.segundos_hasta_primer_token += self._busqueda.segundos


# ════════════════════════════════════════════════════════════════════════════
# 🧪 DEMOSTRACIÓN EN TERMINAL:  python rag_core.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    motor = MotorRAG()
    informe = motor.indexar(cargar_documentos_de_carpeta(CARPETA_CORPUS_DE_EJEMPLO))
    print(
        f"🗂️ Fichero listo: {informe.numero_de_fragmentos} fichas de ~{informe.caracteres_promedio} "
        f"caracteres, embeddings de {informe.dimensiones} números, en {informe.segundos_indexando:.2f} s"
    )

    pregunta = "¿Cuántas toneladas de Nube Morada se produjeron en 2025 y en qué año se registró esa variedad?"
    print(f"\n❓ {pregunta}")
    for clave, estrategia in ESTRATEGIAS.items():
        motor.estrategia = clave
        respuesta = motor.preguntar(pregunta)
        print(f"\n{estrategia.nombre}  (búsqueda: {respuesta.segundos_buscando:.1f} s)")
        for paso in respuesta.pasos_de_busqueda:
            print(f"   {paso}")
        print(f"   💬 {respuesta.texto.strip()[:200]}")
