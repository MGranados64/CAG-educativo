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

🧩 CÓMO FLUYE LA INFORMACIÓN EN ESTE ARCHIVO
--------------------------------------------
  1. MotorRAG.indexar()   → trocea los documentos en fragmentos y convierte cada
     uno en un "embedding": una lista de números que representa su SIGNIFICADO,
     como unas coordenadas en un mapa donde los textos parecidos quedan cerca.
  2. MotorRAG.buscar()    → convierte la pregunta en embedding y mide qué
     fragmentos quedan más cerca en ese mapa (similitud coseno). Se queda con
     los k más cercanos.
  3. MotorRAG.preguntar_en_vivo() → manda al MISMO modelo de chat que usa CAG
     solo esos fragmentos y la pregunta.

⚖️ UNA PELEA JUSTA CONTRA CAG
-----------------------------
Para comparar con CAG (comparacion.py) usamos el mismo modelo de chat, la misma
mesa (num_ctx), el mismo razonamiento, temperatura 0 y reglas equivalentes. La
ÚNICA diferencia: CAG ve todos los documentos; RAG, solo lo que trae el buscador.

Pruébalo desde la terminal (con Ollama encendido):

    python rag_core.py
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import ollama

from cag_core import (
    CARPETA_CORPUS_DE_EJEMPLO,
    DURACIONES_DE_CACHE,
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
    numero: int  # su posición en el fichero

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
    similitud: float  # de -1 a 1: cuanto más alta, más parecida


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
# 📚 EL MOTOR RAG
# ════════════════════════════════════════════════════════════════════════════


class MotorRAG:
    """
    Trocea, indexa y busca; después responde con el mismo modelo que CAG.

    Uso mínimo:
        motor = MotorRAG("gpt-oss:20b")
        motor.indexar([Documento("manual.md", "...")])
        respuesta = motor.preguntar("¿Cuántos días de vacaciones tengo?")
        print(respuesta.texto, respuesta.fragmentos_recuperados)
    """

    def __init__(
        self,
        modelo: str = MODELO_RECOMENDADO,
        razonar: bool = True,
        modelo_de_embeddings: str = MODELO_DE_EMBEDDINGS,
        fragmentos_a_recuperar: int = FRAGMENTOS_A_RECUPERAR,
        cliente: ollama.Client | None = None,
    ):
        self.cliente = cliente or ollama.Client()
        self.modelo = describir_modelo(self.cliente, modelo)
        self.razonar = razonar
        self.modelo_de_embeddings = modelo_de_embeddings
        self.fragmentos_a_recuperar = fragmentos_a_recuperar

        self.fragmentos: list[Fragmento] = []  # el fichero
        self.vectores: np.ndarray | None = None  # una fila de números por ficha
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
        Trocea los documentos y calcula el embedding de cada ficha.

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
        segundos = time.perf_counter() - inicio

        # La mesa solo necesita espacio para k fichas + la conversación.
        self.num_ctx = num_ctx or calcular_tamano_de_mesa(self.fragmentos_a_recuperar * CARACTERES_POR_FRAGMENTO)

        self.informe_de_indexado = InformeDeIndexado(
            documentos=sorted(documento.nombre for documento in documentos),
            numero_de_fragmentos=len(self.fragmentos),
            caracteres_promedio=round(sum(len(f.texto) for f in self.fragmentos) / len(self.fragmentos)),
            dimensiones=int(self.vectores.shape[1]),
            segundos_indexando=segundos,
            modelo_de_embeddings=self.modelo_de_embeddings,
        )
        return self.informe_de_indexado

    # ── Paso 2: buscar ──────────────────────────────────────────────────────

    def buscar(self, pregunta: str) -> tuple[list[FragmentoRecuperado], float]:
        """El bibliotecario: devuelve las k fichas más parecidas a la pregunta y cuánto tardó."""
        if not self.tiene_indice:
            raise RuntimeError("Primero indexa documentos con indexar().")
        inicio = time.perf_counter()
        _, prefijo_de_consulta = prefijos_para(self.modelo_de_embeddings)
        vector_de_la_pregunta = self._a_embeddings([pregunta], prefijo_de_consulta)[0]
        # Similitud coseno con TODAS las fichas de una vez (una multiplicación de matrices).
        similitudes = self.vectores @ vector_de_la_pregunta
        mejores = np.argsort(-similitudes)[: self.fragmentos_a_recuperar]
        recuperados = [
            FragmentoRecuperado(self.fragmentos[indice], float(similitudes[indice])) for indice in mejores
        ]
        return recuperados, time.perf_counter() - inicio

    # ── Paso 3: responder ───────────────────────────────────────────────────

    def preguntar_en_vivo(self, pregunta: str, historial: list[dict] | None = None) -> "RespuestaEnVivoRAG":
        """Busca las fichas y prepara la respuesta en streaming (como un grifo)."""
        recuperados, segundos_buscando = self.buscar(pregunta)
        fichas = "\n\n".join(
            f'<fragmento documento="{r.fragmento.documento}" seccion="{r.fragmento.seccion}">\n'
            f"{r.fragmento.texto}\n</fragmento>"
            for r in recuperados
        )
        parametros_de_la_peticion = {
            **self.parametros_base(),
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
        return RespuestaEnVivoRAG(self.cliente, parametros_de_la_peticion, pregunta, recuperados, segundos_buscando)

    def preguntar(self, pregunta: str, historial: list[dict] | None = None) -> RespuestaDelModelo:
        """Versión sin streaming: espera a que termine y devuelve la respuesta completa."""
        respuesta_en_vivo = self.preguntar_en_vivo(pregunta, historial)
        for _fragmento in respuesta_en_vivo:
            pass
        return respuesta_en_vivo.resultado


class RespuestaEnVivoRAG(RespuestaEnVivo):
    """El mismo "grifo" que usa CAG, pero que además recuerda qué fichas se usaron."""

    def __init__(
        self,
        cliente: ollama.Client,
        parametros_de_la_peticion: dict,
        pregunta: str,
        recuperados: list[FragmentoRecuperado],
        segundos_buscando: float,
    ):
        super().__init__(cliente, parametros_de_la_peticion, pregunta)
        self._recuperados = recuperados
        self._segundos_buscando = segundos_buscando

    def __iter__(self) -> Iterator[str]:
        yield from super().__iter__()
        self.resultado.fragmentos_recuperados = self._recuperados
        self.resultado.segundos_buscando = self._segundos_buscando
        # Para comparar con CAG, el tiempo de RAG incluye el tiempo de búsqueda.
        self.resultado.segundos_totales += self._segundos_buscando
        if self.resultado.segundos_hasta_primer_token is not None:
            self.resultado.segundos_hasta_primer_token += self._segundos_buscando


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

    pregunta = "Si me lastimo trabajando, ¿a quién tengo que avisar?"
    respuesta = motor.preguntar(pregunta)
    print(f"\n❓ {pregunta}\n🔎 El buscador trajo (en {respuesta.segundos_buscando * 1000:.0f} ms):")
    for recuperado in respuesta.fragmentos_recuperados:
        ficha = recuperado.fragmento
        print(f"   {recuperado.similitud:.3f}  {ficha.documento} › {ficha.seccion}")
    print(f"💬 {respuesta.texto}")
