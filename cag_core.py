"""
cag_core.py — El corazón del sistema CAG (Cache-Augmented Generation)
======================================================================

Todo corre en TU computadora con Ollama: gratis, privado y sin internet.

🎓 LA ANALOGÍA CENTRAL
----------------------
Imagina un examen:

  • RAG (Retrieval-Augmented Generation) es obligar al estudiante a ir a la
    biblioteca CADA VEZ que le hacen una pregunta: busca en el catálogo, saca
    unos cuantos libros, lee unas páginas y responde. Si el bibliotecario le
    trae la página equivocada, la respuesta sale mal.

  • CAG (Cache-Augmented Generation) es darle al estudiante el libro completo
    ANTES del examen para que lo memorice. Cuando llega la pregunta, la
    respuesta ya está en su cabeza: no hay búsqueda y no hay página equivocada.

¿Y la "caché"? Para "leer" un texto, el modelo calcula para cada token unas
notas internas llamadas claves y valores (Keys y Values). Guardarlas es la
"caché KV": son los apuntes del estudiante. Ollama las conserva en la memoria
de tu tarjeta gráfica, así que si la siguiente pregunta empieza con el mismo
libro, el modelo NO lo vuelve a leer: consulta sus apuntes.

🧩 CÓMO FLUYE LA INFORMACIÓN EN ESTE ARCHIVO
--------------------------------------------
  1. MotorCAG.cargar_conocimiento() → junta tus documentos en un solo bloque
     (`conocimiento_en_cache`), calcula qué tan grande debe ser la "mesa de
     estudio" (num_ctx) y precalienta la caché leyendo el libro una vez.
  2. MotorCAG.preguntar_en_vivo()  → envía [conocimiento + instrucciones +
     pregunta] y devuelve la respuesta palabra por palabra (streaming).
  3. RespuestaDelModelo             → la respuesta final más sus "tiempos":
     cuánto tardó en leer el prompt y cuánto habría tardado sin caché.

📏 LA REGLA DE ORO DE LA CACHÉ: EL PREFIJO
-----------------------------------------
Las notas de cada token dependen de TODOS los tokens anteriores. Por eso solo
se pueden reutilizar si el INICIO del prompt es idéntico al de antes. Así que
el conocimiento va SIEMPRE primero y nunca cambia, y lo que sí cambia
(instrucciones, preguntas) va DESPUÉS.

    ┌───────────────────────────────────┬───────────────┬──────────────┐
    │ 📚 Conocimiento (siempre idéntico) │ Instrucciones │   Pregunta   │
    └───────────────────────────────────┴───────────────┴──────────────┘
      ▲ esto se reutiliza de la caché     ▲ se lee cada vez (es corto)

Pruébalo desde la terminal (con Ollama encendido):

    python cag_core.py
"""

from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import ollama

# ════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

# El modelo local que "memoriza" el conocimiento y responde. gpt-oss:20b fue el
# que mejor equilibró precisión, velocidad y calidad como juez en nuestras
# pruebas con una GPU de 16 GB. Puedes elegir otro en la app o con CAG_MODELO.
MODELO_RECOMENDADO = os.environ.get("CAG_MODELO", "gpt-oss:20b")

# La "mesa de estudio" (ventana de contexto, `num_ctx` en Ollama) es cuánto
# texto cabe a la vez en la memoria de trabajo del modelo. Una mesa más grande
# ocupa más memoria de la tarjeta gráfica (VRAM): con 16 GB, gpt-oss:20b cabe
# completo en la GPU hasta 65,536 tokens; con más, empieza a usar la CPU y se
# vuelve lento. Ajústalo a tu equipo con la variable CAG_NUM_CTX_MAXIMO.
NUM_CTX_MAXIMO = int(os.environ.get("CAG_NUM_CTX_MAXIMO", "65536"))

# Espacio que dejamos libre en la mesa para la conversación, el razonamiento
# interno del modelo y su respuesta.
MARGEN_PARA_CONVERSACION = 8_192

# Para elegir el tamaño de la mesa ANTES de leer el libro, estimamos sus tokens.
# En español un token suele ser de 3 a 4 caracteres; suponemos 2.5 para
# quedarnos del lado seguro (es mejor que sobre mesa a que falte).
CARACTERES_POR_TOKEN_PESIMISTA = 2.5

# 0 = respuestas reproducibles: la misma pregunta da la misma respuesta.
TEMPERATURA = 0

# Cuánto tiempo mantiene Ollama el modelo (y su caché) en memoria desde la
# última pregunta. Al pasar ese tiempo lo descarga y la caché se pierde.
DURACIONES_DE_CACHE = {"10m": "10 minutos", "1h": "1 hora"}

# Margen al comparar tamaños de prompt para detectar recortes (ver RespuestaEnVivo).
TOLERANCIA_DE_TOKENS = 256

# Los documentos de ejemplo: cuatro textos ficticios de una cooperativa cafetalera.
CARPETA_CORPUS_DE_EJEMPLO = Path(__file__).parent / "datos" / "cooperativa"

# Para la proyección educativa: "¿cuánto tardaría en releer un libro grande?"
TOKENS_DE_UN_LIBRO_GRANDE = 100_000

# ════════════════════════════════════════════════════════════════════════════
# 🗣️ INSTRUCCIONES PARA EL MODELO
# Van DESPUÉS del conocimiento, así que podemos cambiarlas sin perder la caché
# del conocimiento (recuerda la regla del prefijo).
# ════════════════════════════════════════════════════════════════════════════

INSTRUCCIONES_MODO_ESTRICTO = """\
Eres un asistente que responde preguntas usando EXCLUSIVAMENTE la biblioteca de \
conocimiento que aparece arriba, dentro de <biblioteca_de_conocimiento>.

Reglas:
- Responde solo con información que esté en esos documentos. No uses tu \
conocimiento general ni supongas datos que no aparecen.
- Si la respuesta no está en los documentos, dilo con claridad (por ejemplo: \
"Esa información no aparece en los documentos cargados").
- Cuando sea útil, indica de qué sección sale el dato.
- Responde en el mismo idioma de la pregunta, de forma clara y breve."""

# Este modo existe para EXPERIMENTAR: le damos permiso al modelo de usar lo que
# sabe "de memoria", y así puedes ver cómo el juez de veracidad (evals.py)
# detecta las afirmaciones que no salen del documento.
INSTRUCCIONES_MODO_LIBRE = """\
Eres un asistente útil. Arriba tienes una biblioteca de conocimiento dentro de \
<biblioteca_de_conocimiento>: úsala como fuente principal, pero también puedes \
completar tus respuestas con tu conocimiento general cuando los documentos no \
alcancen.
Responde en el mismo idioma de la pregunta, de forma clara y breve."""


# ════════════════════════════════════════════════════════════════════════════
# 📦 ESTRUCTURAS DE DATOS
# Son como "formularios" con campos fijos: hacen explícito qué información
# viaja de una parte del programa a otra.
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Documento:
    """Un documento que el usuario quiere que el modelo memorice."""

    nombre: str
    texto: str


@dataclass
class ModeloLocal:
    """La "ficha técnica" de un modelo instalado en Ollama."""

    nombre: str  # p. ej. "gpt-oss:20b"
    familia: str  # p. ej. "gptoss", "gemma4"
    tamano: str  # número de parámetros, p. ej. "20.9B"
    cuantizacion: str  # cómo se comprimieron sus números, p. ej. "Q4_K_M"
    contexto_maximo: int  # la mesa más grande que el modelo admite
    soporta_razonamiento: bool  # ¿puede "pensar" antes de responder?


@dataclass
class MedicionDeTiempos:
    """
    El "cronómetro" de una llamada al modelo. Ollama nos dice cuánto tardó en
    cada fase; con eso vemos la caché en acción.

    Analogía: leer el libro entero toma minutos; consultar tus apuntes, segundos.
    Si el tiempo de lectura es diminuto, el modelo usó sus apuntes (la caché).
    """

    tokens_del_prompt: int  # todo lo que se le envió: libro + instrucciones + pregunta
    segundos_leyendo: float  # cuánto tardó en procesar (leer) ese prompt
    tokens_generados: int  # lo que escribió: razonamiento interno + respuesta
    segundos_generando: float
    segundos_cargando_modelo: float  # > 0 si tuvo que cargar el modelo en memoria

    @classmethod
    def desde_respuesta(cls, respuesta) -> "MedicionDeTiempos":
        """Traduce los campos de Ollama (en nanosegundos) a segundos."""
        return cls(
            tokens_del_prompt=respuesta.prompt_eval_count or 0,
            segundos_leyendo=(respuesta.prompt_eval_duration or 0) / 1e9,
            tokens_generados=respuesta.eval_count or 0,
            segundos_generando=(respuesta.eval_duration or 0) / 1e9,
            segundos_cargando_modelo=(respuesta.load_duration or 0) / 1e9,
        )

    @property
    def velocidad_de_lectura(self) -> float:
        """Tokens 'leídos' por segundo. Con caché sale altísima: no leyó, recordó."""
        return self.tokens_del_prompt / self.segundos_leyendo if self.segundos_leyendo else 0.0

    @property
    def velocidad_de_escritura(self) -> float:
        """Tokens generados por segundo."""
        return self.tokens_generados / self.segundos_generando if self.segundos_generando else 0.0

    @property
    def modelo_recien_cargado(self) -> bool:
        """
        Si Ollama tuvo que cargar el modelo, la caché anterior se había perdido
        (pasó el tiempo de `keep_alive` o se cambió algo que obliga a recargar).
        """
        return self.segundos_cargando_modelo > 0.5

    def segundos_sin_cache(self, velocidad_en_frio: float) -> float | None:
        """Estimación: cuánto habría tardado en leer TODO el prompt desde cero."""
        if not velocidad_en_frio:
            return None
        return self.tokens_del_prompt / velocidad_en_frio


@dataclass
class InformeDeCarga:
    """Lo que pasó al inyectar el conocimiento en la caché."""

    documentos: list[str]
    caracteres: int
    tokens_en_memoria: int  # tokens que ocupa el libro (más instrucciones) en la mesa
    num_ctx: int  # tamaño de la mesa elegido
    lectura_en_frio: MedicionDeTiempos  # la primera lectura, sin caché
    aviso: str | None = None

    @property
    def fraccion_de_mesa(self) -> float:
        """Qué parte de la mesa ocupa el libro (0.0 a 1.0)."""
        return self.tokens_en_memoria / self.num_ctx


@dataclass
class RespuestaDelModelo:
    """
    Una respuesta del modelo con toda la información para auditarla.
    La usan los dos sistemas: CAG (cag_core.py) y RAG (rag_core.py).
    """

    pregunta: str
    texto: str
    medicion: MedicionDeTiempos
    modelo: str
    segundos_hasta_primer_token: float | None
    segundos_totales: float
    razonamiento: str  # lo que el modelo "pensó" antes de responder (si razona)
    prompt_recortado: bool  # ⚠️ True si el libro no cupo completo en la mesa
    cache_rota_a_proposito: bool = False
    # Solo en RAG: los fragmentos que trajo el buscador y cuánto tardó en buscarlos.
    fragmentos_recuperados: list | None = None
    segundos_buscando: float = 0.0


class ConocimientoDemasiadoGrande(ValueError):
    """Los documentos no caben en la mesa de estudio del modelo."""


# ════════════════════════════════════════════════════════════════════════════
# 🔎 CONOCER LOS MODELOS INSTALADOS
# ════════════════════════════════════════════════════════════════════════════


def describir_modelo(cliente: ollama.Client, nombre: str) -> ModeloLocal:
    """Pide a Ollama la ficha técnica de un modelo instalado."""
    ficha = cliente.show(nombre)
    arquitectura = (ficha.modelinfo or {}).get("general.architecture", "")
    return ModeloLocal(
        nombre=nombre,
        familia=ficha.details.family or arquitectura,
        tamano=ficha.details.parameter_size or "?",
        cuantizacion=ficha.details.quantization_level or "?",
        contexto_maximo=int((ficha.modelinfo or {}).get(f"{arquitectura}.context_length", 8192)),
        soporta_razonamiento="thinking" in (ficha.capabilities or []),
    )


def listar_modelos_de_chat(cliente: ollama.Client) -> list[ModeloLocal]:
    """
    Todos los modelos instalados que saben conversar. Descartamos los de
    "embeddings" (como nomic-embed-text), que convierten texto en números para
    buscar (los usa RAG), pero no saben escribir respuestas.
    """
    modelos = []
    for instalado in cliente.list().models:
        if "completion" in (cliente.show(instalado.model).capabilities or []):
            modelos.append(describir_modelo(cliente, instalado.model))
    return sorted(modelos, key=lambda modelo: modelo.nombre)


def nivel_de_razonamiento(modelo: ModeloLocal, razonar: bool) -> bool | str | None:
    """
    Traduce "¿quieres que piense antes de responder?" al valor que entiende
    cada modelo. Pensar lo hace más lento pero más preciso (por ejemplo, en
    cálculos como "días de vacaciones en el tercer año").
    """
    if not modelo.soporta_razonamiento:
        return None  # este modelo no tiene modo de razonamiento
    if modelo.familia == "gptoss":
        # gpt-oss no permite apagarlo del todo: solo bajar de "medium" a "low".
        return "medium" if razonar else "low"
    return razonar


# ════════════════════════════════════════════════════════════════════════════
# 📚 PREPARAR EL CONOCIMIENTO
# ════════════════════════════════════════════════════════════════════════════


def extraer_texto(nombre_archivo: str, contenido: bytes) -> str:
    """
    Convierte un archivo subido en texto plano.

    Los PDF se "desarman" página por página; los demás formatos de texto
    (.txt, .md, .csv, .json) solo se decodifican.
    """
    if nombre_archivo.lower().endswith(".pdf"):
        from pypdf import PdfReader  # solo se importa si de verdad hay un PDF

        lector_pdf = PdfReader(io.BytesIO(contenido))
        paginas = [
            f"[Página {numero}]\n{pagina.extract_text() or ''}"
            for numero, pagina in enumerate(lector_pdf.pages, start=1)
        ]
        return "\n\n".join(paginas)
    return contenido.decode("utf-8", errors="replace")


def cargar_documentos_de_carpeta(carpeta: Path) -> list[Documento]:
    """
    Lee todos los documentos (.md, .txt, .pdf) de una carpeta. Los archivos que
    empiezan con "_" (notas, licencias) y los .json (preguntas de prueba) no son
    conocimiento, así que se ignoran.
    """
    documentos = []
    for ruta in sorted(Path(carpeta).iterdir()):
        if ruta.suffix.lower() in {".md", ".txt", ".pdf"} and not ruta.name.startswith("_"):
            documentos.append(Documento(ruta.name, extraer_texto(ruta.name, ruta.read_bytes())))
    return documentos


def construir_conocimiento_en_cache(documentos: list[Documento]) -> str:
    """
    Empaqueta todos los documentos en un único bloque de texto con etiquetas
    XML, para que el modelo sepa dónde empieza y termina cada uno.

    Ordenamos por nombre para que el MISMO conjunto de documentos produzca
    SIEMPRE exactamente el mismo texto. Si el orden cambiara al azar, el prefijo
    cambiaría y la caché nunca se reutilizaría (un "invalidador silencioso").
    """
    documentos_ordenados = sorted(documentos, key=lambda documento: documento.nombre)
    partes = ["<biblioteca_de_conocimiento>"]
    for indice, documento in enumerate(documentos_ordenados, start=1):
        partes.append(
            f'<documento indice="{indice}" nombre="{documento.nombre}">\n'
            f"{documento.texto.strip()}\n"
            f"</documento>"
        )
    partes.append("</biblioteca_de_conocimiento>")
    return "\n".join(partes)


def calcular_tamano_de_mesa(caracteres: int) -> int:
    """
    Elige `num_ctx`: lo que (pesimistamente) ocupará el libro, más el margen
    para conversar, redondeado hacia arriba a múltiplos de 4096 tokens.
    """
    tokens_estimados = caracteres / CARACTERES_POR_TOKEN_PESIMISTA
    return max(8192, math.ceil((tokens_estimados + MARGEN_PARA_CONVERSACION) / 4096) * 4096)


# ════════════════════════════════════════════════════════════════════════════
# 🧠 EL MOTOR CAG
# ════════════════════════════════════════════════════════════════════════════


class MotorCAG:
    """
    Guarda el conocimiento "memorizado" y responde preguntas sobre él.

    Uso mínimo:
        motor = MotorCAG("gpt-oss:20b")
        motor.cargar_conocimiento([Documento("manual.md", "...")])
        respuesta = motor.preguntar("¿Cuántos días de vacaciones tengo?")
        print(respuesta.texto)
    """

    def __init__(
        self,
        modelo: str = MODELO_RECOMENDADO,
        razonar: bool = True,
        cliente: ollama.Client | None = None,
    ):
        # El "cliente" es el teléfono con el que hablamos con Ollama. Busca el
        # servidor en http://localhost:11434 (o en la variable OLLAMA_HOST).
        self.cliente = cliente or ollama.Client()
        self.modelo = describir_modelo(self.cliente, modelo)
        self.razonar = razonar

        self.conocimiento_en_cache: str = ""  # el "libro" que el modelo memoriza
        self.documentos_cargados: list[str] = []
        self.duracion_cache: str = "10m"
        self.num_ctx: int = 8192
        self.informe_de_carga: InformeDeCarga | None = None

    @property
    def tiene_conocimiento(self) -> bool:
        return bool(self.conocimiento_en_cache)

    @property
    def velocidad_de_lectura_en_frio(self) -> float:
        """Tokens por segundo cuando el modelo lee SIN caché (medido al inyectar)."""
        return self.informe_de_carga.lectura_en_frio.velocidad_de_lectura if self.informe_de_carga else 0.0

    # ── Piezas reutilizables de cada petición ───────────────────────────────

    def parametros_base(self) -> dict:
        """
        Parámetros idénticos en TODAS las llamadas (precalentamiento, chat y
        jueces). Mantenerlos iguales es lo que permite reutilizar la caché:

        • cambiar de modelo es cambiar de estudiante (el nuevo no recuerda nada);
        • cambiar `num_ctx` obliga a Ollama a recargar el modelo (caché perdida);
        • en algunos modelos, como gpt-oss, el nivel de razonamiento se escribe
          al INICIO del prompt, así que cambiarlo también rompe el prefijo.
        """
        return {
            "model": self.modelo.nombre,
            "think": nivel_de_razonamiento(self.modelo, self.razonar),
            "keep_alive": self.duracion_cache,
            "options": {"num_ctx": self.num_ctx, "temperature": TEMPERATURA},
        }

    def prompt_de_sistema(self, instrucciones: str, romper_cache: bool = False) -> str:
        """
        Arma el mensaje de sistema: primero el conocimiento (prefijo estable),
        después las instrucciones.

        `romper_cache=True` es un EXPERIMENTO: pone la hora exacta al principio.
        Como el inicio cambia en cada pregunta, el modelo ya no reconoce el
        prefijo y tiene que releer todo el libro. Es el error más común en la
        vida real: un "invalidador silencioso" (todo funciona, pero lento).
        """
        partes = [self.conocimiento_en_cache, instrucciones]
        if romper_cache:
            partes.insert(0, f"[Consulta realizada a las {datetime.now():%H:%M:%S.%f}]")
        return "\n\n".join(partes)

    # ── Paso 1: inyectar el conocimiento ────────────────────────────────────

    def cargar_conocimiento(
        self, documentos: list[Documento], duracion_cache: str = "10m"
    ) -> InformeDeCarga:
        """
        "Entrega el libro al estudiante antes del examen".

        1. Junta los documentos en `conocimiento_en_cache`.
        2. Calcula el tamaño de la mesa (num_ctx) y comprueba que el libro cabe.
        3. Borra la memoria del modelo (lo descarga) para medir una lectura
           100 % en frío, sin apuntes previos.
        4. Precalienta la caché: le pide leer el libro y escribir solo 1 token.
           Así tu primera pregunta real ya encuentra el libro memorizado.
        """
        if not documentos:
            raise ValueError("No hay documentos que cargar.")
        if duracion_cache not in DURACIONES_DE_CACHE:
            raise ValueError(f"Duración de caché no válida: {duracion_cache!r}")

        conocimiento = construir_conocimiento_en_cache(documentos)
        num_ctx = calcular_tamano_de_mesa(len(conocimiento))
        limite = min(self.modelo.contexto_maximo, NUM_CTX_MAXIMO)
        if num_ctx > limite:
            raise ConocimientoDemasiadoGrande(
                f"Tus documentos necesitan una mesa de ~{num_ctx:,} tokens y el límite "
                f"para '{self.modelo.nombre}' en este equipo es {limite:,}. Esta es justo "
                "la limitación de CAG: todo el conocimiento debe caber en la memoria del "
                "modelo. Para bibliotecas más grandes se usa RAG."
            )

        self.conocimiento_en_cache = conocimiento
        self.documentos_cargados = sorted(documento.nombre for documento in documentos)
        self.duracion_cache = duracion_cache
        self.num_ctx = num_ctx

        # 🧽 Mente en blanco: keep_alive=0 descarga el modelo y con él su caché.
        self.cliente.generate(model=self.modelo.nombre, keep_alive=0)

        # 📖 Primera lectura (en frío). Usamos EXACTAMENTE el mismo prompt de
        # sistema que usará el chat, para que la caché quede lista.
        parametros = self.parametros_base()
        respuesta = self.cliente.chat(
            **{**parametros, "options": {**parametros["options"], "num_predict": 1}},
            messages=[
                {"role": "system", "content": self.prompt_de_sistema(INSTRUCCIONES_MODO_ESTRICTO)},
                # Texto de relleno: el modelo lo lee, pero solo escribe 1 token.
                {"role": "user", "content": "."},
            ],
        )
        lectura_en_frio = MedicionDeTiempos.desde_respuesta(respuesta)

        aviso = None
        if lectura_en_frio.tokens_del_prompt > num_ctx - MARGEN_PARA_CONVERSACION:
            aviso = (
                "El libro ocupó más tokens de lo previsto y podría haberse recortado. "
                "Prueba con menos documentos."
            )

        self.informe_de_carga = InformeDeCarga(
            documentos=self.documentos_cargados,
            caracteres=len(conocimiento),
            tokens_en_memoria=lectura_en_frio.tokens_del_prompt,
            num_ctx=num_ctx,
            lectura_en_frio=lectura_en_frio,
            aviso=aviso,
        )
        return self.informe_de_carga

    # ── Paso 2: preguntar ───────────────────────────────────────────────────

    def preguntar_en_vivo(
        self,
        pregunta: str,
        historial: list[dict] | None = None,
        permitir_conocimiento_externo: bool = False,
        romper_cache: bool = False,
    ) -> "RespuestaEnVivo":
        """
        Prepara la petición y devuelve un objeto que entrega la respuesta
        pedacito a pedacito (streaming), como un grifo que abres y del que
        sale el agua poco a poco.

        `historial` son los turnos anteriores de la conversación:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        if not self.tiene_conocimiento:
            raise RuntimeError("Primero carga conocimiento con cargar_conocimiento().")

        instrucciones = (
            INSTRUCCIONES_MODO_LIBRE if permitir_conocimiento_externo else INSTRUCCIONES_MODO_ESTRICTO
        )
        parametros_de_la_peticion = {
            **self.parametros_base(),
            "messages": [
                # 1) El mensaje de sistema empieza con el conocimiento (prefijo estable).
                {"role": "system", "content": self.prompt_de_sistema(instrucciones, romper_cache)},
                # 2) Al final, lo que cambia en cada turno: la conversación y la pregunta.
                *(historial or []),
                {"role": "user", "content": pregunta},
            ],
        }
        return RespuestaEnVivo(
            self.cliente,
            parametros_de_la_peticion,
            pregunta,
            # El prompt siempre incluye el libro completo: si Ollama dice que leyó
            # menos tokens que el libro, lo recortó en silencio. (Damos un margen
            # porque las instrucciones del modo libre son más cortas.)
            tokens_minimos_esperados=self.informe_de_carga.tokens_en_memoria - TOLERANCIA_DE_TOKENS,
            romper_cache=romper_cache,
        )

    def preguntar(self, pregunta: str, historial: list[dict] | None = None, **opciones) -> RespuestaDelModelo:
        """Versión sin streaming: espera a que termine y devuelve la respuesta completa."""
        respuesta_en_vivo = self.preguntar_en_vivo(pregunta, historial, **opciones)
        for _fragmento in respuesta_en_vivo:
            pass  # consumimos el grifo hasta el final
        return respuesta_en_vivo.resultado


class RespuestaEnVivo:
    """
    Un "grifo" de texto: al recorrerlo con `for`, entrega la respuesta a medida
    que el modelo la escribe. Cuando termina, `resultado` contiene la
    RespuestaDelModelo completa con sus tiempos. Lo usan CAG y RAG.
    """

    def __init__(
        self,
        cliente: ollama.Client,
        parametros_de_la_peticion: dict,
        pregunta: str,
        tokens_minimos_esperados: int = 0,
        romper_cache: bool = False,
    ):
        self._cliente = cliente
        self._parametros = parametros_de_la_peticion
        self._pregunta = pregunta
        self._tokens_minimos_esperados = tokens_minimos_esperados
        self._romper_cache = romper_cache
        self.resultado: RespuestaDelModelo | None = None

    def __iter__(self) -> Iterator[str]:
        inicio = time.perf_counter()
        segundos_hasta_primer_token = None
        textos, pensamientos = [], []
        ultimo_fragmento = None

        for fragmento in self._cliente.chat(**self._parametros, stream=True):
            # Los modelos que razonan primero "piensan" (no se muestra en el chat)
            # y después escriben la respuesta.
            if fragmento.message.thinking:
                pensamientos.append(fragmento.message.thinking)
            if fragmento.message.content:
                if segundos_hasta_primer_token is None:
                    segundos_hasta_primer_token = time.perf_counter() - inicio
                textos.append(fragmento.message.content)
                yield fragmento.message.content
            ultimo_fragmento = fragmento  # el último trae los tiempos medidos

        medicion = MedicionDeTiempos.desde_respuesta(ultimo_fragmento)
        self.resultado = RespuestaDelModelo(
            pregunta=self._pregunta,
            texto="".join(textos),
            medicion=medicion,
            modelo=self._parametros["model"],
            segundos_hasta_primer_token=segundos_hasta_primer_token,
            segundos_totales=time.perf_counter() - inicio,
            razonamiento="".join(pensamientos),
            # 🚨 Si Ollama leyó menos tokens de los que sabemos que tiene el
            # prompt, es que lo recortó en silencio.
            prompt_recortado=medicion.tokens_del_prompt < self._tokens_minimos_esperados,
            cache_rota_a_proposito=self._romper_cache,
        )


# ════════════════════════════════════════════════════════════════════════════
# 🩺 ERRORES EXPLICADOS EN LENGUAJE HUMANO
# ════════════════════════════════════════════════════════════════════════════


def explicar_error(error: Exception) -> str:
    """Traduce los errores técnicos más comunes a un mensaje comprensible."""
    if isinstance(error, ConnectionError):
        return (
            "🔌 No encuentro Ollama. ¿Está encendido? Ábrelo o ejecuta `ollama serve` "
            "en una terminal. Si no lo tienes, descárgalo gratis en https://ollama.com/download"
        )
    if isinstance(error, ollama.ResponseError):
        if error.status_code == 404:
            return f"📦 Ese modelo no está instalado. Descárgalo con: `ollama pull {MODELO_RECOMENDADO}`"
        return f"⚠️ Ollama respondió con un error ({error.status_code}): {error.error}"
    return f"❌ Error inesperado: {error}"


# ════════════════════════════════════════════════════════════════════════════
# 🧪 DEMOSTRACIÓN EN TERMINAL:  python cag_core.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    motor = MotorCAG()

    print(f"📚 Inyectando el corpus de ejemplo en la caché de {motor.modelo.nombre}…")
    informe = motor.cargar_conocimiento(cargar_documentos_de_carpeta(CARPETA_CORPUS_DE_EJEMPLO))
    frio = informe.lectura_en_frio
    print(f"   Mesa (num_ctx): {informe.num_ctx:,} tokens | libro: {informe.tokens_en_memoria:,} tokens")
    print(f"   Lectura en frío: {frio.segundos_leyendo:.2f} s ({frio.velocidad_de_lectura:,.0f} tokens/s)")

    for pregunta in [
        "¿Cuántas horas fermenta el café y hasta qué humedad se seca?",
        "¿Cuántos días de vacaciones tiene un empleado en su tercer año?",
    ]:
        print(f"\n❓ {pregunta}\n💬 ", end="")
        respuesta_en_vivo = motor.preguntar_en_vivo(pregunta)
        for fragmento in respuesta_en_vivo:
            print(fragmento, end="", flush=True)
        medicion = respuesta_en_vivo.resultado.medicion
        sin_cache = medicion.segundos_sin_cache(motor.velocidad_de_lectura_en_frio)
        print(
            f"\n   ⏱️ Leyó {medicion.tokens_del_prompt:,} tokens en {medicion.segundos_leyendo:.2f} s "
            f"(sin caché: ~{sin_cache:.2f} s)"
        )
