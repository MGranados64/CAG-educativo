"""
cag_core.py — El corazón del sistema CAG (Cache-Augmented Generation)
======================================================================

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

¿Y la "caché"? Leer un libro entero cada vez cuesta tiempo y dinero. El
"Prompt Caching" de Anthropic permite que el modelo lo "lea" una sola vez: el
servidor guarda el trabajo ya hecho (lo que técnicamente se llama KV Cache) y
las siguientes preguntas lo reutilizan. Es como un estudiante con memoria
fotográfica: lee el libro una vez y después solo lo "recuerda".

🧩 CÓMO FLUYE LA INFORMACIÓN EN ESTE ARCHIVO
--------------------------------------------
  1. MotorCAG.cargar_conocimiento() → junta tus documentos en un solo bloque de
     texto (`conocimiento_en_cache`), cuenta sus tokens y "precalienta" la
     caché del servidor.
  2. MotorCAG.preguntar_en_vivo()  → envía [conocimiento + instrucciones +
     pregunta] y devuelve la respuesta palabra por palabra (streaming).
  3. RespuestaCAG                   → la respuesta final más su "recibo":
     cuántos tokens salieron de la caché y cuánto costó (y cuánto habría
     costado sin caché).

📏 LA REGLA DE ORO DEL PROMPT CACHING
------------------------------------
La caché funciona por PREFIJO: el servidor compara tu petición byte por byte
desde el principio. Si el inicio es idéntico al de una petición anterior,
reutiliza ese trabajo. Por eso el conocimiento va SIEMPRE primero y nunca
cambia, y lo que sí cambia (instrucciones, preguntas) va DESPUÉS.

    ┌───────────────────────────────────┬───────────────┬──────────────┐
    │ 📚 Conocimiento (siempre idéntico) │ Instrucciones │   Pregunta   │
    └───────────────────────────────────┴───────────────┴──────────────┘
                                        ▲
             aquí va la "marca de caché": el servidor guarda todo lo
             que está a su izquierda para reutilizarlo en la siguiente
             pregunta.

Pruébalo desde la terminal (necesitas la variable ANTHROPIC_API_KEY):

    python cag_core.py
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import anthropic

# ════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

# El modelo que "memoriza" el conocimiento y responde. Puedes cambiarlo con la
# variable de entorno CAG_MODELO sin tocar el código.
MODELO = os.environ.get("CAG_MODELO", "claude-opus-5-5")

# La "mesa de estudio" del modelo: cuántos tokens caben a la vez en su memoria
# de trabajo (ventana de contexto). Un token es un pedacito de palabra
# (≈ 4 caracteres en español). 1,000,000 de tokens ≈ 10 novelas largas.
VENTANA_DE_CONTEXTO_TOKENS = 1_000_000

# No llenamos la mesa hasta el borde: dejamos espacio para la conversación y la
# respuesta. Si tus documentos no caben aquí, CAG deja de ser la herramienta
# adecuada y conviene usar RAG (buscar solo los fragmentos necesarios).
LIMITE_SEGURO_DE_CONOCIMIENTO = 900_000

# El servidor no se molesta en guardar en caché textos muy cortos: por debajo de
# este tamaño la "marca de caché" se ignora sin avisar (no hay error, pero
# tampoco hay ahorro). 512 tokens es el mínimo para claude-opus-5-5.
MINIMO_TOKENS_PARA_CACHE = 512

# Cuánto "se esfuerza" el modelo en pensar antes de responder:
# "low" | "medium" | "high" | "xhigh" | "max". Más esfuerzo = más calidad en
# problemas difíciles, pero más lento y caro. Usamos el MISMO valor en todas las
# llamadas porque cambiarlo entre peticiones puede invalidar la caché.
ESFUERZO = "medium"

# El modelo siempre razona internamente antes de contestar. Con
# "summarized" nos devuelve un RESUMEN legible de ese razonamiento, que
# mostramos en el panel de transparencia. También es idéntico en todas las
# llamadas, por la misma razón que ESFUERZO.
PENSAMIENTO = {"type": "adaptive", "display": "summarized"}

# Límite de longitud de cada respuesta (en tokens de salida, razonamiento
# interno incluido).
MAX_TOKENS_RESPUESTA = 16_000

# 🛟 Red de seguridad: los filtros de seguridad del modelo pueden rechazar una
# petición (a veces por error). Con `fallbacks="default"` el propio servidor
# reintenta con un modelo de respaldo recomendado por Anthropic, en la misma
# llamada, en vez de devolvernos un rechazo. Requiere esta cabecera "beta".
BETAS = ["server-side-fallback-2026-07-01"]
RESPALDO_EN_SERVIDOR = "default"

# Cuánto tiempo guarda el servidor la caché desde su último uso.
DURACIONES_DE_CACHE = {"5m": "5 minutos", "1h": "1 hora"}

# Precios en dólares por cada MILLÓN de tokens (claude-opus-5-5, sept. 2026).
# Leer de la caché cuesta 20 veces menos que leer "en frío": ahí está el ahorro.
# Escribir en la caché cuesta un poco más que leer normal (es la "inversión").
PRECIOS_USD_POR_MILLON = {
    "claude-opus-5-5": {
        "entrada": 4.00,  # tokens leídos sin caché
        "salida": 20.00,  # tokens generados (incluye el razonamiento interno)
        "lectura_cache": 0.20,  # tokens recuperados de la caché
        "escritura_cache_5m": 5.00,  # guardar en caché por 5 minutos (1.25×)
        "escritura_cache_1h": 8.00,  # guardar en caché por 1 hora (2×)
    },
}

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
"Esa información no aparece en los documentos cargados") y, si ayuda, menciona \
qué temas sí cubren.
- Cuando sea útil, indica de qué documento o sección sale el dato.
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
class ReciboDeTokens:
    """
    El "recibo" de una llamada al modelo: de dónde salió cada token de entrada
    y cuántos tokens generó. Con él calculamos el costo y el ahorro.

    Analogía: en una fotocopiadora, imprimir el original la primera vez es caro
    (escribir en caché); las copias siguientes son baratísimas (leer de caché).
    """

    leidos_de_cache: int  # tokens que el servidor ya tenía "memorizados"
    escritos_en_cache: int  # tokens que el servidor acaba de "memorizar"
    sin_cache: int  # tokens procesados desde cero (p. ej. tu pregunta)
    de_salida: int  # tokens generados: razonamiento interno + respuesta
    duracion_cache: str  # "5m" o "1h" (cambia el precio de escritura)

    @classmethod
    def desde_uso(cls, uso, duracion_cache: str) -> "ReciboDeTokens":
        """Traduce el objeto `usage` que devuelve la API a nuestro recibo."""
        return cls(
            leidos_de_cache=uso.cache_read_input_tokens or 0,
            escritos_en_cache=uso.cache_creation_input_tokens or 0,
            sin_cache=uso.input_tokens,
            de_salida=uso.output_tokens,
            duracion_cache=duracion_cache,
        )

    @property
    def total_de_entrada(self) -> int:
        """Todo lo que el modelo 'leyó', venga o no de la caché."""
        return self.leidos_de_cache + self.escritos_en_cache + self.sin_cache

    @property
    def fraccion_desde_cache(self) -> float:
        """Qué parte de la lectura se sirvió desde la caché (0.0 a 1.0)."""
        if self.total_de_entrada == 0:
            return 0.0
        return self.leidos_de_cache / self.total_de_entrada

    def costo_usd(self, modelo: str) -> float | None:
        """Costo real de la llamada. None si no conocemos los precios del modelo."""
        precios = PRECIOS_USD_POR_MILLON.get(modelo)
        if precios is None:
            return None
        precio_escritura = precios[f"escritura_cache_{self.duracion_cache}"]
        return (
            self.sin_cache * precios["entrada"]
            + self.escritos_en_cache * precio_escritura
            + self.leidos_de_cache * precios["lectura_cache"]
            + self.de_salida * precios["salida"]
        ) / 1_000_000

    def costo_sin_cache_usd(self, modelo: str) -> float | None:
        """Lo que habría costado si TODO se hubiera leído en frío, sin caché."""
        precios = PRECIOS_USD_POR_MILLON.get(modelo)
        if precios is None:
            return None
        return (
            self.total_de_entrada * precios["entrada"]
            + self.de_salida * precios["salida"]
        ) / 1_000_000


@dataclass
class InformeDeCarga:
    """Lo que pasó al inyectar el conocimiento en la caché."""

    documentos: list[str]
    caracteres: int
    tokens_de_conocimiento: int
    fraccion_de_ventana: float  # qué parte de la ventana de contexto ocupa
    es_cacheable: bool  # ¿supera el mínimo de tokens para caché?
    precalentada: bool  # ¿logramos escribir la caché por adelantado?
    recibo_precalentamiento: ReciboDeTokens | None
    aviso: str | None = None


@dataclass
class RespuestaCAG:
    """Una respuesta del modelo con toda la información para auditarla."""

    pregunta: str
    texto: str
    recibo: ReciboDeTokens
    modelo_que_respondio: str  # puede ser el de respaldo si hubo un rechazo
    segundos_hasta_primer_token: float | None
    segundos_totales: float
    razonamiento_resumido: str  # resumen del "pensamiento" interno del modelo
    fue_rechazada: bool = False  # True si los filtros de seguridad la bloquearon


class ConocimientoDemasiadoGrande(ValueError):
    """Los documentos no caben en la memoria de trabajo del modelo."""


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


# ════════════════════════════════════════════════════════════════════════════
# 🧠 EL MOTOR CAG
# ════════════════════════════════════════════════════════════════════════════


class MotorCAG:
    """
    Guarda el conocimiento "memorizado" y responde preguntas sobre él.

    Uso mínimo:
        motor = MotorCAG()
        motor.cargar_conocimiento([Documento("manual.md", "...")])
        respuesta = motor.preguntar("¿Cuántos días de vacaciones tengo?")
        print(respuesta.texto)
    """

    def __init__(self, api_key: str | None = None, cliente: anthropic.Anthropic | None = None):
        # El "cliente" es el teléfono con el que llamamos a la API de Anthropic.
        # Sin api_key explícita, lo busca solo en la variable ANTHROPIC_API_KEY.
        if cliente is not None:
            self.cliente = cliente
        elif api_key:
            self.cliente = anthropic.Anthropic(api_key=api_key)
        else:
            self.cliente = anthropic.Anthropic()

        self.conocimiento_en_cache: str = ""  # el "libro" que el modelo memoriza
        self.documentos_cargados: list[str] = []
        self.duracion_cache: str = "5m"
        self.informe_de_carga: InformeDeCarga | None = None

    @property
    def tiene_conocimiento(self) -> bool:
        return bool(self.conocimiento_en_cache)

    # ── Piezas reutilizables de cada petición ───────────────────────────────

    def marca_de_cache(self) -> dict:
        """
        La etiqueta `cache_control` que le dice al servidor: "guarda todo lo que
        va desde el inicio de la petición hasta aquí".
        """
        marca = {"type": "ephemeral"}
        if self.duracion_cache == "1h":
            marca["ttl"] = "1h"
        return marca

    def bloque_de_conocimiento(self) -> dict:
        """
        El conocimiento listo para enviarse, con su marca de caché pegada.
        Siempre es el PRIMER bloque de la petición (regla del prefijo).
        """
        return {
            "type": "text",
            "text": self.conocimiento_en_cache,
            "cache_control": self.marca_de_cache(),
        }

    def parametros_base(self) -> dict:
        """
        Parámetros idénticos en TODAS las llamadas (precalentamiento, chat y
        jueces). Mantenerlos iguales es lo que permite reutilizar la caché:
        cambiar de modelo o de forma de razonar es como cambiar de estudiante,
        y el nuevo no recuerda lo que memorizó el anterior.
        """
        return {
            "model": MODELO,
            "thinking": PENSAMIENTO,
            "betas": BETAS,
            "fallbacks": RESPALDO_EN_SERVIDOR,
        }

    # ── Paso 1: inyectar el conocimiento ────────────────────────────────────

    def cargar_conocimiento(
        self, documentos: list[Documento], duracion_cache: str = "5m"
    ) -> InformeDeCarga:
        """
        "Entrega el libro al estudiante antes del examen".

        1. Junta los documentos en `conocimiento_en_cache`.
        2. Cuenta sus tokens (gratis) para saber si caben en la memoria.
        3. Precalienta la caché: una llamada con `max_tokens=0` hace que el
           servidor lea y guarde el conocimiento SIN generar respuesta. Así tu
           primera pregunta real ya encuentra el libro memorizado.
        """
        if not documentos:
            raise ValueError("No hay documentos que cargar.")
        if duracion_cache not in DURACIONES_DE_CACHE:
            raise ValueError(f"Duración de caché no válida: {duracion_cache!r}")

        conocimiento = construir_conocimiento_en_cache(documentos)

        # ¿Cabe en la mesa de estudio? count_tokens no genera texto ni cobra.
        conteo = self.cliente.messages.count_tokens(
            model=MODELO,
            system=[{"type": "text", "text": conocimiento}],
            messages=[{"role": "user", "content": "."}],
        )
        tokens_de_conocimiento = conteo.input_tokens
        if tokens_de_conocimiento > LIMITE_SEGURO_DE_CONOCIMIENTO:
            raise ConocimientoDemasiadoGrande(
                f"Tus documentos ocupan ~{tokens_de_conocimiento:,} tokens y el límite "
                f"seguro es {LIMITE_SEGURO_DE_CONOCIMIENTO:,}. Esta es justo la "
                "limitación de CAG: todo el conocimiento debe caber en la memoria "
                "del modelo. Para bibliotecas más grandes se usa RAG."
            )

        # Guardamos el "libro". A partir de aquí, cada pregunta lo incluye.
        self.conocimiento_en_cache = conocimiento
        self.documentos_cargados = sorted(documento.nombre for documento in documentos)
        self.duracion_cache = duracion_cache

        es_cacheable = tokens_de_conocimiento >= MINIMO_TOKENS_PARA_CACHE
        precalentada = False
        recibo_precalentamiento = None
        aviso = None

        if not es_cacheable:
            aviso = (
                f"El conocimiento mide ~{tokens_de_conocimiento} tokens, menos del "
                f"mínimo de {MINIMO_TOKENS_PARA_CACHE} que el servidor guarda en "
                "caché. Funcionará igual, pero se leerá 'en frío' en cada pregunta."
            )
        else:
            try:
                mensaje = self.cliente.beta.messages.create(
                    **self.parametros_base(),
                    max_tokens=0,  # 👈 solo leer y memorizar, no responder
                    output_config={"effort": ESFUERZO},
                    system=[self.bloque_de_conocimiento()],
                    # Texto de relleno: el modelo lo lee pero nunca lo contesta.
                    messages=[{"role": "user", "content": "precalentamiento"}],
                )
                recibo_precalentamiento = ReciboDeTokens.desde_uso(mensaje.usage, duracion_cache)
                precalentada = True
            except anthropic.BadRequestError as error:
                # No es grave: la primera pregunta real escribirá la caché.
                aviso = (
                    "No se pudo precalentar la caché "
                    f"({error.message}). Se escribirá con tu primera pregunta."
                )

        self.informe_de_carga = InformeDeCarga(
            documentos=self.documentos_cargados,
            caracteres=len(conocimiento),
            tokens_de_conocimiento=tokens_de_conocimiento,
            fraccion_de_ventana=tokens_de_conocimiento / VENTANA_DE_CONTEXTO_TOKENS,
            es_cacheable=es_cacheable,
            precalentada=precalentada,
            recibo_precalentamiento=recibo_precalentamiento,
            aviso=aviso,
        )
        return self.informe_de_carga

    # ── Paso 2: preguntar ───────────────────────────────────────────────────

    def preguntar_en_vivo(
        self,
        pregunta: str,
        historial: list[dict] | None = None,
        permitir_conocimiento_externo: bool = False,
    ) -> "RespuestaEnVivo":
        """
        Prepara la petición y devuelve un objeto que entrega la respuesta
        pedacito a pedacito (streaming), como un grifo que abres y del que
        sale el agua poco a poco.

        `historial` son los turnos anteriores de la conversación, en el
        formato de la API: [{"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}, ...]
        """
        if not self.tiene_conocimiento:
            raise RuntimeError("Primero carga conocimiento con cargar_conocimiento().")

        instrucciones = (
            INSTRUCCIONES_MODO_LIBRE if permitir_conocimiento_externo else INSTRUCCIONES_MODO_ESTRICTO
        )
        parametros_de_la_peticion = {
            **self.parametros_base(),
            "max_tokens": MAX_TOKENS_RESPUESTA,
            "output_config": {"effort": ESFUERZO},
            # El orden importa: 1) conocimiento con marca de caché (idéntico
            # siempre), 2) instrucciones (pueden cambiar sin romper la caché).
            "system": [
                self.bloque_de_conocimiento(),
                {"type": "text", "text": instrucciones},
            ],
            # Al final, lo que cambia en cada turno: la conversación y la pregunta.
            "messages": [*(historial or []), {"role": "user", "content": pregunta}],
        }
        return RespuestaEnVivo(self, parametros_de_la_peticion, pregunta)

    def preguntar(
        self,
        pregunta: str,
        historial: list[dict] | None = None,
        permitir_conocimiento_externo: bool = False,
    ) -> RespuestaCAG:
        """Versión sin streaming: espera a que termine y devuelve la respuesta completa."""
        respuesta_en_vivo = self.preguntar_en_vivo(pregunta, historial, permitir_conocimiento_externo)
        for _fragmento in respuesta_en_vivo:
            pass  # consumimos el grifo hasta el final
        return respuesta_en_vivo.resultado


class RespuestaEnVivo:
    """
    Un "grifo" de texto: al recorrerlo con `for`, entrega la respuesta a medida
    que el modelo la escribe. Cuando termina, `resultado` contiene la
    RespuestaCAG completa con su recibo de tokens.
    """

    def __init__(self, motor: MotorCAG, parametros_de_la_peticion: dict, pregunta: str):
        self._motor = motor
        self._parametros = parametros_de_la_peticion
        self._pregunta = pregunta
        self.resultado: RespuestaCAG | None = None

    def __iter__(self) -> Iterator[str]:
        inicio = time.perf_counter()
        segundos_hasta_primer_token = None

        with self._motor.cliente.beta.messages.stream(**self._parametros) as flujo:
            for fragmento_de_texto in flujo.text_stream:
                if segundos_hasta_primer_token is None:
                    # ⏱️ Latencia percibida: cuánto tardó en aparecer la primera
                    # palabra. Con el conocimiento en caché, el modelo no tiene
                    # que volver a "leer el libro" y empieza antes.
                    segundos_hasta_primer_token = time.perf_counter() - inicio
                yield fragmento_de_texto
            mensaje_final = flujo.get_final_message()

        # La respuesta llega en "bloques": de texto (lo que ves) y de
        # pensamiento (el resumen de su razonamiento interno).
        texto = "".join(bloque.text for bloque in mensaje_final.content if bloque.type == "text")
        razonamiento = "\n\n".join(
            bloque.thinking
            for bloque in mensaje_final.content
            if bloque.type == "thinking" and bloque.thinking
        )

        self.resultado = RespuestaCAG(
            pregunta=self._pregunta,
            texto=texto,
            recibo=ReciboDeTokens.desde_uso(mensaje_final.usage, self._motor.duracion_cache),
            modelo_que_respondio=mensaje_final.model,
            segundos_hasta_primer_token=segundos_hasta_primer_token,
            segundos_totales=time.perf_counter() - inicio,
            razonamiento_resumido=razonamiento,
            # "refusal" = incluso el modelo de respaldo rechazó la petición.
            fue_rechazada=mensaje_final.stop_reason == "refusal",
        )


# ════════════════════════════════════════════════════════════════════════════
# 🩺 ERRORES EXPLICADOS EN LENGUAJE HUMANO
# ════════════════════════════════════════════════════════════════════════════


def explicar_error(error: Exception) -> str:
    """Traduce los errores técnicos más comunes a un mensaje comprensible."""
    # Vamos de lo más específico a lo más general.
    if isinstance(error, anthropic.AuthenticationError):
        return "🔑 La API key no es válida. Revísala en la barra lateral o en ANTHROPIC_API_KEY."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "🚫 Tu API key no tiene permiso para usar este modelo o función."
    if isinstance(error, anthropic.NotFoundError):
        return f"🔎 No se encontró el modelo '{MODELO}'. Revisa la variable CAG_MODELO."
    if isinstance(error, anthropic.RateLimitError):
        return "⏳ Demasiadas peticiones seguidas. Espera un minuto y vuelve a intentarlo."
    if isinstance(error, anthropic.BadRequestError):
        return f"📝 La API rechazó la petición: {error.message}"
    if isinstance(error, anthropic.APIStatusError):
        if error.status_code >= 500:
            return "🛠️ El servidor de Anthropic tuvo un problema temporal. Intenta de nuevo."
        return f"⚠️ Error de la API ({error.status_code}): {error.message}"
    if isinstance(error, anthropic.APIConnectionError):
        return "🌐 No hay conexión con la API de Anthropic. Revisa tu internet."
    if "authentication" in str(error).lower() or "api_key" in str(error).lower():
        return (
            "🔑 No encontré credenciales. Escribe tu API key en la barra lateral o "
            "define la variable de entorno ANTHROPIC_API_KEY."
        )
    return f"❌ Error inesperado: {error}"


# ════════════════════════════════════════════════════════════════════════════
# 🧪 DEMOSTRACIÓN EN TERMINAL:  python cag_core.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ruta_ejemplo = Path(__file__).parent / "datos" / "manual_cooperativa_nube_alta.md"
    motor = MotorCAG()

    print("📚 Inyectando el documento de ejemplo en la caché…")
    informe = motor.cargar_conocimiento([Documento(ruta_ejemplo.name, ruta_ejemplo.read_text("utf-8"))])
    print(f"   Tokens memorizados: {informe.tokens_de_conocimiento:,}")
    if informe.recibo_precalentamiento:
        print(f"   Escritos en caché (precalentamiento): {informe.recibo_precalentamiento.escritos_en_cache:,}")
    if informe.aviso:
        print(f"   Aviso: {informe.aviso}")

    for pregunta in [
        "¿Cuántas horas fermenta el café y hasta qué humedad se seca?",
        "¿Cuántos días de vacaciones tiene un empleado en su tercer año?",
    ]:
        print(f"\n❓ {pregunta}\n💬 ", end="")
        respuesta_en_vivo = motor.preguntar_en_vivo(pregunta)
        for fragmento in respuesta_en_vivo:
            print(fragmento, end="", flush=True)
        recibo = respuesta_en_vivo.resultado.recibo
        print(
            f"\n   🧾 Leídos de caché: {recibo.leidos_de_cache:,} | "
            f"Escritos en caché: {recibo.escritos_en_cache:,} | "
            f"Sin caché: {recibo.sin_cache:,} | Salida: {recibo.de_salida:,}"
        )
