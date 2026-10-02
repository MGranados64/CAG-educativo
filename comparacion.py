"""
comparacion.py — CAG vs RAG, cara a cara
=========================================

🥊 LAS REGLAS DE UNA PELEA JUSTA
--------------------------------
Para que la comparación enseñe algo, todo debe ser igual excepto lo que
queremos medir:

  • mismos documentos y mismas preguntas (con su respuesta correcta conocida);
  • mismo modelo de chat, misma mesa (num_ctx), mismo nivel de razonamiento y
    temperatura 0;
  • mismos jueces para calificar a ambos.

La ÚNICA diferencia: CAG lee TODOS los documentos (desde su caché) y RAG solo
los fragmentos que trae su buscador.

📏 LO QUE MEDIMOS EN CADA PREGUNTA
----------------------------------
  ✅ Exactitud     ¿coincide con la respuesta correcta? (juez de exactitud)
  ⚖️ Veracidad     ¿todo lo que dijo está en los documentos? (juez de veracidad)
  🔎 Recuperación  solo RAG: ¿el buscador trajo lo necesario? (juez de recuperación)
  ⏱️ Tiempos       lectura del prompt, búsqueda y tiempo total
  📖 Tokens        cuánto texto tuvo que leer cada uno

Todo se exporta a un archivo JSON en la carpeta resultados/ para que puedas
analizarlo o reutilizarlo en otros proyectos.

Ejecuta (con Ollama encendido):

    python comparacion.py cooperativa
    python comparacion.py paper_cag --descargar   # baja el paper de arXiv la primera vez
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

from cag_core import (
    MODELO_RECOMENDADO,
    Documento,
    MotorCAG,
    RespuestaDelModelo,
    cargar_documentos_de_carpeta,
)
from evals import (
    ResultadoConVeredicto,
    ResultadoVeracidad,
    evaluar_exactitud,
    evaluar_recuperacion,
    evaluar_veracidad,
)
from rag_core import CARACTERES_POR_FRAGMENTO, FRAGMENTOS_A_RECUPERAR, MotorRAG

CARPETA_DATOS = Path(__file__).parent / "datos"
CARPETA_RESULTADOS = Path(__file__).parent / "resultados"

# ════════════════════════════════════════════════════════════════════════════
# 📚 LOS CORPUS DE PRUEBA
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Corpus:
    """Un conjunto de documentos con sus preguntas de referencia (preguntas.json)."""

    clave: str
    nombre: str
    carpeta: Path
    descripcion: str
    url_de_descarga: str | None = None  # si los documentos se descargan bajo demanda

    @property
    def disponible(self) -> bool:
        """¿Ya están los documentos en la carpeta?"""
        return bool(cargar_documentos_de_carpeta(self.carpeta))

    def documentos(self) -> list[Documento]:
        return cargar_documentos_de_carpeta(self.carpeta)

    def preguntas(self) -> dict:
        return json.loads((self.carpeta / "preguntas.json").read_text("utf-8"))


CORPUS = {
    "cooperativa": Corpus(
        clave="cooperativa",
        nombre="Cooperativa Nube Alta (ficticio, español)",
        carpeta=CARPETA_DATOS / "cooperativa",
        descripcion=(
            "Cuatro documentos inventados: el modelo no puede saber las respuestas de memoria. "
            "Incluye trampas para RAG: cifras de varios años, preguntas que necesitan dos "
            "documentos y preguntas con palabras distintas a las del texto."
        ),
    ),
    "paper_cag": Corpus(
        clave="paper_cag",
        nombre="Paper de CAG — Chan et al., 2025 (inglés, CC BY-SA 4.0)",
        carpeta=CARPETA_DATOS / "paper_cag",
        descripcion=(
            "El artículo real que propuso CAG. Es posterior al entrenamiento de gpt-oss, así que "
            "el modelo tampoco puede responder de memoria. Se descarga de arXiv bajo demanda."
        ),
        url_de_descarga="https://arxiv.org/html/2412.15605v2",
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 📥 DESCARGAR EL PAPER (con su atribución y licencia)
# ════════════════════════════════════════════════════════════════════════════

ENCABEZADO_DE_ATRIBUCION = """\
# Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks

> Autores: Brian J. Chan, Chao-Ting Chen, Jui-Hung Cheng y Hen-Hsen Huang (2025).
> The ACM Web Conference 2025 (WWW '25), artículo corto. https://arxiv.org/abs/2412.15605
> Licencia: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
> Texto extraído automáticamente de la versión HTML de arXiv (v2) para fines
> educativos; puede contener errores de formato. Este archivo conserva la
> licencia CC BY-SA 4.0 del original.
"""


class _ExtractorDeTexto(HTMLParser):
    """
    Convierte la página HTML del paper en texto plano tipo Markdown: conserva
    títulos ("#"), párrafos, listas y tablas, y descarta menús, scripts y la
    bibliografía (que solo agregaría ruido).
    """

    ETIQUETAS_SIN_CIERRE = {"br", "img", "meta", "link", "input", "hr", "col", "area", "base", "source", "wbr"}
    ETIQUETAS_IGNORADAS = {"script", "style", "nav", "header", "footer", "button", "svg"}
    CLASES_IGNORADAS = ("ltx_bibliography", "ltx_page_footer", "ltx_authors", "ltx_dates")
    BLOQUES = {"p", "div", "section", "table", "tr", "figcaption", "blockquote", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self.partes: list[str] = []
        self._pila: list[tuple[str, bool]] = []  # (etiqueta, ¿se ignora su contenido?)

    @property
    def _ignorando(self) -> bool:
        return bool(self._pila) and self._pila[-1][1]

    def handle_starttag(self, etiqueta, atributos):
        atributos = dict(atributos)
        if etiqueta in self.ETIQUETAS_SIN_CIERRE:
            return
        clases = atributos.get("class") or ""
        ignorar = (
            self._ignorando
            or etiqueta in self.ETIQUETAS_IGNORADAS
            or etiqueta == "math"  # usamos su versión en texto (alttext) en lugar del MathML
            or any(clase in clases for clase in self.CLASES_IGNORADAS)
        )
        if etiqueta == "math" and not self._ignorando:
            self.partes.append(f" {atributos.get('alttext', '')} ")
        self._pila.append((etiqueta, ignorar))
        if ignorar:
            return
        if re.fullmatch(r"h[1-6]", etiqueta):
            self.partes.append("\n\n" + "#" * int(etiqueta[1]) + " ")
        elif etiqueta == "li":
            self.partes.append("\n- ")
        elif etiqueta in ("td", "th"):
            self.partes.append(" | ")
        elif etiqueta in self.BLOQUES:
            self.partes.append("\n\n")

    def handle_endtag(self, etiqueta):
        # Cerramos hasta encontrar la etiqueta que abrió (tolera HTML imperfecto).
        while self._pila:
            abierta, ignorada = self._pila.pop()
            if not ignorada and (re.fullmatch(r"h[1-6]", abierta) or abierta in self.BLOQUES):
                self.partes.append("\n\n")
            if abierta == etiqueta:
                break

    def handle_data(self, datos):
        if not self._ignorando:
            self.partes.append(datos)

    def texto(self) -> str:
        texto = "".join(self.partes)
        texto = re.sub(r"[ \t\r\f\v]+", " ", texto)  # espacios repetidos → uno
        texto = re.sub(r" *\n *", "\n", texto)  # sin espacios al borde de las líneas
        return re.sub(r"\n{3,}", "\n\n", texto).strip()  # máximo una línea en blanco


def descargar_paper(corpus: Corpus) -> Path:
    """
    Baja el paper de arXiv, lo convierte en texto y lo guarda con su atribución.
    No lo incluimos en el repositorio: cada persona lo descarga para su propio uso.
    """
    peticion = urllib.request.Request(corpus.url_de_descarga, headers={"User-Agent": "CAG-Educativo/1.0"})
    with urllib.request.urlopen(peticion, timeout=60) as respuesta:
        html = respuesta.read().decode("utf-8", errors="replace")
    extractor = _ExtractorDeTexto()
    extractor.feed(html)
    destino = corpus.carpeta / "paper_cag.md"
    destino.write_text(ENCABEZADO_DE_ATRIBUCION + "\n" + extractor.texto() + "\n", encoding="utf-8")
    return destino


# ════════════════════════════════════════════════════════════════════════════
# 🥊 LA PELEA
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ResultadoCaraACara:
    """Todo lo que pasó con UNA pregunta de referencia."""

    pregunta: dict  # la pregunta de preguntas.json (con su respuesta esperada y su tipo)
    cag: RespuestaDelModelo
    rag: RespuestaDelModelo
    exactitud_cag: ResultadoConVeredicto
    exactitud_rag: ResultadoConVeredicto
    veracidad_cag: ResultadoVeracidad
    veracidad_rag: ResultadoVeracidad
    recuperacion_rag: ResultadoConVeredicto | None  # None si la respuesta no está en los documentos


def preparar_contendientes(
    documentos: list[Documento],
    modelo: str = MODELO_RECOMENDADO,
    razonar: bool = True,
    fragmentos_a_recuperar: int = FRAGMENTOS_A_RECUPERAR,
    duracion_cache: str = "10m",
) -> tuple[MotorCAG, MotorRAG]:
    """
    Prepara a los dos rivales con los mismos documentos:
      • CAG lee todo una vez y lo guarda en su caché;
      • RAG trocea e indexa, con la MISMA mesa (num_ctx) que CAG. Si usara otra,
        Ollama tendría que recargar el modelo de chat y CAG perdería su caché.
    """
    motor_cag = MotorCAG(modelo, razonar=razonar)
    motor_cag.cargar_conocimiento(documentos, duracion_cache)
    motor_rag = MotorRAG(modelo, razonar=razonar, fragmentos_a_recuperar=fragmentos_a_recuperar, cliente=motor_cag.cliente)
    motor_rag.indexar(documentos, num_ctx=motor_cag.num_ctx, duracion_cache=duracion_cache)
    return motor_cag, motor_rag


def enfrentar(motor_cag: MotorCAG, motor_rag: MotorRAG, pregunta: dict) -> ResultadoCaraACara:
    """
    Hace la misma pregunta a ambos y llama a los jueces. Los jueces usan el
    motor CAG porque necesitan ver TODOS los documentos para auditar (y así
    reutilizan su caché).
    """
    texto_de_la_pregunta = pregunta["pregunta"]
    esperada = pregunta["respuesta_esperada"]

    respuesta_cag = motor_cag.preguntar(texto_de_la_pregunta)
    respuesta_rag = motor_rag.preguntar(texto_de_la_pregunta)

    recuperacion = None
    if pregunta["en_documentos"]:
        recuperacion = evaluar_recuperacion(
            motor_cag,
            texto_de_la_pregunta,
            esperada,
            [r.fragmento.texto_con_contexto for r in respuesta_rag.fragmentos_recuperados],
        )

    return ResultadoCaraACara(
        pregunta=pregunta,
        cag=respuesta_cag,
        rag=respuesta_rag,
        exactitud_cag=evaluar_exactitud(motor_cag, texto_de_la_pregunta, respuesta_cag.texto, esperada),
        exactitud_rag=evaluar_exactitud(motor_cag, texto_de_la_pregunta, respuesta_rag.texto, esperada),
        veracidad_cag=evaluar_veracidad(motor_cag, texto_de_la_pregunta, respuesta_cag.texto),
        veracidad_rag=evaluar_veracidad(motor_cag, texto_de_la_pregunta, respuesta_rag.texto),
        recuperacion_rag=recuperacion,
    )


def correr_comparacion(
    motor_cag: MotorCAG, motor_rag: MotorRAG, preguntas: list[dict]
) -> Iterator[ResultadoCaraACara]:
    """Enfrenta a los dos sistemas pregunta por pregunta (entrega cada resultado al terminarlo)."""
    for pregunta in preguntas:
        yield enfrentar(motor_cag, motor_rag, pregunta)


# ════════════════════════════════════════════════════════════════════════════
# 📊 RESUMEN Y EXPORTACIÓN
# ════════════════════════════════════════════════════════════════════════════


def _promedio(valores: list[float]) -> float | None:
    return sum(valores) / len(valores) if valores else None


def resumir(resultados: list[ResultadoCaraACara]) -> dict:
    """Promedios por sistema y exactitud por tipo de pregunta."""
    resumen = {}
    for sistema in ("cag", "rag"):
        respuestas = [getattr(r, sistema) for r in resultados]
        resumen[sistema] = {
            "exactitud": _promedio([getattr(r, f"exactitud_{sistema}").puntaje for r in resultados]),
            "veracidad": _promedio(
                [getattr(r, f"veracidad_{sistema}").puntaje for r in resultados if not getattr(r, f"veracidad_{sistema}").error]
            ),
            "segundos_totales": _promedio([r.segundos_totales for r in respuestas]),
            "segundos_hasta_primer_token": _promedio(
                [r.segundos_hasta_primer_token for r in respuestas if r.segundos_hasta_primer_token is not None]
            ),
            "segundos_leyendo": _promedio([r.medicion.segundos_leyendo for r in respuestas]),
            "tokens_leidos": _promedio([r.medicion.tokens_del_prompt for r in respuestas]),
        }
    resumen["rag"]["recuperacion"] = _promedio(
        [r.recuperacion_rag.puntaje for r in resultados if r.recuperacion_rag and not r.recuperacion_rag.error]
    )
    resumen["rag"]["segundos_buscando"] = _promedio([r.rag.segundos_buscando for r in resultados])

    por_tipo: dict[str, dict] = {}
    for tipo in sorted({r.pregunta["tipo"] for r in resultados}):
        del_tipo = [r for r in resultados if r.pregunta["tipo"] == tipo]
        por_tipo[tipo] = {
            "preguntas": len(del_tipo),
            "exactitud_cag": _promedio([r.exactitud_cag.puntaje for r in del_tipo]),
            "exactitud_rag": _promedio([r.exactitud_rag.puntaje for r in del_tipo]),
        }
    resumen["por_tipo"] = por_tipo
    return resumen


def exportar(
    corpus: Corpus, motor_cag: MotorCAG, motor_rag: MotorRAG, resultados: list[ResultadoCaraACara]
) -> Path:
    """Guarda TODO (configuración, preparación, resumen y cada respuesta) en un JSON."""
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    marca_de_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = CARPETA_RESULTADOS / f"comparacion_{corpus.clave}_{marca_de_tiempo}.json"
    carga = motor_cag.informe_de_carga
    indice = motor_rag.informe_de_indexado
    contenido = {
        "corpus": {"clave": corpus.clave, "nombre": corpus.nombre, "documentos": carga.documentos},
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "configuracion": {
            "modelo": motor_cag.modelo.nombre,
            "razonar": motor_cag.razonar,
            "num_ctx": motor_cag.num_ctx,
            "temperatura": 0,
            "modelo_de_embeddings": motor_rag.modelo_de_embeddings,
            "fragmentos_a_recuperar": motor_rag.fragmentos_a_recuperar,
            "caracteres_por_fragmento": CARACTERES_POR_FRAGMENTO,
        },
        "preparacion": {
            "cag": {
                "tokens_en_memoria": carga.tokens_en_memoria,
                "segundos_lectura_en_frio": carga.lectura_en_frio.segundos_leyendo,
                "segundos_cargando_modelo": carga.lectura_en_frio.segundos_cargando_modelo,
            },
            "rag": {
                "fragmentos": indice.numero_de_fragmentos,
                "caracteres_promedio": indice.caracteres_promedio,
                "dimensiones": indice.dimensiones,
                "segundos_indexando": indice.segundos_indexando,
            },
        },
        "resumen": resumir(resultados),
        "resultados": [asdict(resultado) for resultado in resultados],
    }
    destino.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


# ════════════════════════════════════════════════════════════════════════════
# 🧪 DESDE LA TERMINAL:  python comparacion.py cooperativa
# ════════════════════════════════════════════════════════════════════════════


def _porcentaje(valor: float | None) -> str:
    return "  —" if valor is None else f"{valor:4.0%}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara CAG y RAG con las mismas preguntas.")
    parser.add_argument("corpus", choices=list(CORPUS), help="qué conjunto de documentos usar")
    parser.add_argument("--modelo", default=MODELO_RECOMENDADO, help="modelo de chat de Ollama")
    parser.add_argument("--k", type=int, default=FRAGMENTOS_A_RECUPERAR, help="fragmentos que recupera RAG")
    parser.add_argument("--sin-razonar", action="store_true", help="desactiva el razonamiento previo")
    parser.add_argument("--descargar", action="store_true", help="descarga el corpus si no está")
    argumentos = parser.parse_args()

    corpus = CORPUS[argumentos.corpus]
    if not corpus.disponible:
        if not (argumentos.descargar and corpus.url_de_descarga):
            raise SystemExit(f"El corpus '{corpus.clave}' no está descargado. Añade --descargar.")
        print(f"📥 Descargando {corpus.url_de_descarga} …")
        print(f"   Guardado en {descargar_paper(corpus)}")

    preguntas = corpus.preguntas()["preguntas"]
    print(f"🥊 {corpus.nombre}: {len(preguntas)} preguntas, modelo {argumentos.modelo}, k={argumentos.k}")
    inicio = time.perf_counter()
    motor_cag, motor_rag = preparar_contendientes(
        corpus.documentos(), argumentos.modelo, not argumentos.sin_razonar, argumentos.k
    )

    resultados = []
    print(f"\n{'id':4} {'tipo':20} {'Exactitud CAG':>14} {'Exactitud RAG':>14} {'Recup. RAG':>11}")
    for resultado in correr_comparacion(motor_cag, motor_rag, preguntas):
        resultados.append(resultado)
        recuperacion = resultado.recuperacion_rag.puntaje if resultado.recuperacion_rag else None
        print(
            f"{resultado.pregunta['id']:4} {resultado.pregunta['tipo']:20} "
            f"{resultado.exactitud_cag.veredicto:>14} {resultado.exactitud_rag.veredicto:>14} "
            f"{_porcentaje(recuperacion):>11}"
        )

    resumen = resumir(resultados)
    print("\n📊 Resumen          CAG       RAG")
    for nombre, clave in [("Exactitud", "exactitud"), ("Veracidad", "veracidad")]:
        print(f"   {nombre:15} {_porcentaje(resumen['cag'][clave]):>6}    {_porcentaje(resumen['rag'][clave]):>6}")
    for nombre, clave in [("Tiempo total (s)", "segundos_totales"), ("Lectura (s)", "segundos_leyendo")]:
        print(f"   {nombre:15} {resumen['cag'][clave]:6.2f}    {resumen['rag'][clave]:6.2f}")
    print(f"   {'Tokens leídos':15} {resumen['cag']['tokens_leidos']:6.0f}    {resumen['rag']['tokens_leidos']:6.0f}")
    print(f"   Recuperación de RAG: {_porcentaje(resumen['rag']['recuperacion'])}")
    print(f"\n💾 Resultados en {exportar(corpus, motor_cag, motor_rag, resultados)}")
    print(f"⏱️ {time.perf_counter() - inicio:.0f} s en total")
