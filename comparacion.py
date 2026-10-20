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

🏆 LA LIGA DE ESTRATEGIAS
-------------------------
RAG tiene varias formas de buscar (ver rag_core.ESTRATEGIAS): clásica, híbrida,
padre-hijo y agéntica. En la liga, CAG responde cada pregunta UNA vez y luego
cada estrategia de RAG responde la misma pregunta, con los mismos jueces.

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

    python comparacion.py cooperativa                       # CAG vs RAG clásico
    python comparacion.py cooperativa --estrategia hibrido  # CAG vs RAG híbrido
    python comparacion.py cooperativa --liga                # CAG vs las 4 estrategias
    python comparacion.py paper_cag --descargar             # baja el paper de arXiv la primera vez
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
from rag_core import CARACTERES_POR_FRAGMENTO, ESTRATEGIAS, FRAGMENTOS_A_RECUPERAR, MotorRAG

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
    """Todo lo que pasó con UNA pregunta de referencia y UNA estrategia de RAG."""

    pregunta: dict  # la pregunta de preguntas.json (con su respuesta esperada y su tipo)
    estrategia: str  # la estrategia de RAG que compitió (ver rag_core.ESTRATEGIAS)
    cag: RespuestaDelModelo
    rag: RespuestaDelModelo
    exactitud_cag: ResultadoConVeredicto
    exactitud_rag: ResultadoConVeredicto
    veracidad_cag: ResultadoVeracidad
    veracidad_rag: ResultadoVeracidad
    recuperacion_rag: ResultadoConVeredicto | None  # None si la respuesta no está en los documentos


@dataclass
class TurnoDeCAG:
    """La respuesta de CAG a una pregunta, ya calificada (se reutiliza en la liga)."""

    respuesta: RespuestaDelModelo
    exactitud: ResultadoConVeredicto
    veracidad: ResultadoVeracidad


def preparar_contendientes(
    documentos: list[Documento],
    modelo: str = MODELO_RECOMENDADO,
    razonar: bool = True,
    fragmentos_a_recuperar: int = FRAGMENTOS_A_RECUPERAR,
    duracion_cache: str = "10m",
    estrategia: str = "clasico",
) -> tuple[MotorCAG, MotorRAG]:
    """
    Prepara a los dos rivales con los mismos documentos:
      • CAG lee todo una vez y lo guarda en su caché;
      • RAG trocea e indexa, con la MISMA mesa (num_ctx) que CAG. Si usara otra,
        Ollama tendría que recargar el modelo de chat y CAG perdería su caché.
    La estrategia de RAG se puede cambiar después (motor_rag.estrategia = ...).
    """
    motor_cag = MotorCAG(modelo, razonar=razonar)
    motor_cag.cargar_conocimiento(documentos, duracion_cache)
    motor_rag = MotorRAG(
        modelo,
        razonar=razonar,
        fragmentos_a_recuperar=fragmentos_a_recuperar,
        estrategia=estrategia,
        cliente=motor_cag.cliente,
    )
    motor_rag.indexar(documentos, num_ctx=motor_cag.num_ctx, duracion_cache=duracion_cache)
    return motor_cag, motor_rag


def responder_con_cag(motor_cag: MotorCAG, pregunta: dict) -> TurnoDeCAG:
    """CAG responde y los jueces lo califican."""
    texto_de_la_pregunta = pregunta["pregunta"]
    respuesta = motor_cag.preguntar(texto_de_la_pregunta)
    return TurnoDeCAG(
        respuesta=respuesta,
        exactitud=evaluar_exactitud(motor_cag, texto_de_la_pregunta, respuesta.texto, pregunta["respuesta_esperada"]),
        veracidad=evaluar_veracidad(motor_cag, texto_de_la_pregunta, respuesta.texto),
    )


def enfrentar(
    motor_cag: MotorCAG, motor_rag: MotorRAG, pregunta: dict, turno_de_cag: TurnoDeCAG | None = None
) -> ResultadoCaraACara:
    """
    Hace la misma pregunta a ambos y llama a los jueces. Los jueces usan el
    motor CAG porque necesitan ver TODOS los documentos para auditar (y así
    reutilizan su caché). Si ya tenemos la respuesta de CAG (`turno_de_cag`),
    no la repetimos.
    """
    texto_de_la_pregunta = pregunta["pregunta"]
    esperada = pregunta["respuesta_esperada"]
    turno_de_cag = turno_de_cag or responder_con_cag(motor_cag, pregunta)
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
        estrategia=motor_rag.estrategia,
        cag=turno_de_cag.respuesta,
        rag=respuesta_rag,
        exactitud_cag=turno_de_cag.exactitud,
        exactitud_rag=evaluar_exactitud(motor_cag, texto_de_la_pregunta, respuesta_rag.texto, esperada),
        veracidad_cag=turno_de_cag.veracidad,
        veracidad_rag=evaluar_veracidad(motor_cag, texto_de_la_pregunta, respuesta_rag.texto),
        recuperacion_rag=recuperacion,
    )


def correr_comparacion(
    motor_cag: MotorCAG, motor_rag: MotorRAG, preguntas: list[dict]
) -> Iterator[ResultadoCaraACara]:
    """Enfrenta a CAG con la estrategia actual de RAG, pregunta por pregunta."""
    for pregunta in preguntas:
        yield enfrentar(motor_cag, motor_rag, pregunta)


def correr_liga(
    motor_cag: MotorCAG, motor_rag: MotorRAG, preguntas: list[dict], estrategias: list[str] | None = None
) -> Iterator[ResultadoCaraACara]:
    """
    🏆 La liga: en cada pregunta, CAG responde UNA vez y después responde cada
    estrategia de RAG. Entrega un resultado por cada (pregunta, estrategia).
    """
    estrategias = estrategias or list(ESTRATEGIAS)
    estrategia_original = motor_rag.estrategia
    try:
        for pregunta in preguntas:
            turno_de_cag = responder_con_cag(motor_cag, pregunta)
            for estrategia in estrategias:
                motor_rag.estrategia = estrategia
                yield enfrentar(motor_cag, motor_rag, pregunta, turno_de_cag)
    finally:
        motor_rag.estrategia = estrategia_original


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


def resumir_liga(resultados: list[ResultadoCaraACara]) -> dict:
    """
    Resumen de la liga: CAG (una sola vez, porque respondió una vez por pregunta)
    y cada estrategia de RAG, más la exactitud por tipo de pregunta de todos.
    """
    por_estrategia: dict[str, list[ResultadoCaraACara]] = {}
    for resultado in resultados:
        por_estrategia.setdefault(resultado.estrategia, []).append(resultado)
    resumenes = {estrategia: resumir(grupo) for estrategia, grupo in por_estrategia.items()}
    primero = next(iter(resumenes.values()))
    liga = {"cag": primero["cag"], **{estrategia: r["rag"] for estrategia, r in resumenes.items()}}
    liga["por_tipo"] = {
        tipo: {
            "preguntas": datos["preguntas"],
            "cag": datos["exactitud_cag"],
            **{estrategia: resumenes[estrategia]["por_tipo"][tipo]["exactitud_rag"] for estrategia in resumenes},
        }
        for tipo, datos in primero["por_tipo"].items()
    }
    return liga


def exportar(
    corpus: Corpus, motor_cag: MotorCAG, motor_rag: MotorRAG, resultados: list[ResultadoCaraACara]
) -> Path:
    """Guarda TODO (configuración, preparación, resumen y cada respuesta) en un JSON."""
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    estrategias = list(dict.fromkeys(resultado.estrategia for resultado in resultados))
    es_liga = len(estrategias) > 1
    marca_de_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefijo = "liga" if es_liga else f"comparacion_{estrategias[0]}"
    destino = CARPETA_RESULTADOS / f"{prefijo}_{corpus.clave}_{marca_de_tiempo}.json"
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
            "estrategias_de_rag": estrategias,
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
        "resumen": resumir_liga(resultados) if es_liga else resumir(resultados),
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
    parser.add_argument("--estrategia", choices=list(ESTRATEGIAS), default="clasico", help="estrategia de RAG")
    parser.add_argument("--liga", action="store_true", help="enfrenta a CAG con las 4 estrategias de RAG")
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
    estrategias = list(ESTRATEGIAS) if argumentos.liga else [argumentos.estrategia]
    print(
        f"🥊 {corpus.nombre}: {len(preguntas)} preguntas, modelo {argumentos.modelo}, k={argumentos.k}, "
        f"RAG: {', '.join(ESTRATEGIAS[e].nombre for e in estrategias)}"
    )
    inicio = time.perf_counter()
    motor_cag, motor_rag = preparar_contendientes(
        corpus.documentos(), argumentos.modelo, not argumentos.sin_razonar, argumentos.k, estrategia=estrategias[0]
    )

    ICONO = {"correcta": "✅", "parcial": "🟡", "incorrecta": "❌"}
    resultados = []
    print(f"\n{'id':4} {'tipo':20} {'CAG':>4} " + " ".join(f"{e[:10]:>10}" for e in estrategias))
    fila: list[ResultadoCaraACara] = []
    for resultado in correr_liga(motor_cag, motor_rag, preguntas, estrategias):
        resultados.append(resultado)
        fila.append(resultado)
        if len(fila) == len(estrategias):  # ya respondieron todas: imprimimos la fila
            print(
                f"{resultado.pregunta['id']:4} {resultado.pregunta['tipo']:20} "
                f"{ICONO.get(resultado.exactitud_cag.veredicto, '⚠️'):>3} "
                + " ".join(f"{ICONO.get(r.exactitud_rag.veredicto, '⚠️'):>9}" for r in fila)
            )
            fila = []

    resumen = resumir_liga(resultados)
    sistemas = ["cag", *estrategias]
    print("\n📊 Resumen" + " " * 14 + " ".join(f"{s[:10]:>10}" for s in sistemas))
    for nombre, clave in [("Exactitud", "exactitud"), ("Veracidad", "veracidad"), ("Recuperación", "recuperacion")]:
        print(f"   {nombre:20}" + " ".join(f"{_porcentaje(resumen[s].get(clave)):>10}" for s in sistemas))
    for nombre, clave in [("Tiempo total (s)", "segundos_totales"), ("Búsqueda (s)", "segundos_buscando")]:
        print(f"   {nombre:20}" + " ".join(f"{(resumen[s].get(clave) or 0):>10.2f}" for s in sistemas))
    print(f"   {'Tokens leídos':20}" + " ".join(f"{resumen[s]['tokens_leidos']:>10.0f}" for s in sistemas))
    print(f"\n💾 Resultados en {exportar(corpus, motor_cag, motor_rag, resultados)}")
    print(f"⏱️ {time.perf_counter() - inicio:.0f} s en total")
