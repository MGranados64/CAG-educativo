"""
paginas/cara_a_cara.py — CAG vs RAG, cara a cara
================================================

Los dos sistemas responden las mismas preguntas, sobre los mismos documentos,
con el mismo modelo y los mismos jueces. Toda la lógica vive en
comparacion.py; esta página solo la muestra.

  🥊 Pregunta libre  → una pregunta, dos respuestas lado a lado.
  🏁 Banco de pruebas → todas las preguntas de referencia, con resumen y JSON.
"""

from __future__ import annotations

import json

import ollama
import streamlit as st

from cag_core import DURACIONES_DE_CACHE, MODELO_RECOMENDADO, explicar_error, listar_modelos_de_chat
from comparacion import (
    CORPUS,
    correr_comparacion,
    descargar_paper,
    enfrentar,
    exportar,
    preparar_contendientes,
    resumir,
)
from evals import evaluar_veracidad, semaforo
from rag_core import FRAGMENTOS_A_RECUPERAR

ICONO_EXACTITUD = {"correcta": "✅ correcta", "parcial": "🟡 parcial", "incorrecta": "❌ incorrecta"}
ICONO_RECUPERACION = {"completa": "✅ completa", "parcial": "🟡 parcial", "nada": "❌ nada"}

st.session_state.setdefault("rivales", None)  # (motor_cag, motor_rag, clave_del_corpus)
st.session_state.setdefault("pregunta_libre", None)  # el último ResultadoCaraACara de la pregunta libre
st.session_state.setdefault("banco", None)  # {"resultados": [...], "ruta": Path}


@st.cache_data(ttl=30, show_spinner=False)
def modelos_instalados():
    """Pregunta a Ollama qué modelos hay (y lo recuerda 30 s)."""
    return listar_modelos_de_chat(ollama.Client())


def porcentaje(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.0%}"


def segundos(valor: float | None, decimales: int = 2) -> str:
    return "—" if valor is None else f"{valor:.{decimales}f} s"


def mostrar_respuesta(respuesta, exactitud, veracidad, recuperacion=None):
    """Una columna del cara a cara: respuesta, tiempos y veredictos."""
    st.markdown(respuesta.texto)
    columna_a, columna_b = st.columns(2)
    columna_a.metric(
        "Tiempo total", f"{respuesta.segundos_totales:.1f} s", help="Desde que se envía la pregunta hasta la última palabra."
    )
    columna_b.metric(
        "Tokens leídos",
        f"{respuesta.medicion.tokens_del_prompt:,}",
        help="Todo lo que el modelo tuvo que leer en esta pregunta (documentos o fragmentos + instrucciones + pregunta).",
    )
    lectura = f"📖 Lectura del prompt: {respuesta.medicion.segundos_leyendo:.2f} s"
    if respuesta.fragmentos_recuperados is not None:
        lectura += f" · 🔎 búsqueda: {respuesta.segundos_buscando * 1000:.0f} ms"
    st.caption(lectura)
    if exactitud is not None:
        st.markdown(f"**Exactitud:** {ICONO_EXACTITUD.get(exactitud.veredicto, '⚠️ sin veredicto')}")
        st.caption(exactitud.razonamiento or exactitud.error or "")
    if veracidad is not None and not veracidad.error:
        st.markdown(
            f"**Veracidad:** {porcentaje(veracidad.puntaje)} {semaforo(veracidad.puntaje)} "
            f"({veracidad.respaldadas} de {len(veracidad.afirmaciones)} afirmaciones respaldadas)"
        )
    if recuperacion is not None:
        st.markdown(f"**Recuperación del buscador:** {ICONO_RECUPERACION.get(recuperacion.veredicto, '⚠️')}")
        st.caption(recuperacion.razonamiento or recuperacion.error or "")
    if respuesta.fragmentos_recuperados:
        with st.expander(f"🔎 Las {len(respuesta.fragmentos_recuperados)} fichas que trajo el buscador"):
            for recuperado in respuesta.fragmentos_recuperados:
                ficha = recuperado.fragmento
                st.markdown(f"**{ficha.documento} › {ficha.seccion}** · similitud {recuperado.similitud:.3f}")
                st.caption(ficha.texto)


# ════════════════════════════════════════════════════════════════════════════
# 🏷️ ENCABEZADO
# ════════════════════════════════════════════════════════════════════════════
st.title("⚔️ CAG vs RAG: cara a cara")
st.caption(
    "Las mismas preguntas, los mismos documentos, el mismo modelo y los mismos jueces. "
    "La única diferencia: CAG lee todo (desde su caché) y RAG solo lo que encuentra su buscador."
)
with st.expander("⚖️ Las reglas de una pelea justa", expanded=st.session_state.rivales is None):
    st.markdown(
        """
| | 🧠 **CAG** | 🔎 **RAG** |
|---|---|---|
| **Qué lee el modelo** | Todos los documentos, guardados en la caché KV | Solo los *k* fragmentos más parecidos a la pregunta |
| **Preparación** | Leer todo una vez (lectura en frío) | Trocear y calcular un *embedding* por fragmento |
| **Lo que comparten** | Modelo de chat, mesa (`num_ctx`), razonamiento, temperatura 0, reglas y jueces | ← lo mismo |

**Las métricas:** ✅ *Exactitud* (¿coincide con la respuesta correcta?), ⚖️ *Veracidad*
(¿todo lo que dijo está en los documentos?) y, solo para RAG, 🔎 *Recuperación*
(¿el buscador trajo lo necesario?). Esta última separa dos fallos distintos: que
el buscador no encuentre la ficha correcta, o que el modelo la tenga y aun así
responda mal.
"""
    )

try:
    modelos = modelos_instalados()
except Exception as error:
    st.error(explicar_error(error))
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# ⚙️ PREPARACIÓN DE LOS RIVALES
# ════════════════════════════════════════════════════════════════════════════
st.subheader("1. Prepara a los rivales")
columna_corpus, columna_modelo, columna_k = st.columns([1.4, 1, 1])

with columna_corpus:
    clave_corpus = st.selectbox(
        "📚 Documentos de prueba",
        options=list(CORPUS),
        format_func=lambda clave: CORPUS[clave].nombre,
        help="Cada corpus trae sus propias preguntas de referencia con la respuesta correcta.",
    )
    corpus = CORPUS[clave_corpus]
    st.caption(corpus.descripcion)
    if not corpus.disponible and corpus.url_de_descarga:
        st.info(
            "Este corpus se descarga de arXiv (unos cientos de KB). El paper se distribuye bajo "
            "**CC BY-SA 4.0**: se guarda con su atribución y conserva esa licencia."
        )
        if st.button("📥 Descargar el paper desde arXiv", help=corpus.url_de_descarga):
            try:
                with st.spinner("Descargando y convirtiendo a texto…"):
                    descargar_paper(corpus)
                st.rerun()
            except Exception as error:
                st.error(f"No se pudo descargar: {error}")

with columna_modelo:
    nombres = [modelo.nombre for modelo in modelos]
    nombre_modelo = st.selectbox(
        "🤖 Modelo de chat (el mismo para ambos)",
        options=nombres,
        index=nombres.index(MODELO_RECOMENDADO) if MODELO_RECOMENDADO in nombres else 0,
    )
    razonar = st.toggle("🧠 Razonar primero", value=True, help="Se aplica igual a CAG y a RAG.")

with columna_k:
    fragmentos_a_recuperar = st.slider(
        "🔎 Fichas que trae RAG (k)",
        min_value=1,
        max_value=10,
        value=FRAGMENTOS_A_RECUPERAR,
        help=(
            "Más fichas = más probabilidad de traer la correcta, pero más texto que leer y más "
            "'ruido'. Prueba con k=1 y con k=8 para ver la diferencia."
        ),
    )
    duracion_cache = st.radio(
        "Duración de la caché", options=list(DURACIONES_DE_CACHE), format_func=DURACIONES_DE_CACHE.get, horizontal=True
    )

if st.button("🥊 Preparar a los rivales", type="primary", disabled=not corpus.disponible):
    try:
        with st.spinner("CAG lee todos los documentos y RAG prepara su fichero…"):
            motor_cag, motor_rag = preparar_contendientes(
                corpus.documentos(), nombre_modelo, razonar, fragmentos_a_recuperar, duracion_cache
            )
        st.session_state.rivales = (motor_cag, motor_rag, clave_corpus)
        st.session_state.pregunta_libre = None
        st.session_state.banco = None
    except Exception as error:
        st.error(explicar_error(error))

if st.session_state.rivales is None:
    st.info("Elige los documentos y pulsa **Preparar a los rivales** para empezar.")
    st.stop()

motor_cag, motor_rag, clave_preparada = st.session_state.rivales
corpus = CORPUS[clave_preparada]
carga, indice = motor_cag.informe_de_carga, motor_rag.informe_de_indexado
if (clave_preparada, motor_cag.modelo.nombre, motor_cag.razonar, motor_rag.fragmentos_a_recuperar) != (
    clave_corpus,
    nombre_modelo,
    razonar,
    fragmentos_a_recuperar,
):
    st.info("Cambiaste la configuración: pulsa **Preparar a los rivales** para aplicarla.")

columna_cag, columna_rag = st.columns(2)
with columna_cag:
    st.markdown("#### 🧠 CAG")
    a, b = st.columns(2)
    a.metric("Tokens memorizados", f"{carga.tokens_en_memoria:,}")
    b.metric(
        "Preparación",
        f"{carga.lectura_en_frio.segundos_leyendo:.2f} s",
        help="La lectura en frío de todos los documentos (sin contar la carga del modelo).",
    )
with columna_rag:
    st.markdown("#### 🔎 RAG")
    a, b = st.columns(2)
    a.metric("Fichas en el índice", f"{indice.numero_de_fragmentos}", help=f"De ~{indice.caracteres_promedio} caracteres cada una.")
    b.metric(
        "Preparación",
        f"{indice.segundos_indexando:.2f} s",
        help=f"Trocear y calcular {indice.numero_de_fragmentos} embeddings de {indice.dimensiones} números con {indice.modelo_de_embeddings}.",
    )
st.caption(
    f"Ambos usan `{motor_cag.modelo.nombre}` con la misma mesa (`num_ctx` = {motor_cag.num_ctx:,}). "
    f"RAG trae {motor_rag.fragmentos_a_recuperar} fichas por pregunta."
)

st.divider()
st.subheader("2. Pon a prueba a los rivales")
pestana_libre, pestana_banco = st.tabs(["🥊 Pregunta libre", "🏁 Banco de pruebas"])

# ════════════════════════════════════════════════════════════════════════════
# 🥊 PREGUNTA LIBRE
# ════════════════════════════════════════════════════════════════════════════
with pestana_libre:
    preguntas_de_referencia = corpus.preguntas()["preguntas"]
    opciones = ["(escribe tu propia pregunta)"] + [p["pregunta"] for p in preguntas_de_referencia]
    elegida = st.selectbox(
        "Elige una pregunta de referencia…",
        options=opciones,
        help="Con una pregunta de referencia también verás la exactitud y la recuperación.",
    )
    pregunta_propia = st.text_input("…o escribe la tuya", disabled=elegida != opciones[0])
    if st.button("Preguntar a ambos", type="primary"):
        if elegida != opciones[0]:
            pregunta = next(p for p in preguntas_de_referencia if p["pregunta"] == elegida)
        elif pregunta_propia.strip():
            # Sin respuesta correcta conocida: solo podemos medir la veracidad.
            pregunta = {"id": "libre", "tipo": "libre", "pregunta": pregunta_propia.strip(), "respuesta_esperada": None}
        else:
            pregunta = None
            st.warning("Escribe una pregunta o elige una de referencia.")
        if pregunta:
            try:
                with st.spinner("Respondiendo y auditando con los dos sistemas…"):
                    if pregunta["respuesta_esperada"]:
                        st.session_state.pregunta_libre = enfrentar(motor_cag, motor_rag, pregunta)
                    else:
                        respuesta_cag = motor_cag.preguntar(pregunta["pregunta"])
                        respuesta_rag = motor_rag.preguntar(pregunta["pregunta"])
                        st.session_state.pregunta_libre = {
                            "pregunta": pregunta,
                            "cag": respuesta_cag,
                            "rag": respuesta_rag,
                            "veracidad_cag": evaluar_veracidad(motor_cag, pregunta["pregunta"], respuesta_cag.texto),
                            "veracidad_rag": evaluar_veracidad(motor_cag, pregunta["pregunta"], respuesta_rag.texto),
                        }
            except Exception as error:
                st.error(explicar_error(error))

    resultado = st.session_state.pregunta_libre
    if resultado is not None:
        # Puede ser un ResultadoCaraACara (pregunta de referencia) o un dict (pregunta propia).
        datos = resultado if isinstance(resultado, dict) else vars(resultado)
        st.markdown(f"**❓ {datos['pregunta']['pregunta']}**")
        if datos["pregunta"].get("respuesta_esperada"):
            st.caption(f"Respuesta correcta: {datos['pregunta']['respuesta_esperada']}")
        columna_cag, columna_rag = st.columns(2, border=True)
        with columna_cag:
            st.markdown("#### 🧠 CAG")
            mostrar_respuesta(datos["cag"], datos.get("exactitud_cag"), datos["veracidad_cag"])
        with columna_rag:
            st.markdown("#### 🔎 RAG")
            mostrar_respuesta(datos["rag"], datos.get("exactitud_rag"), datos["veracidad_rag"], datos.get("recuperacion_rag"))

# ════════════════════════════════════════════════════════════════════════════
# 🏁 BANCO DE PRUEBAS
# ════════════════════════════════════════════════════════════════════════════
with pestana_banco:
    preguntas = corpus.preguntas()["preguntas"]
    st.markdown(
        f"Corre las **{len(preguntas)} preguntas de referencia** de *{corpus.nombre}* con ambos "
        "sistemas y califica cada respuesta con tres jueces. Con gpt-oss:20b en una GPU de 16 GB "
        "tarda unos 20 segundos por pregunta."
    )
    if st.button("🏁 Correr el banco de pruebas", type="primary"):
        resultados = []
        barra = st.progress(0.0, text="Empezando…")
        try:
            for numero, resultado in enumerate(correr_comparacion(motor_cag, motor_rag, preguntas), start=1):
                resultados.append(resultado)
                barra.progress(numero / len(preguntas), text=f"Pregunta {numero} de {len(preguntas)} lista")
            ruta = exportar(corpus, motor_cag, motor_rag, resultados)
            st.session_state.banco = {"resultados": resultados, "ruta": ruta}
        except Exception as error:
            st.error(explicar_error(error))
        barra.empty()

    banco = st.session_state.banco
    if banco is not None:
        resultados = banco["resultados"]
        resumen = resumir(resultados)

        st.markdown("#### 📊 Resumen")
        columna_a, columna_b, columna_c = st.columns(3)
        columna_a.metric(
            "Exactitud CAG / RAG",
            f"{porcentaje(resumen['cag']['exactitud'])} / {porcentaje(resumen['rag']['exactitud'])}",
            help="Correcta = 100 %, parcial = 50 %, incorrecta = 0 %, promediado en todas las preguntas.",
        )
        columna_b.metric(
            "Veracidad CAG / RAG",
            f"{porcentaje(resumen['cag']['veracidad'])} / {porcentaje(resumen['rag']['veracidad'])}",
            help="Qué parte de lo que dijo cada uno está respaldado por los documentos.",
        )
        columna_c.metric(
            "Recuperación de RAG",
            porcentaje(resumen["rag"]["recuperacion"]),
            help="Qué tan seguido el buscador trajo la información necesaria (solo preguntas con respuesta en los documentos).",
        )
        st.markdown(
            f"""
| Promedio por pregunta | 🧠 CAG | 🔎 RAG |
|---|---|---|
| Tiempo total | {segundos(resumen['cag']['segundos_totales'])} | {segundos(resumen['rag']['segundos_totales'])} |
| Primera palabra | {segundos(resumen['cag']['segundos_hasta_primer_token'])} | {segundos(resumen['rag']['segundos_hasta_primer_token'])} |
| Lectura del prompt | {segundos(resumen['cag']['segundos_leyendo'], 3)} | {segundos(resumen['rag']['segundos_leyendo'], 3)} |
| Tokens leídos | {resumen['cag']['tokens_leidos']:,.0f} | {resumen['rag']['tokens_leidos']:,.0f} |
| Búsqueda | — | {resumen['rag']['segundos_buscando'] * 1000:.0f} ms |
"""
        )

        st.markdown("#### 🧩 Exactitud por tipo de pregunta")
        st.dataframe(
            [
                {
                    "Tipo": tipo,
                    "Preguntas": datos["preguntas"],
                    "CAG": porcentaje(datos["exactitud_cag"]),
                    "RAG": porcentaje(datos["exactitud_rag"]),
                }
                for tipo, datos in resumen["por_tipo"].items()
            ],
            hide_index=True,
            width="stretch",
        )

        st.markdown("#### 🔍 Pregunta por pregunta")
        st.dataframe(
            [
                {
                    "ID": r.pregunta["id"],
                    "Tipo": r.pregunta["tipo"],
                    "Pregunta": r.pregunta["pregunta"],
                    "CAG": ICONO_EXACTITUD.get(r.exactitud_cag.veredicto, "⚠️"),
                    "RAG": ICONO_EXACTITUD.get(r.exactitud_rag.veredicto, "⚠️"),
                    "Buscador RAG": ICONO_RECUPERACION.get(r.recuperacion_rag.veredicto, "⚠️") if r.recuperacion_rag else "—",
                    "Tiempo CAG (s)": round(r.cag.segundos_totales, 1),
                    "Tiempo RAG (s)": round(r.rag.segundos_totales, 1),
                }
                for r in resultados
            ],
            hide_index=True,
            width="stretch",
        )
        for r in resultados:
            with st.expander(f"{r.pregunta['id']} · {r.pregunta['pregunta']}"):
                st.caption(f"Tipo: {r.pregunta['tipo']} · Respuesta correcta: {r.pregunta['respuesta_esperada']}")
                columna_cag, columna_rag = st.columns(2, border=True)
                with columna_cag:
                    st.markdown("**🧠 CAG**")
                    mostrar_respuesta(r.cag, r.exactitud_cag, r.veracidad_cag)
                with columna_rag:
                    st.markdown("**🔎 RAG**")
                    mostrar_respuesta(r.rag, r.exactitud_rag, r.veracidad_rag, r.recuperacion_rag)

        st.download_button(
            "💾 Descargar los resultados (JSON)",
            data=banco["ruta"].read_text("utf-8"),
            file_name=banco["ruta"].name,
            mime="application/json",
            help="Configuración, preparación, resumen y cada respuesta con sus veredictos.",
        )
        st.caption(f"También se guardó en `{banco['ruta'].relative_to(banco['ruta'].parent.parent)}`.")

        with st.expander("📖 ¿Cómo leer estos resultados?"):
            st.markdown(
                """
- **Tiempo:** RAG lee pocos tokens en cada pregunta; CAG lee todo, pero desde la caché
  (por eso su lectura también es rápida). La diferencia de tiempo suele estar en la
  **escritura** (cuánto razona y escribe cada uno), no en la lectura.
- **Exactitud por tipo:** RAG suele fallar en preguntas que necesitan **varios
  documentos** o **juntar datos dispersos**, porque su buscador trae solo *k* fichas.
  Si la columna *Buscador RAG* dice ❌, el fallo fue del buscador, no del modelo.
- **Preguntas trampa** (*fuera_del_documento*): ambos deberían decir que no saben.
- **Ojo:** son pocas preguntas y los jueces son IA. Úsalo para entender, no como una
  verdad estadística: cambia *k*, el modelo o el corpus y vuelve a correrlo.
"""
            )
