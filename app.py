"""
app.py — La interfaz educativa de CAG (hecha con Streamlit)
============================================================

Streamlit convierte un script de Python en una página web. Cada vez que haces
clic en algo, el script se vuelve a ejecutar de arriba a abajo; lo que debe
"sobrevivir" entre clics se guarda en `st.session_state` (la memoria de la
página).

La pantalla tiene tres paneles:
  ① Inyección a la caché  → le das al modelo el "libro" que debe memorizar.
  ② Chat                  → le haces preguntas sobre ese libro.
  ③ Transparencia/EVALS   → ves cuánto se reutilizó de la caché y qué
                            calificación le pusieron los jueces a la respuesta.

Ejecuta:   streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from cag_core import (
    DURACIONES_DE_CACHE,
    MINIMO_TOKENS_PARA_CACHE,
    MODELO,
    ConocimientoDemasiadoGrande,
    Documento,
    MotorCAG,
    explicar_error,
    extraer_texto,
)
from evals import evaluar_respuesta, semaforo

RUTA_DOCUMENTO_EJEMPLO = Path(__file__).parent / "datos" / "manual_cooperativa_nube_alta.md"

PREGUNTAS_SUGERIDAS = [
    "¿Cuántos días de vacaciones tiene un empleado en su tercer año?",
    "¿Cuánto tiempo fermenta el café y hasta qué humedad se seca?",
    "¿Quién fundó la cooperativa y con cuántas familias?",
    # 👇 Esta NO está en el documento: sirve para ver si el modelo inventa.
    "¿Cuál es la capital de Australia?",
]

st.set_page_config(page_title="CAG Educativo", page_icon="🧠", layout="wide")

# ── Memoria de la página (sobrevive entre clics) ────────────────────────────
st.session_state.setdefault("motor", None)  # el MotorCAG con el conocimiento cargado
st.session_state.setdefault("turnos", [])  # cada pregunta con su respuesta y su evaluación
st.session_state.setdefault("pendiente_de_auditar", False)


def historial_para_la_api(turnos: list[dict]) -> list[dict]:
    """
    Convierte los turnos guardados al formato que espera la API: una lista que
    alterna usuario/asistente. Omitimos las respuestas rechazadas por los
    filtros de seguridad para que la conversación siga siendo coherente.
    """
    mensajes = []
    for turno in turnos:
        respuesta = turno["respuesta"]
        if respuesta.fue_rechazada:
            continue
        mensajes.append({"role": "user", "content": respuesta.pregunta})
        mensajes.append({"role": "assistant", "content": respuesta.texto})
    return mensajes


def formato_usd(cantidad: float | None) -> str:
    return "—" if cantidad is None else f"${cantidad:.4f}"


# ════════════════════════════════════════════════════════════════════════════
# 🧭 BARRA LATERAL: configuración y explicación
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input(
        "API key de Anthropic",
        type="password",
        help=(
            "Tu llave para usar el modelo. Solo vive en esta sesión del navegador; "
            "no se guarda en disco. Si la dejas vacía, se usa la variable de "
            "entorno ANTHROPIC_API_KEY."
        ),
    )
    if not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        st.caption("⚠️ No hay API key. Escríbela arriba o define ANTHROPIC_API_KEY.")

    permitir_conocimiento_externo = st.toggle(
        "🧪 Permitir conocimiento externo",
        value=False,
        help=(
            "APAGADO (recomendado): el modelo solo puede usar tus documentos. "
            "ENCENDIDO: puede completar con lo que 'sabe' de memoria. Enciéndelo y "
            "pregunta algo que no esté en el documento: verás cómo el juez de "
            "veracidad marca esas afirmaciones como no respaldadas."
        ),
    )
    auditar_automaticamente = st.toggle(
        "⚖️ Auditar cada respuesta automáticamente",
        value=True,
        help=(
            "Después de cada respuesta, dos 'jueces' (el mismo modelo con otras "
            "instrucciones) la califican. Cuesta una llamada extra por juez; "
            "apágalo si quieres auditar solo algunas respuestas."
        ),
    )
    if st.button(
        "🧹 Borrar conversación",
        help="Empieza una conversación nueva. El conocimiento sigue en la caché.",
        width="stretch",
    ):
        st.session_state.turnos = []
        st.session_state.pendiente_de_auditar = False

    st.divider()
    with st.expander("🎓 ¿Qué es CAG? (en 1 minuto)"):
        st.markdown(
            """
**RAG** (*Retrieval-Augmented Generation*) es mandar al estudiante a la
biblioteca **en cada pregunta**: busca fragmentos, los lee y responde.

**CAG** (*Cache-Augmented Generation*) es darle **el libro completo antes
del examen** para que lo memorice. Luego responde sin buscar nada.

La **caché** es la "memoria fotográfica": el servidor guarda el trabajo de
haber leído el libro, así las siguientes preguntas **no lo vuelven a leer**
y salen más rápidas y hasta 20 veces más baratas en esa parte.

**Límite de CAG:** todo el libro debe caber en la memoria del modelo
(su *ventana de contexto*).
"""
        )
    st.caption(f"Modelo: `{MODELO}`")

# ════════════════════════════════════════════════════════════════════════════
# 🏷️ ENCABEZADO
# ════════════════════════════════════════════════════════════════════════════
st.title("🧠 CAG Educativo")
st.caption(
    "Cache-Augmented Generation: mira cómo un modelo de IA *memoriza* tus documentos "
    "y cómo auditamos si lo que responde es verdad."
)
with st.expander("👣 Cómo usar esta app en 3 pasos", expanded=st.session_state.motor is None):
    st.markdown(
        """
1. **Panel ①** — Elige documentos (o usa el de ejemplo) y pulsa **Inyectar**.
   El modelo los "memoriza" en la caché.
2. **Panel ②** — Haz preguntas. La respuesta aparece palabra por palabra.
3. **Panel ③** — Mira cuántos tokens salieron de la caché, cuánto costó y qué
   calificaciones de **Veracidad** y **Relevancia** le dieron los jueces, y por qué.
"""
    )

panel_carga, panel_chat, panel_transparencia = st.columns([1, 1.3, 1.3], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# ① PANEL DE CARGA: inyección a la caché
# ════════════════════════════════════════════════════════════════════════════
with panel_carga:
    st.subheader("① Inyección a la caché")
    st.caption("Aquí le entregas al modelo el \"libro\" que debe memorizar antes del examen.")

    archivos_subidos = st.file_uploader(
        "Sube tus documentos",
        type=["txt", "md", "pdf", "csv", "json"],
        accept_multiple_files=True,
        help=(
            "Todo el texto de estos archivos se enviará completo al modelo. "
            "A diferencia de RAG, no se corta en pedazos ni se busca: se memoriza entero."
        ),
    )
    usar_documento_ejemplo = st.checkbox(
        "Incluir documento de ejemplo",
        value=True,
        help=(
            "Un manual FICTICIO de una cooperativa cafetalera. Como todo es inventado, "
            "es fácil notar cuándo el modelo responde con algo que no está escrito ahí."
        ),
    )
    texto_pegado = st.text_area(
        "…o pega texto directamente",
        height=90,
        placeholder="Pega aquí apuntes, un artículo, reglas de tu empresa…",
        help="Se agrega como un documento más llamado 'texto_pegado.txt'.",
    )
    duracion_cache = st.radio(
        "Duración de la caché",
        options=list(DURACIONES_DE_CACHE),
        format_func=DURACIONES_DE_CACHE.get,
        horizontal=True,
        help=(
            "Cuánto tiempo recuerda el servidor el conocimiento desde su último uso. "
            "5 minutos: guardar cuesta 1.25× el precio normal. 1 hora: cuesta 2×, pero "
            "conviene si haces pausas largas entre preguntas. Cada pregunta renueva el reloj."
        ),
    )

    if st.button(
        "💉 Inyectar en la caché",
        type="primary",
        width="stretch",
        help=(
            "1) Junta los documentos en un solo bloque. 2) Cuenta sus tokens para "
            "comprobar que caben. 3) 'Precalienta' la caché: el servidor lee y guarda "
            "el conocimiento sin generar respuesta, para que tu primera pregunta ya lo "
            "encuentre memorizado. Inyectar de nuevo reinicia la conversación."
        ),
    ):
        documentos = [
            Documento(archivo.name, extraer_texto(archivo.name, archivo.getvalue()))
            for archivo in archivos_subidos or []
        ]
        if usar_documento_ejemplo:
            documentos.append(
                Documento(RUTA_DOCUMENTO_EJEMPLO.name, RUTA_DOCUMENTO_EJEMPLO.read_text("utf-8"))
            )
        if texto_pegado.strip():
            documentos.append(Documento("texto_pegado.txt", texto_pegado))

        if not documentos:
            st.warning("Primero sube un archivo, pega texto o marca el documento de ejemplo.")
        else:
            try:
                with st.spinner("Midiendo y memorizando el conocimiento…"):
                    motor_nuevo = MotorCAG(api_key=api_key or None)
                    motor_nuevo.cargar_conocimiento(documentos, duracion_cache)
                st.session_state.motor = motor_nuevo
                st.session_state.turnos = []  # conocimiento nuevo = conversación nueva
                st.session_state.pendiente_de_auditar = False
            except ConocimientoDemasiadoGrande as error:
                st.error(str(error))
            except Exception as error:  # mostramos cualquier fallo en lenguaje humano
                st.error(explicar_error(error))

    motor: MotorCAG | None = st.session_state.motor
    if motor is not None:
        informe = motor.informe_de_carga
        st.success(f"✅ Conocimiento en caché: {len(informe.documentos)} documento(s)")

        columna_a, columna_b = st.columns(2)
        columna_a.metric(
            "Tokens",
            f"{informe.tokens_de_conocimiento:,}",
            help=(
                "Tokens memorizados. Un token es un pedacito de palabra (≈ 4 caracteres): "
                "es la unidad con la que el modelo lee y con la que se cobra."
            ),
        )
        columna_b.metric(
            "Caracteres",
            f"{informe.caracteres:,}",
            help="Longitud del bloque de conocimiento, incluyendo las etiquetas que separan documentos.",
        )
        st.progress(
            min(informe.fraccion_de_ventana, 1.0),
            text=f"Ocupa {informe.fraccion_de_ventana:.2%} de la memoria de trabajo del modelo",
        )
        st.caption(
            "La *ventana de contexto* es cuánto texto puede tener el modelo 'en la mesa' a la vez. "
            "Si tus documentos no caben, CAG no sirve y se usa RAG."
        )

        if informe.precalentada and informe.recibo_precalentamiento:
            recibo = informe.recibo_precalentamiento
            st.metric(
                "Escritos en caché al precalentar",
                f"{recibo.escritos_en_cache:,} tokens",
                help=(
                    "Tokens que el servidor acaba de guardar. Esta es la 'inversión' inicial: "
                    "cuesta un poco más que una lectura normal, pero cada pregunta siguiente "
                    "los leerá a una fracción del precio."
                ),
            )
            st.caption(f"Costo del precalentamiento: {formato_usd(recibo.costo_usd(MODELO))}")
        if informe.aviso:
            st.warning(informe.aviso)
        if not informe.es_cacheable:
            st.info(
                f"💡 Consejo: con más de {MINIMO_TOKENS_PARA_CACHE} tokens de conocimiento "
                "verás la caché en acción."
            )

        with st.expander("👀 Ver exactamente lo que memorizó el modelo"):
            limite_de_vista_previa = 6_000
            st.code(
                motor.conocimiento_en_cache[:limite_de_vista_previa],
                language="markdown",
                wrap_lines=True,
                height=300,
            )
            if len(motor.conocimiento_en_cache) > limite_de_vista_previa:
                st.caption(f"(Vista previa: primeros {limite_de_vista_previa:,} caracteres.)")

# ════════════════════════════════════════════════════════════════════════════
# ② PANEL DE CHAT
# ════════════════════════════════════════════════════════════════════════════
with panel_chat:
    st.subheader("② Chat con el conocimiento")
    motor = st.session_state.motor

    if motor is None:
        st.info("👈 Primero inyecta conocimiento en el panel ①. Después podrás preguntar aquí.")
    else:
        if permitir_conocimiento_externo:
            st.caption("🧪 Modo libre: el modelo puede usar conocimiento externo (puede alucinar).")
        else:
            st.caption("🔒 Modo estricto: el modelo solo puede usar tus documentos.")

        caja_de_conversacion = st.container(height=480, border=True, autoscroll=True)
        with caja_de_conversacion:
            aviso_conversacion_vacia = st.empty()
            if not st.session_state.turnos:
                aviso_conversacion_vacia.caption("Aquí aparecerá la conversación.")
            for numero, turno in enumerate(st.session_state.turnos, start=1):
                respuesta = turno["respuesta"]
                with st.chat_message("user"):
                    st.markdown(respuesta.pregunta)
                with st.chat_message("assistant"):
                    if respuesta.fue_rechazada:
                        st.warning("Los filtros de seguridad del modelo rechazaron esta pregunta.")
                    else:
                        st.markdown(respuesta.texto)
                    st.caption(
                        f"#{numero} · ⚡ {respuesta.recibo.fraccion_desde_cache:.0%} leído de caché "
                        f"· ⏱️ {respuesta.segundos_totales:.1f} s"
                    )

        pregunta_sugerida = None
        if RUTA_DOCUMENTO_EJEMPLO.name in motor.documentos_cargados:
            with st.expander("💡 Preguntas de ejemplo"):
                for sugerencia in PREGUNTAS_SUGERIDAS:
                    if st.button(sugerencia, key=f"sugerencia-{sugerencia}", width="stretch"):
                        pregunta_sugerida = sugerencia

        pregunta = st.chat_input("Escribe tu pregunta sobre los documentos…") or pregunta_sugerida

        if pregunta:
            aviso_conversacion_vacia.empty()
            with caja_de_conversacion:
                with st.chat_message("user"):
                    st.markdown(pregunta)
                with st.chat_message("assistant"):
                    try:
                        respuesta_en_vivo = motor.preguntar_en_vivo(
                            pregunta,
                            historial=historial_para_la_api(st.session_state.turnos),
                            permitir_conocimiento_externo=permitir_conocimiento_externo,
                        )
                        # write_stream va pintando el texto a medida que llega.
                        st.write_stream(respuesta_en_vivo)
                        respuesta_nueva = respuesta_en_vivo.resultado
                    except Exception as error:
                        st.error(explicar_error(error))
                        respuesta_nueva = None

            if respuesta_nueva is not None:
                st.session_state.turnos.append(
                    {
                        "respuesta": respuesta_nueva,
                        "evaluacion": None,
                        "modo_libre": permitir_conocimiento_externo,
                    }
                )
                st.session_state.pendiente_de_auditar = (
                    auditar_automaticamente and not respuesta_nueva.fue_rechazada
                )
                if respuesta_nueva.fue_rechazada:
                    st.warning("Los filtros de seguridad del modelo rechazaron esta pregunta.")

# ════════════════════════════════════════════════════════════════════════════
# ③ PANEL DE TRANSPARENCIA Y EVALS
# ════════════════════════════════════════════════════════════════════════════
with panel_transparencia:
    st.subheader("③ Transparencia y EVALS")
    turnos = st.session_state.turnos
    motor = st.session_state.motor

    if not turnos:
        st.info(
            "Cuando el modelo responda, aquí verás:\n\n"
            "- ⚡ **La caché en acción**: cuántos tokens se reutilizaron y cuánto ahorraste.\n"
            "- ⚖️ **Veracidad**: ¿todo lo que dijo está en el documento?\n"
            "- 🎯 **Relevancia**: ¿contestó lo que preguntaste?"
        )
    else:
        indice = st.selectbox(
            "Respuesta a inspeccionar",
            options=range(len(turnos)),
            index=len(turnos) - 1,
            format_func=lambda i: f"#{i + 1}: {turnos[i]['respuesta'].pregunta[:60]}",
            help="Elige cualquier respuesta anterior para ver su auditoría.",
        )
        turno = turnos[indice]
        respuesta = turno["respuesta"]
        recibo = respuesta.recibo

        # ── ⚡ La caché en acción ────────────────────────────────────────────
        st.markdown("#### ⚡ La caché en acción")
        st.progress(
            recibo.fraccion_desde_cache,
            text=f"{recibo.fraccion_desde_cache:.0%} de la lectura salió de la caché",
        )
        columna_a, columna_b = st.columns(2)
        columna_a.metric(
            "Leídos de caché",
            f"{recibo.leidos_de_cache:,}",
            help=(
                "Tokens que el modelo ya tenía memorizados: no los volvió a leer. "
                "Cuestan 20 veces menos que una lectura normal."
            ),
        )
        columna_b.metric(
            "Escritos en caché",
            f"{recibo.escritos_en_cache:,}",
            help=(
                "Tokens que se memorizaron en esta llamada. Si es la primera pregunta y "
                "no hubo precalentamiento, o si la caché expiró, verás el documento aquí."
            ),
        )
        columna_a.metric(
            "Sin caché",
            f"{recibo.sin_cache:,}",
            help="Tokens leídos 'en frío': instrucciones, historial y tu pregunta.",
        )
        columna_b.metric(
            "Generados",
            f"{recibo.de_salida:,}",
            help="Tokens de salida: el razonamiento interno del modelo más la respuesta visible.",
        )
        costo = recibo.costo_usd(respuesta.modelo_que_respondio)
        costo_sin_cache = recibo.costo_sin_cache_usd(respuesta.modelo_que_respondio)
        columna_a.metric(
            "Primer token",
            "—"
            if respuesta.segundos_hasta_primer_token is None
            else f"{respuesta.segundos_hasta_primer_token:.1f} s",
            help=(
                "Cuánto tardó en aparecer la primera palabra. Incluye el tiempo que el "
                "modelo pasa razonando antes de escribir."
            ),
        )
        columna_b.metric(
            "Costo",
            formato_usd(costo),
            help=(
                "Costo estimado de esta respuesta en dólares. La mayor parte suele ser la "
                "salida (el modelo pensando y escribiendo), que la caché no abarata."
            ),
        )
        if costo is not None and costo_sin_cache:
            ahorro = 1 - costo / costo_sin_cache
            if ahorro >= 0:
                st.caption(
                    f"💰 Sin caché habría costado {formato_usd(costo_sin_cache)}: "
                    f"la caché ahorró un **{ahorro:.0%}** en esta respuesta."
                )
            else:
                st.caption(
                    f"💰 Sin caché habría costado {formato_usd(costo_sin_cache)}. Esta vez "
                    "costó un poco más porque se *escribió* la caché: es la inversión que "
                    "abarata las preguntas siguientes."
                )
        if respuesta.modelo_que_respondio != MODELO:
            st.caption(
                f"🛟 Respondió el modelo de respaldo `{respuesta.modelo_que_respondio}` "
                "porque el principal rechazó la petición."
            )
        if turno["modo_libre"]:
            st.caption("🧪 Esta respuesta se generó en modo libre (con conocimiento externo).")

        if respuesta.razonamiento_resumido:
            with st.expander("🧠 Resumen del razonamiento del modelo"):
                st.caption(
                    "Antes de responder, el modelo razona internamente. Esto es un resumen "
                    "de ese razonamiento."
                )
                st.markdown(respuesta.razonamiento_resumido)

        st.divider()

        # ── ⚖️🎯 EVALS ────────────────────────────────────────────────────────
        if respuesta.fue_rechazada:
            st.info("No hay nada que auditar: la pregunta fue rechazada.")
        else:
            debe_auditarse_ahora = (
                turno["evaluacion"] is None
                and st.session_state.pendiente_de_auditar
                and indice == len(turnos) - 1
            )
            if turno["evaluacion"] is None and not debe_auditarse_ahora:
                debe_auditarse_ahora = st.button(
                    "⚖️ Auditar esta respuesta",
                    type="primary",
                    width="stretch",
                    help="Llama a los dos jueces (Veracidad y Relevancia) para calificar esta respuesta.",
                )

            if debe_auditarse_ahora:
                try:
                    with st.spinner("Los jueces están revisando la respuesta…"):
                        turno["evaluacion"] = evaluar_respuesta(motor, respuesta.pregunta, respuesta.texto)
                except Exception as error:
                    st.error(explicar_error(error))
                st.session_state.pendiente_de_auditar = False

            evaluacion = turno["evaluacion"]
            if evaluacion is not None:
                # ── ⚖️ Veracidad ──────────────────────────────────────────────
                veracidad = evaluacion.veracidad
                st.markdown("#### ⚖️ Veracidad *(Faithfulness)*")
                if veracidad.error:
                    st.warning(f"El juez no pudo evaluar: {veracidad.error}")
                else:
                    columna_a, columna_b = st.columns(2)
                    columna_a.metric(
                        "Puntaje",
                        f"{veracidad.puntaje:.0%}",
                        help=(
                            "Afirmaciones respaldadas ÷ afirmaciones totales. 100 % = nada "
                            "inventado. Menos de 100 % = hay datos que no salen del documento "
                            "(posible alucinación)."
                        ),
                    )
                    columna_b.metric(
                        "Respaldadas",
                        f"{veracidad.respaldadas} de {len(veracidad.afirmaciones)}",
                        help=(
                            "Afirmaciones respaldadas por el documento. El juez dividió la "
                            "respuesta en frases con un solo hecho cada una."
                        ),
                    )
                    st.progress(veracidad.puntaje, text=semaforo(veracidad.puntaje))
                    st.markdown(f"**Veredicto del juez:** {veracidad.resumen}")
                    for afirmacion in veracidad.afirmaciones:
                        icono = "✅" if afirmacion["respaldada"] else "❌"
                        with st.expander(f"{icono} {afirmacion['afirmacion']}"):
                            if afirmacion["evidencia"]:
                                st.markdown(f"📄 **Evidencia en el documento:**\n\n> {afirmacion['evidencia']}")
                            else:
                                st.markdown("📄 **No se encontró evidencia en el documento.**")
                            st.caption(f"Razonamiento del juez: {afirmacion['razonamiento']}")

                # ── 🎯 Relevancia ─────────────────────────────────────────────
                relevancia = evaluacion.relevancia
                st.markdown("#### 🎯 Relevancia *(Answer Relevancy)*")
                if relevancia.error:
                    st.warning(f"El juez no pudo evaluar: {relevancia.error}")
                else:
                    columna_a, columna_b = st.columns(2)
                    columna_a.metric(
                        "Puntaje",
                        f"{relevancia.puntaje:.0%}",
                        help="La rúbrica de 1 a 5 convertida a porcentaje: 1→0 %, 3→50 %, 5→100 %.",
                    )
                    columna_b.metric(
                        "Rúbrica",
                        f"{relevancia.puntuacion_1_a_5} de 5",
                        help=(
                            "5: directa y completa · 4: con relleno o falta un detalle · "
                            "3: parcial o 'no tengo esa información' · 2: toca el tema sin "
                            "contestar · 1: no responde."
                        ),
                    )
                    st.progress(relevancia.puntaje, text=semaforo(relevancia.puntaje))
                    st.markdown(f"**Veredicto del juez:** {relevancia.razonamiento}")
                    st.markdown(
                        "**Preguntas inversas** — lo que la respuesta parece estar contestando:",
                        help=(
                            "Técnica de RAGas: si estas preguntas se parecen a la tuya, la "
                            "respuesta es relevante."
                        ),
                    )
                    for pregunta_inversa in relevancia.preguntas_inversas:
                        st.markdown(f"- {pregunta_inversa}")
                    if relevancia.partes_irrelevantes:
                        st.markdown("**Partes que sobran:**")
                        for parte in relevancia.partes_irrelevantes:
                            st.markdown(f"- {parte}")

                # ── Lo que costó auditar ─────────────────────────────────────
                recibos_de_jueces = [
                    resultado.recibo
                    for resultado in (veracidad, relevancia)
                    if resultado.recibo is not None
                ]
                costos_de_jueces = [r.costo_usd(MODELO) for r in recibos_de_jueces]
                costo_auditoria = None if None in costos_de_jueces else sum(costos_de_jueces)
                leidos_por_juez_veracidad = veracidad.recibo.leidos_de_cache if veracidad.recibo else 0
                st.caption(
                    f"🧾 Auditoría: {evaluacion.segundos:.1f} s · {formato_usd(costo_auditoria)} · "
                    f"el juez de veracidad reutilizó {leidos_por_juez_veracidad:,} tokens de la caché."
                )

        with st.expander("📐 ¿Cómo se calculan estas métricas?"):
            st.markdown(
                """
**Veracidad (Faithfulness)** — inspirada en RAGas:
1. Un juez divide la respuesta en *afirmaciones atómicas* (un hecho por frase).
2. Para cada una busca la **cita textual** del documento que la respalda.
3. `puntaje = afirmaciones respaldadas ÷ afirmaciones totales` (lo calcula Python, no el modelo).

**Relevancia (Answer Relevancy)** — adaptada de RAGas:
1. Otro juez, que **no ve el documento**, lee la respuesta y escribe 3
   *preguntas inversas*: ¿qué preguntas estaría contestando esta respuesta?
2. Las compara con tu pregunta y califica con una rúbrica de 1 a 5.
3. `puntaje = (rúbrica − 1) ÷ 4`

**¿Por qué dos jueces separados?** Un juez con una sola tarea se equivoca
menos. Además, una respuesta puede ser verdadera pero irrelevante
("la tienda cierra los martes" cuando preguntaste por vacaciones), o
relevante pero inventada.

**Los jueces también son IA** y pueden equivocarse. Por eso muestran su
evidencia y su razonamiento: tú tienes la última palabra.
"""
            )
