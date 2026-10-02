"""
paginas/laboratorio.py — El laboratorio de CAG (tres paneles)
==============================================================

Streamlit vuelve a ejecutar este script de arriba a abajo cada vez que haces
clic en algo; lo que debe "sobrevivir" entre clics se guarda en
`st.session_state` (la memoria de la página).

  ① Inyección a la caché  → eliges un modelo local y le das el "libro".
  ② Chat                  → le haces preguntas sobre ese libro.
  ③ Transparencia/EVALS   → ves si recordó el libro (caché) y qué calificación
                            le pusieron los jueces a la respuesta.
"""

from __future__ import annotations

import ollama
import streamlit as st

from cag_core import (
    CARPETA_CORPUS_DE_EJEMPLO,
    DURACIONES_DE_CACHE,
    MODELO_RECOMENDADO,
    TOKENS_DE_UN_LIBRO_GRANDE,
    ConocimientoDemasiadoGrande,
    Documento,
    MotorCAG,
    explicar_error,
    cargar_documentos_de_carpeta,
    extraer_texto,
    listar_modelos_de_chat,
)
from evals import evaluar_respuesta, semaforo

# Los nombres de los documentos de ejemplo (para saber si mostrar preguntas sugeridas).
DOCUMENTOS_DE_EJEMPLO = {documento.nombre for documento in cargar_documentos_de_carpeta(CARPETA_CORPUS_DE_EJEMPLO)}

PREGUNTAS_SUGERIDAS = [
    "¿Cuántos días de vacaciones tiene un empleado en su tercer año?",
    "¿Cuánto tiempo fermenta el café y hasta qué humedad se seca?",
    "¿Quién fundó la cooperativa y con cuántas familias?",
    # 👇 Esta NO está en el documento: sirve para ver si el modelo inventa.
    "¿Cuál es la capital de Australia?",
]

# ── Memoria de la página (sobrevive entre clics) ────────────────────────────
st.session_state.setdefault("motor", None)  # el MotorCAG con el conocimiento cargado
st.session_state.setdefault("turnos", [])  # cada pregunta con su respuesta y su evaluación
st.session_state.setdefault("pendiente_de_auditar", False)


@st.cache_data(ttl=30, show_spinner=False)
def modelos_instalados():
    """Pregunta a Ollama qué modelos hay (y lo recuerda 30 s para no preguntar en cada clic)."""
    return listar_modelos_de_chat(ollama.Client())


def historial_para_el_modelo(turnos: list[dict]) -> list[dict]:
    """Convierte los turnos guardados en la lista usuario/asistente que espera el modelo."""
    mensajes = []
    for turno in turnos:
        respuesta = turno["respuesta"]
        mensajes.append({"role": "user", "content": respuesta.pregunta})
        mensajes.append({"role": "assistant", "content": respuesta.texto})
    return mensajes


# ════════════════════════════════════════════════════════════════════════════
# 🧭 BARRA LATERAL: experimentos
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.page_link("paginas/teoria.py", label="Leer la teoría (12 min)", icon="📖")
    st.header("🧪 Experimentos")
    permitir_conocimiento_externo = st.toggle(
        "Permitir conocimiento externo",
        value=False,
        help=(
            "APAGADO (recomendado): el modelo solo puede usar tus documentos. "
            "ENCENDIDO: puede completar con lo que 'sabe' de memoria. Enciéndelo y "
            "pregunta algo que no esté en el documento: verás cómo el juez de "
            "veracidad marca esas afirmaciones como no respaldadas."
        ),
    )
    romper_cache = st.toggle(
        "🧨 Romper la caché a propósito",
        value=False,
        help=(
            "Pone la hora exacta AL PRINCIPIO del prompt. Como el inicio cambia en cada "
            "pregunta, el modelo ya no reconoce el libro y tiene que releerlo entero. "
            "Compara en el panel ③ el tiempo de lectura con y sin este interruptor."
        ),
    )
    auditar_automaticamente = st.toggle(
        "⚖️ Auditar cada respuesta",
        value=True,
        help=(
            "Después de cada respuesta, dos 'jueces' (el mismo modelo con otras "
            "instrucciones) la califican. Tarda unos segundos más; apágalo si "
            "prefieres auditar solo algunas respuestas."
        ),
    )
    if st.button(
        "🧹 Borrar conversación",
        help="Empieza una conversación nueva. El libro sigue memorizado.",
        width="stretch",
    ):
        st.session_state.turnos = []
        st.session_state.pendiente_de_auditar = False

    st.divider()
    st.caption(
        "💰 **Costo: $0.** Todo corre en tu computadora con "
        "[Ollama](https://ollama.com): sin internet y sin enviar tus documentos a nadie."
    )

# ════════════════════════════════════════════════════════════════════════════
# 🏷️ ENCABEZADO
# ════════════════════════════════════════════════════════════════════════════
st.title("🧠 CAG Educativo")
st.caption(
    "Cache-Augmented Generation con modelos locales: mira cómo una IA *memoriza* tus "
    "documentos y cómo auditamos si lo que responde es verdad."
)
with st.expander("👣 Cómo usar este laboratorio en 3 pasos", expanded=st.session_state.motor is None):
    st.markdown(
        """
1. **Panel ①** — Elige un modelo y documentos (o usa el de ejemplo) y pulsa
   **Inyectar**. El modelo lee el "libro" una vez y lo guarda en su caché.
2. **Panel ②** — Haz preguntas. La respuesta aparece palabra por palabra.
3. **Panel ③** — Comprueba si el modelo *recordó* el libro (lectura casi
   instantánea) y qué calificaciones de **Veracidad** y **Relevancia** le dieron
   los jueces, y por qué.
"""
    )

try:
    modelos = modelos_instalados()
except Exception as error:  # típicamente: Ollama apagado
    st.error(explicar_error(error))
    st.stop()
if not modelos:
    st.error(f"No hay modelos de chat instalados. Descarga uno con: `ollama pull {MODELO_RECOMENDADO}`")
    st.stop()

panel_carga, panel_chat, panel_transparencia = st.columns([1.1, 1.3, 1.3], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# ① PANEL DE CARGA: inyección a la caché
# ════════════════════════════════════════════════════════════════════════════
with panel_carga:
    st.subheader("① Inyección a la caché")
    st.caption("Aquí eliges al \"estudiante\" y le entregas el \"libro\" que debe memorizar.")

    nombres = [modelo.nombre for modelo in modelos]
    nombre_modelo = st.selectbox(
        "🤖 Modelo local",
        options=nombres,
        index=nombres.index(MODELO_RECOMENDADO) if MODELO_RECOMENDADO in nombres else 0,
        format_func=lambda nombre: f"{nombre}  ⭐ recomendado" if nombre == MODELO_RECOMENDADO else nombre,
        help=(
            "Los modelos que tienes instalados en Ollama. Cada uno es un 'estudiante' "
            "distinto: cambiar de modelo obliga a memorizar el libro otra vez."
        ),
    )
    ficha = next(modelo for modelo in modelos if modelo.nombre == nombre_modelo)
    st.caption(
        f"{ficha.tamano} parámetros · {ficha.cuantizacion} · mesa máxima "
        f"{ficha.contexto_maximo:,} tokens · "
        + ("🧠 puede razonar" if ficha.soporta_razonamiento else "sin modo de razonamiento")
    )
    razonar = st.toggle(
        "🧠 Razonar primero",
        value=True,
        disabled=not ficha.soporta_razonamiento,
        help=(
            "El modelo 'piensa' antes de responder: tarda más, pero se equivoca menos en "
            "cálculos (p. ej. 'días de vacaciones en el tercer año'). En gpt-oss no se "
            "apaga del todo: baja de nivel 'medium' a 'low'."
        ),
    )

    archivos_subidos = st.file_uploader(
        "Sube tus documentos",
        type=["txt", "md", "pdf", "csv", "json"],
        accept_multiple_files=True,
        help=(
            "Todo el texto de estos archivos se le da completo al modelo. A diferencia "
            "de RAG, no se corta en pedazos ni se busca: se memoriza entero."
        ),
    )
    usar_documento_ejemplo = st.checkbox(
        "Incluir documentos de ejemplo",
        value=True,
        help=(
            "Cuatro documentos FICTICIOS de una cooperativa cafetalera: manual, informe "
            "anual, acta de asamblea y protocolo de calidad. Como todo es inventado, es "
            "fácil notar cuándo el modelo responde con algo que no está escrito ahí."
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
            "Cuánto tiempo mantiene Ollama el modelo y su caché en memoria desde la "
            "última pregunta (`keep_alive`). Pasado ese tiempo, lo descarga y la próxima "
            "pregunta tendrá que releer el libro. Mientras tanto ocupa memoria de tu GPU."
        ),
    )

    if st.button(
        "💉 Inyectar en la caché",
        type="primary",
        width="stretch",
        help=(
            "1) Junta los documentos en un solo bloque. 2) Calcula el tamaño de la mesa "
            "(num_ctx) y comprueba que el libro cabe. 3) Borra la memoria del modelo y le "
            "hace leer el libro una vez, midiendo cuánto tarda en frío. Inyectar de nuevo "
            "reinicia la conversación."
        ),
    ):
        documentos = [
            Documento(archivo.name, extraer_texto(archivo.name, archivo.getvalue()))
            for archivo in archivos_subidos or []
        ]
        if usar_documento_ejemplo:
            documentos.extend(cargar_documentos_de_carpeta(CARPETA_CORPUS_DE_EJEMPLO))
        if texto_pegado.strip():
            documentos.append(Documento("texto_pegado.txt", texto_pegado))

        if not documentos:
            st.warning("Primero sube un archivo, pega texto o marca los documentos de ejemplo.")
        else:
            try:
                with st.spinner("Cargando el modelo y leyendo el libro… (la primera vez tarda más)"):
                    motor_nuevo = MotorCAG(nombre_modelo, razonar=razonar)
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
        frio = informe.lectura_en_frio
        st.success(f"✅ {motor.modelo.nombre} memorizó {len(informe.documentos)} documento(s)")

        columna_a, columna_b = st.columns(2)
        columna_a.metric(
            "Tokens",
            f"{informe.tokens_en_memoria:,}",
            help=(
                "Tokens del libro: lo que ocupa (más las instrucciones) en la mesa. Un token es un "
                "pedacito de palabra (≈ 3–4 caracteres en español)."
            ),
        )
        columna_b.metric(
            "Mesa",
            f"{informe.num_ctx:,}",
            help=(
                "La ventana de contexto (`num_ctx`) que reservamos: el libro + espacio para conversar. "
                "Si fuera demasiado pequeña, Ollama recortaría el libro SIN AVISAR."
            ),
        )
        st.progress(
            min(informe.fraccion_de_mesa, 1.0),
            text=f"El libro ocupa el {informe.fraccion_de_mesa:.0%} de la mesa",
        )
        st.markdown(
            f"📖 **Lectura en frío:** {frio.segundos_leyendo:.2f} s "
            f"({frio.velocidad_de_lectura:,.0f} tokens/s)",
            help=(
                "Lo que tardó en leer el libro por primera vez, sin apuntes previos (sin "
                "caché). Es la referencia para medir cuánto acelera la caché después."
            ),
        )
        if frio.velocidad_de_lectura:
            segundos_libro_grande = TOKENS_DE_UN_LIBRO_GRANDE / frio.velocidad_de_lectura
            st.caption(
                f"📈 A esta velocidad, un libro de {TOKENS_DE_UN_LIBRO_GRANDE:,} tokens tardaría "
                f"al menos ~{segundos_libro_grande:.0f} s en leerse **en cada pregunta** si no "
                "hubiera caché. Con CAG se lee una sola vez."
            )
        if frio.modelo_recien_cargado:
            st.caption(f"Cargar el modelo en la memoria tomó {frio.segundos_cargando_modelo:.1f} s.")
        if informe.aviso:
            st.warning(informe.aviso)
        if motor.modelo.nombre != nombre_modelo or motor.razonar != razonar:
            st.info("Cambiaste el modelo o el razonamiento: pulsa **Inyectar** para aplicarlo.")

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
        if romper_cache:
            st.caption("🧨 Caché rota a propósito: cada pregunta releerá el libro completo.")

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
                    st.markdown(respuesta.texto)
                    st.caption(
                        f"#{numero} · 📖 leyó el prompt en {respuesta.medicion.segundos_leyendo:.2f} s "
                        f"· ⏱️ {respuesta.segundos_totales:.1f} s en total"
                    )

        pregunta_sugerida = None
        if DOCUMENTOS_DE_EJEMPLO & set(motor.documentos_cargados):
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
                            historial=historial_para_el_modelo(st.session_state.turnos),
                            permitir_conocimiento_externo=permitir_conocimiento_externo,
                            romper_cache=romper_cache,
                        )
                        with st.spinner("Pensando…"):
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
                st.session_state.pendiente_de_auditar = auditar_automaticamente

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
            "- ⚡ **La caché en acción**: si recordó el libro o tuvo que releerlo.\n"
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
        medicion = respuesta.medicion

        # ── ⚡ La caché en acción ────────────────────────────────────────────
        st.markdown("#### ⚡ La caché en acción")
        segundos_sin_cache = medicion.segundos_sin_cache(motor.velocidad_de_lectura_en_frio)
        aceleracion = (
            segundos_sin_cache / medicion.segundos_leyendo
            if segundos_sin_cache and medicion.segundos_leyendo
            else None
        )

        # El diagnóstico, en orden de gravedad.
        if respuesta.prompt_recortado:
            st.error(
                "🚨 **El libro no llegó completo.** Ollama leyó menos tokens de los que ocupa "
                "el libro: lo recortó en silencio y el modelo pudo inventar datos. Es el fallo "
                "más peligroso de CAG. Borra la conversación o carga menos documentos."
            )
        if respuesta.cache_rota_a_proposito:
            st.warning(
                "🧨 **Caché rota a propósito.** El inicio del prompt cambió, así que releyó el "
                "libro entero. Compara este tiempo de lectura con el de una pregunta normal."
            )
        elif medicion.modelo_recien_cargado:
            st.warning(
                "❄️ **El modelo se había descargado** (pasó el tiempo de la caché o se cambió "
                "algún ajuste). Tuvo que cargarse y releer el libro desde cero."
            )
        elif aceleracion and aceleracion >= 3:
            st.success(
                f"✅ **Recordó el libro:** leyó el prompt ~{aceleracion:.0f}× más rápido que en "
                "frío, porque reutilizó sus apuntes (la caché KV)."
            )
        else:
            st.info(
                "ℹ️ La lectura tardó parecido a una lectura en frío: la caché casi no se notó "
                "(pasa con libros muy cortos o si se rompió el prefijo)."
            )

        columna_a, columna_b = st.columns(2)
        columna_a.metric(
            "Tokens leídos",
            f"{medicion.tokens_del_prompt:,}",
            help="Todo el prompt: libro + instrucciones + conversación previa + tu pregunta.",
        )
        columna_b.metric(
            "Tiempo de lectura",
            f"{medicion.segundos_leyendo:.2f} s",
            help="Lo que tardó en procesar el prompt. Con caché, solo lee lo nuevo.",
        )
        columna_a.metric(
            "Sin caché",
            "—" if segundos_sin_cache is None else f"~{segundos_sin_cache:.2f} s",
            help=(
                "Lo que habría tardado en leer el prompt sin caché (estimación: tokens "
                "leídos ÷ velocidad en frío medida al inyectar)."
            ),
        )
        columna_b.metric(
            "Tokens/s aparentes",
            f"{medicion.velocidad_de_lectura:,.0f}",
            help=(
                "Tokens leídos ÷ tiempo de lectura. Si es mucho mayor que la velocidad en "
                "frío, es porque no leyó: recordó."
            ),
        )
        columna_a.metric(
            "Primer token",
            "—"
            if respuesta.segundos_hasta_primer_token is None
            else f"{respuesta.segundos_hasta_primer_token:.1f} s",
            help="Cuánto tardó en aparecer la primera palabra (incluye el tiempo razonando).",
        )
        columna_b.metric(
            "Generados",
            f"{medicion.tokens_generados:,}",
            help=(
                f"Tokens escritos (razonamiento + respuesta) a "
                f"{medicion.velocidad_de_escritura:,.0f} tok/s. La caché acelera la lectura, "
                "no la escritura."
            ),
        )
        if turno["modo_libre"]:
            st.caption("🧪 Esta respuesta se generó en modo libre (con conocimiento externo).")

        if respuesta.razonamiento:
            with st.expander("🧠 Lo que pensó el modelo antes de responder"):
                st.caption("Es su razonamiento interno, tal cual lo escribió (a veces en inglés).")
                st.markdown(respuesta.razonamiento)

        st.divider()

        # ── ⚖️🎯 EVALS ────────────────────────────────────────────────────────
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
            # ── ⚖️ Veracidad ──────────────────────────────────────────────────
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

            # ── 🎯 Relevancia ─────────────────────────────────────────────────
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

            # ── Lo que tomó auditar ──────────────────────────────────────────
            lectura_del_juez = (
                f" · el juez de veracidad leyó el libro en {veracidad.medicion.segundos_leyendo:.2f} s"
                if veracidad.medicion
                else ""
            )
            st.caption(f"🧾 Auditoría: {evaluacion.segundos:.1f} s{lectura_del_juez}.")
            st.caption(
                "⚠️ El juez es el mismo modelo que respondió y podría ser indulgente consigo "
                "mismo. Por eso te muestra su evidencia: tú tienes la última palabra."
            )

        with st.expander("📐 ¿Cómo se calculan estas métricas?"):
            st.markdown(
                """
**La caché en acción** — Ollama informa cuántos tokens tenía el prompt y cuánto
tardó en leerlos. Comparamos con la **velocidad en frío** que medimos al
inyectar: si ahora "leyó" mucho más rápido, es porque reutilizó la caché.

**Veracidad (Faithfulness)** — inspirada en RAGas:
1. Un juez divide la respuesta en *afirmaciones atómicas* (un hecho por frase).
2. Para cada una busca la **cita textual** del documento que la respalda, y
   Python comprueba que esa cita exista de verdad (no le creemos sin pruebas).
3. `puntaje = afirmaciones respaldadas ÷ afirmaciones totales` (lo calcula Python, no el modelo).

**Relevancia (Answer Relevancy)** — adaptada de RAGas:
1. Otro juez, que **no ve el documento**, escribe 3 *preguntas inversas*:
   ¿qué preguntas estaría contestando esta respuesta?
2. Las compara con tu pregunta y califica con una rúbrica de 1 a 5.
3. `puntaje = (rúbrica − 1) ÷ 4`

Más detalles y fuentes en la página **📖 Teoría**.
"""
            )
