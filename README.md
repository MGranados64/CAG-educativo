# 🧠 CAG Educativo

**Aprende qué es Cache-Augmented Generation (CAG) viendo un sistema real funcionar:**
cómo un modelo de IA *memoriza* tus documentos, cuánto acelera la caché, cómo
auditamos si lo que responde es verdad y **cómo se compara cara a cara con RAG**.

**100 % gratis y privado:** todo corre en tu computadora con [Ollama](https://ollama.com).
Sin API keys, sin pagos y sin enviar tus documentos a nadie.

![Licencia MIT](https://img.shields.io/badge/licencia-MIT-green) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/interfaz-Streamlit-red) ![Ollama](https://img.shields.io/badge/modelos-Ollama%20local-black) ![Costo $0](https://img.shields.io/badge/costo-%240-brightgreen)

---

## 📚 ¿Qué es CAG? ¿En qué se diferencia de RAG?

Imagina que un estudiante tiene que presentar un examen sobre un libro.

| | 🏃 **RAG** (Retrieval-Augmented Generation) | 🧠 **CAG** (Cache-Augmented Generation) |
|---|---|---|
| **La analogía** | En **cada pregunta** el estudiante corre a la biblioteca, busca unas páginas, las lee y responde. | **Antes del examen** le damos el libro completo para que lo memorice. Luego responde sin buscar. |
| **Cómo funciona** | Los documentos se cortan en fragmentos, se guardan en una base de datos vectorial y en cada pregunta se buscan los más parecidos. | Todos los documentos se cargan completos en la memoria de trabajo del modelo (su *ventana de contexto*) y se guarda su caché KV. |
| **Ventaja** | Funciona con bibliotecas gigantes. | Sin búsqueda: nada se pierde porque "el bibliotecario trajo la página equivocada". Más simple. |
| **Límite** | Si la búsqueda falla, la respuesta falla. | Todo el conocimiento debe **caber** en la memoria del modelo. |

### ¿Y la "caché"?

Para leer un texto, el modelo calcula unas notas internas por cada token (las
*claves y valores*, o **caché KV**). Ollama las conserva en la memoria de tu tarjeta
gráfica: si la siguiente pregunta empieza con el mismo libro, el modelo **no lo vuelve
a leer**, consulta sus apuntes. En nuestras pruebas, la lectura del documento de
ejemplo bajó de **0.25 s a 0.03 s** (~9× más rápido).

👉 **¿Quieres entenderlo a fondo?** Lee la [**Teoría de CAG en 12 minutos**](TEORIA.md):
ligera, con analogías y con fuentes científicas (también está dentro de la app).

---

## ⚙️ Cómo funciona este proyecto

```mermaid
flowchart LR
    A["📄 Tus documentos"] -->|"① Inyectar"| B[("🧠 Caché KV en tu GPU<br/>(Ollama)")]
    C["❓ Tu pregunta"] -->|"② Chat"| D["Modelo local<br/>gpt-oss:20b"]
    B -->|"se reutiliza sin releer"| D
    D --> E["💬 Respuesta"]
    E -->|"③ Auditoría"| F["⚖️ Juez de Veracidad"]
    E -->|"③ Auditoría"| G["🎯 Juez de Relevancia"]
    B -.->|"el juez consulta el mismo libro"| F
```

La app tiene tres páginas: **🧪 Laboratorio**, **⚔️ CAG vs RAG** y **📖 Teoría**.
El laboratorio tiene tres paneles:

1. **① Inyección a la caché**: eliges un modelo instalado y tus documentos (o el de
   ejemplo). El sistema calcula el tamaño de la "mesa" (`num_ctx`), comprueba que el
   libro cabe y lo lee una vez **midiendo cuánto tarda en frío**.
2. **② Chat**: preguntas lo que quieras y la respuesta aparece palabra por palabra.
3. **③ Transparencia y EVALS**: ves si el modelo **recordó** el libro (lectura casi
   instantánea) o tuvo que releerlo, y la calificación de dos jueces:
   - ⚖️ **Veracidad** *(Faithfulness)*: divide la respuesta en afirmaciones y busca la
     **cita textual** del documento que respalda cada una; Python comprueba que la cita
     exista de verdad. Si no la encuentra, puede ser una **alucinación**.
     `puntaje = respaldadas ÷ totales`
   - 🎯 **Relevancia** *(Answer Relevancy)*: comprueba si contestó lo que preguntaste,
     con "preguntas inversas" y una rúbrica de 1 a 5.

Ambas métricas están inspiradas en [RAGAS](https://aclanthology.org/2024.eacl-demo.16/)
y usan la técnica **LLM-as-a-Judge**: el mismo modelo, con otras instrucciones, actúa
como profesor que corrige el examen.

### ⚔️ CAG vs RAG, cara a cara

La página **CAG vs RAG** enfrenta a los dos sistemas en una **pelea justa**: mismos
documentos, mismas preguntas, mismo modelo, misma mesa (`num_ctx`), temperatura 0 y
mismos jueces. La única diferencia es qué texto ve el modelo: **todo** (CAG) o las
***k* fichas** que encuentra el buscador (RAG, con `nomic-embed-text`).

- **🥊 Pregunta libre:** una pregunta y dos respuestas lado a lado, con los tiempos y
  las fichas que trajo el buscador.
- **🏁 Banco de pruebas:** todas las preguntas de referencia (con su respuesta correcta),
  calificadas con tres jueces: ✅ **exactitud**, ⚖️ **veracidad** y 🔎 **recuperación**
  (¿el buscador de RAG trajo lo necesario?). Exporta todo a **JSON**.

**Nuestros resultados** (corpus de la cooperativa, `gpt-oss:20b`, k = 4, RTX 5070 Ti):

| 16 preguntas de referencia | 🧠 CAG | 🔎 RAG |
|---|---|---|
| Exactitud | **100 %** | 81 % |
| Veracidad (nada inventado) | **97 %** | 88 % |
| Preguntas que necesitan 2 documentos | **100 %** | 33 % |
| Tokens leídos por pregunta | 4,637 | **916** |
| Tiempo total por pregunta | 4.1 s | **1.3 s** |

RAG empata en datos puntuales, cálculos y preguntas trampa, pero falla cuando la
respuesta está repartida en varios documentos, y una vez **inventó** un dato para
rellenar lo que su buscador no trajo. El análisis completo, incluida la "paradoja de
la velocidad", está en la [sección 9 de la teoría](TEORIA.md#9-cag-vs-rag-cara-a-cara).

### 🗂️ Mapa de archivos

| Archivo | Qué hace |
|---|---|
| [`cag_core.py`](cag_core.py) | El motor CAG: prepara el conocimiento, elige `num_ctx`, precalienta la caché y responde. |
| [`rag_core.py`](rag_core.py) | El rival: trocea, calcula embeddings, busca las *k* fichas más parecidas y responde. |
| [`evals.py`](evals.py) | Los jueces (veracidad, relevancia, exactitud y recuperación) y casos para "evaluar al evaluador". |
| [`comparacion.py`](comparacion.py) | El banco de pruebas CAG vs RAG: corpus, pelea justa, resumen y exportación a JSON. |
| [`app.py`](app.py) | La puerta de entrada de la app (el índice de páginas). |
| [`paginas/`](paginas/) | Las tres páginas: laboratorio, CAG vs RAG y teoría. |
| [`TEORIA.md`](TEORIA.md) | La teoría, ligera y con fuentes. |
| [`datos/cooperativa/`](datos/cooperativa/) | 4 documentos **100 % ficticios** + `preguntas.json` (16 preguntas con su respuesta correcta). |
| [`datos/paper_cag/`](datos/paper_cag/) | `preguntas.json` (10 preguntas) para el paper de CAG, que se descarga bajo demanda. |
| [`pruebas/`](pruebas/) | Pruebas automáticas que no necesitan Ollama. |

Todo el código está comentado en español, con analogías, para leerse de arriba a abajo.

---

## 🚀 Instalación rápida

**Necesitas:** Python 3.10 o superior y [Ollama](https://ollama.com/download).

```bash
# 1. Descarga los modelos (una sola vez): el de chat (≈ 13 GB) y el de
#    embeddings que usa el buscador de RAG (≈ 270 MB)
ollama pull gpt-oss:20b
ollama pull nomic-embed-text

# 2. Descarga el proyecto y entra en la carpeta
git clone https://github.com/MGranados64/CAG-educativo.git
cd CAG-educativo

# 3. Crea un entorno virtual (una "caja" aislada para las librerías)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS (bash/zsh)
# source .venv/bin/activate.fish   # Linux con fish
# .venv\Scripts\activate           # Windows

# 4. Instala las dependencias
pip install -r requirements.txt

# 5. ¡Arranca la app! (con Ollama encendido)
streamlit run app.py
```

Se abrirá tu navegador en `http://localhost:8501`.

### 🤖 ¿Qué modelo usar?

La app muestra todos los modelos de chat que tengas instalados en Ollama; puedes
probar cualquiera. Estos son los resultados de nuestras pruebas con el documento de
ejemplo en una **NVIDIA RTX 5070 Ti (16 GB)**:

| Modelo | ¿Calculó bien las vacaciones? | Como juez | Velocidad de escritura | Memoria de video |
|---|---|---|---|---|
| **`gpt-oss:20b`** ⭐ (razonamiento medio) | ✅ | ✅ correcto, ~6 s | ~165 tokens/s | ~12 GB |
| `gemma4:12b` con razonamiento | ✅ | ✅ excelente, pero ~17 s | ~80 tokens/s | ~8.4 GB |
| `gemma4:12b` sin razonamiento | ❌ | ✅ excelente | ~80 tokens/s | ~8.4 GB |
| `gemma4:latest` (8B) | ❌ | ✅ bueno | ~160 tokens/s | ~3.4 GB |

¿Tu equipo tiene menos memoria? `gemma4:latest` ocupa solo ~3.4 GB (activa *Razonar
primero* para que calcule mejor); también puedes bajar el límite de la mesa con
`CAG_NUM_CTX_MAXIMO`. Ollama también funciona solo con CPU, pero más lento.

### 🧪 Cinco experimentos para empezar

1. **La caché en acción:** pulsa *Inyectar*, haz dos preguntas y mira en el panel ③
   el mensaje *"Recordó el libro"* y el tiempo de lectura.
2. **🧨 Romper la caché:** activa ese interruptor en la barra lateral y repite una
   pregunta. Al cambiar el inicio del prompt, el modelo tiene que releer todo.
3. **Honestidad:** pregunta *"¿Cuál es la capital de Australia?"*. En modo estricto,
   el modelo debe decir que no está en el documento.
4. **Caza de alucinaciones:** activa *"Permitir conocimiento externo"* y repite la
   pregunta. El juez de Veracidad marcará con ❌ lo que no sale del documento.
5. **CAG vs RAG:** en la página ⚔️, corre el banco de pruebas y mira la *exactitud por
   tipo de pregunta*. Después cambia *k* (las fichas que trae RAG) a 1 y a 8 y vuelve a
   correrlo.

### 🖥️ También desde la terminal

```bash
python cag_core.py                         # lectura en frío vs. con caché
python rag_core.py                         # cómo trocea y qué fichas encuentra el buscador
python evals.py                            # pone a prueba a los jueces
python comparacion.py cooperativa          # banco de pruebas CAG vs RAG → resultados/*.json
python comparacion.py paper_cag --descargar
python -m unittest discover -s pruebas -t .   # pruebas automáticas (sin Ollama, < 1 s)
```

### 📄 Documentos de prueba y licencias

¿Se puede usar cualquier paper si se cita bien? **No necesariamente:** citar reconoce
la autoría, pero el permiso para copiar o redistribuir lo da la **licencia**.

- **Cooperativa Nube Alta** (`datos/cooperativa/`): 4 documentos inventados para este
  proyecto. Como son ficticios, el modelo no puede saber las respuestas de memoria: lo
  que no salga de ahí es una alucinación.
- **Paper de CAG** (`datos/paper_cag/`): [Chan et al., 2025](https://arxiv.org/abs/2412.15605),
  bajo licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.es).
  Esa licencia permitiría compartirlo con atribución, pero aun así **no se incluye en el
  repositorio**: la app lo descarga de arXiv cuando lo pides y lo guarda con su
  atribución. Detalles en [`datos/paper_cag/_LEEME.md`](datos/paper_cag/_LEEME.md).

### ⚙️ Ajustes opcionales (variables de entorno)

| Variable | Para qué | Valor por defecto |
|---|---|---|
| `CAG_MODELO` | Modelo preseleccionado | `gpt-oss:20b` |
| `CAG_NUM_CTX_MAXIMO` | Mesa máxima (ajústala a la memoria de tu GPU) | `65536` |
| `CAG_MODELO_EMBEDDINGS` | Modelo de embeddings del buscador de RAG | `nomic-embed-text` |
| `OLLAMA_HOST` | Dónde está el servidor de Ollama | `http://localhost:11434` |

---

## 🎓 Propósito educativo

Este proyecto existe para **enseñar**, no para producción. Está pensado para que
cualquier persona, incluso sin experiencia en programación o IA, pueda:

- entender la diferencia entre **CAG** y **RAG** con un ejemplo que funciona, y
  **compararlos cara a cara** con los mismos documentos y las mismas preguntas;
- **ver** la caché trabajando y **romperla** a propósito para entender por qué funciona;
- conocer los límites reales de CAG, como el **truncado silencioso** cuando el libro no cabe;
- aprender a **auditar** una IA: medir alucinaciones y relevancia con evidencia visible.

Los jueces también son IA (y aquí son el mismo modelo que responde), así que pueden
equivocarse: por eso muestran su evidencia y su razonamiento. **Tú tienes la última
palabra.**

---

## 📄 Licencia

[MIT](LICENSE): úsalo, modifícalo y compártelo libremente.
