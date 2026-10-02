# 📖 Teoría de CAG en 10 minutos

> **Cómo leer esto:** cada sección empieza con la idea en una frase (💡). Si solo
> lees esas frases, ya entiendes lo esencial. Los números entre corchetes, como
> [1], apuntan a las **fuentes** del final: artículos científicos revisados por
> pares y documentación oficial.

---

## 1. Tres palabras que necesitas

💡 *Un modelo de lenguaje lee y escribe en "tokens", y solo puede tener cierta
cantidad de ellos "sobre la mesa" a la vez.*

| Palabra | Qué significa | Analogía |
|---|---|---|
| **LLM** (modelo de lenguaje) | Un programa entrenado para predecir el siguiente pedacito de texto. Sabe lo que había en sus datos de entrenamiento, pero **no conoce tus documentos**. | Un estudiante muy culto que nunca ha visto *tu* libro. |
| **Token** | Un pedacito de palabra. En español, un token suele tener de 3 a 4 caracteres. Todo se mide en tokens. | Las sílabas con las que lee el estudiante. |
| **Ventana de contexto** | Cuántos tokens puede considerar el modelo a la vez. En Ollama se llama `num_ctx`. | El tamaño de la mesa de estudio: lo que no cabe en la mesa, no existe para él. |

---

## 2. El problema: el modelo no conoce lo tuyo

💡 *Para que un modelo responda sobre tus documentos hay tres caminos:
reentrenarlo, buscarle fragmentos (RAG) o hacer que lo lea todo antes (CAG).*

- **Reentrenarlo** (*fine-tuning*): caro, lento y hay que repetirlo cada vez que
  cambian los documentos.
- **RAG**: buscar en cada pregunta los fragmentos relevantes y dárselos.
- **CAG**: darle todo el conocimiento de una vez y guardar el resultado de
  haberlo leído.

---

## 3. RAG: ir a la biblioteca en cada pregunta

💡 *RAG corta tus documentos en fragmentos, busca los más parecidos a la
pregunta y solo esos se los muestra al modelo.* [2]

```
Pregunta ──► 🔎 Buscador ──► 3 fragmentos ──► 🤖 Modelo ──► Respuesta
               (base de datos vectorial)
```

- ✅ **Escala** a bibliotecas enormes (millones de documentos).
- ⚠️ **Si la búsqueda falla, la respuesta falla**: el modelo nunca ve el
  fragmento correcto. Además, hay más piezas que construir y mantener
  (troceado, *embeddings*, base vectorial, buscador).

---

## 4. CAG: memorizar el libro antes del examen

💡 *CAG precarga todo el conocimiento en la ventana de contexto, guarda el
"estado mental" resultante (la caché KV) y lo reutiliza en cada pregunta, sin
buscar nada.* [1]

```
Una sola vez:   📚 Todo el conocimiento ──► 🤖 Modelo ──► 💾 Caché KV
Cada pregunta:  💾 Caché KV + Pregunta  ──► 🤖 Modelo ──► Respuesta
```

El término viene del artículo *"Don't Do RAG: When Cache-Augmented Generation is
All You Need for Knowledge Tasks"* (Chan et al., WWW 2025) [1]. Los autores
proponen precargar los documentos, guardar su caché de claves y valores, y
responder sobre ella. Según su estudio, cuando la base de conocimiento es
**acotada** y cabe en el contexto, CAG elimina la latencia de búsqueda y los
errores de recuperación, con un sistema más simple.

---

## 5. ¿Qué es exactamente la "caché KV"?

💡 *Al leer, el modelo calcula unas notas internas (claves y valores) para cada
token; guardarlas evita volver a leer lo que ya leyó.*

Los modelos actuales son **Transformers** [3]. Al procesar cada token, cada capa
calcula una **clave** (*Key*) y un **valor** (*Value*): son como los apuntes del
estudiante sobre esa parte del texto. Para escribir la siguiente palabra, el
modelo consulta los apuntes de **todos** los tokens anteriores (eso es la
"atención"). Guardar esos apuntes es la **caché KV** [4].

**La regla del prefijo:** los apuntes de cada token dependen de todo lo que vino
antes. Por eso solo se pueden reutilizar si **el inicio del prompt es
idéntico**:

```
Pregunta 1:  [📚 Libro][Instrucciones][¿Pregunta 1?]
Pregunta 2:  [📚 Libro][Instrucciones][¿Pregunta 2?]
              └──── idéntico: se reutiliza ────┘ └ se lee ┘
```

Si cambias **un solo carácter al principio** (por ejemplo, la hora), todo lo
que viene después se tiene que recalcular. En el laboratorio puedes comprobarlo
con el experimento **🧨 Romper la caché**.

Reutilizar la caché entre peticiones que comparten el inicio es una técnica
estudiada y usada en la industria: los sistemas de servicio de modelos la
comparten entre peticiones [5], y *Prompt Cache* [6] midió aceleraciones de 8 a
60 veces en la latencia. Los proveedores comerciales la ofrecen como *prompt
caching* [13]; **Ollama** la aplica automáticamente cuando el inicio del prompt
coincide.

**El precio:** la caché ocupa memoria y **crece con cada token** de contexto
[4]. Por eso la mesa (`num_ctx`) no puede ser infinita: en nuestras pruebas con
una GPU de 16 GB, `gpt-oss:20b` cupo completo en la GPU con 65,536 tokens, pero
con 131,072 tuvo que usar la CPU y se volvió lento.

---

## 6. ¿CAG o RAG? Una guía rápida

💡 *CAG para conocimiento pequeño y estable; RAG para bibliotecas grandes o que
cambian a cada rato.*

| Situación | Mejor opción | Por qué |
|---|---|---|
| Un manual, un reglamento, un temario (cabe en la mesa) | **CAG** | Sin búsqueda, sin piezas extra, respuestas rápidas. |
| Miles de documentos | **RAG** | No caben en ninguna mesa. |
| Información que cambia cada hora | **RAG** | En CAG habría que volver a inyectar todo. |
| Preguntas que necesitan relacionar partes lejanas del documento | **CAG** | El modelo ve todo a la vez; RAG podría traer solo una parte. |
| Equipo con poca memoria (VRAM) | **RAG** | La caché de un contexto grande ocupa mucha memoria. |

También existen híbridos: usar RAG para elegir qué documentos cargar y CAG para
memorizarlos.

---

## 7. Los límites de CAG (lo que no se suele contar)

💡 *Que el libro quepa en la mesa no garantiza que el modelo lo use bien.*

1. **Todo debe caber.** Si no cabe en `num_ctx`, no es CAG.
2. **El truncado silencioso.** Por defecto, Ollama usa una mesa de solo 4,096
   tokens [11]. En nuestras pruebas, con una mesa demasiado pequeña, Ollama
   **recortó el documento sin avisar**, y el modelo respondió con total
   seguridad que la fundadora era *"Ernestina Mena, en 2006"*: un dato
   inventado. Por eso este proyecto calcula `num_ctx` automáticamente y vigila
   que el libro llegue completo.
3. **"Perdido en el medio".** Los modelos aprovechan mejor la información del
   principio y del final del contexto que la del medio [7].
4. **La mesa anunciada no es la mesa útil.** Muchos modelos que dicen aceptar
   32,000 tokens o más pierden calidad mucho antes; en el estudio RULER, solo la
   mitad mantuvo un desempeño satisfactorio a 32K [8].
5. **La caché caduca.** Ollama descarga el modelo tras 5 minutos sin uso por
   defecto (`keep_alive`) [11]; al descargarlo, los apuntes se pierden. Aquí
   puedes elegir 10 minutos o 1 hora.
6. **Actualizar cuesta.** Si cambia el documento, hay que volver a inyectarlo.

---

## 8. ¿Cómo sabemos si responde bien? Evals

💡 *Una "alucinación" es un dato que suena creíble pero no sale de la fuente;
para detectarla usamos un modelo como juez, con evidencia visible.*

- **Alucinación:** contenido sin sentido o que no es fiel a la fuente
  proporcionada [12].
- **RAGAS** [9] propuso métricas para evaluar sistemas que responden con
  documentos **sin necesitar respuestas de referencia**. Usamos dos:
  - **Fidelidad** (*faithfulness*): se divide la respuesta en afirmaciones y se
    verifica cada una contra el contexto.
    `puntaje = afirmaciones respaldadas ÷ afirmaciones totales`.
  - **Relevancia de la respuesta** (*answer relevance*): se generan preguntas a
    partir de la respuesta y se comparan con la pregunta original. RAGAS las
    compara con *embeddings*; aquí un juez las califica con una rúbrica de 1 a 5.
- **LLM-as-a-Judge** [10]: usar un modelo como evaluador. Los jueces fuertes
  coinciden con las personas más del 80 % de las veces, pero tienen **sesgos**:
  prefieren respuestas largas, se dejan influir por el orden y tienden a
  favorecer sus propias respuestas (*autopromoción*). En este proyecto el juez
  es el mismo modelo que respondió, así que **muestra su evidencia y tú tienes
  la última palabra**.

> 🧩 **Reto para quien quiera más:** tienes instalado `nomic-embed-text`.
> Implementa la relevancia "como en el artículo": genera las preguntas
> inversas, conviértelas en *embeddings* y calcula su similitud con la pregunta
> original.

---

## 9. Dónde ver cada idea en este proyecto

| Concepto | En el código | En la app |
|---|---|---|
| Precarga del conocimiento | `MotorCAG.cargar_conocimiento()` en `cag_core.py` | Panel ① *Inyectar* |
| Regla del prefijo | `MotorCAG.prompt_de_sistema()` (el libro va primero) | Panel ③ *Tiempo de lectura* |
| Tamaño de la mesa | `calcular_tamano_de_mesa()` → `num_ctx` | Panel ① *Mesa* |
| Caducidad de la caché | `keep_alive` en `parametros_base()` | Panel ① *Duración de la caché* |
| Invalidador silencioso | `romper_cache=True` | Barra lateral → 🧨 *Romper la caché* |
| Truncado silencioso | `RespuestaCAG.prompt_recortado` | Panel ③, alerta roja |
| Fidelidad | `evaluar_veracidad()` en `evals.py` | Panel ③ ⚖️ |
| Relevancia | `evaluar_relevancia()` en `evals.py` | Panel ③ 🎯 |

---

## 10. Glosario exprés

- **Caché KV:** apuntes internos (claves y valores) que el modelo guarda de lo que ya leyó.
- **Prefill / lectura:** la fase en la que el modelo procesa el prompt. Es lo que la caché acelera.
- **Generación:** la fase en la que escribe la respuesta, token por token. La caché no la acelera.
- **`num_ctx`:** tamaño de la ventana de contexto en Ollama (la mesa).
- **`keep_alive`:** cuánto tiempo mantiene Ollama el modelo, y su caché, en memoria.
- **VRAM:** la memoria de la tarjeta gráfica, donde viven el modelo y su caché.
- **Embedding:** un texto convertido en una lista de números que representa su significado; RAG los usa para buscar.
- **Alucinación:** un dato que suena creíble pero no sale de la fuente.

---

## 📚 Fuentes

1. Chan, B. J., Chen, C.-T., Cheng, J.-H. y Huang, H.-H. (2025). *Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks.* The ACM Web Conference 2025 (WWW '25), artículo corto. [arXiv:2412.15605](https://arxiv.org/abs/2412.15605)
2. Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — Versión visual y amigable: Alammar, J. *[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*.
4. Pope, R., Douglas, S., Chowdhery, A., et al. (2023). *Efficiently Scaling Transformer Inference.* MLSys 2023. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
5. Kwon, W., Li, Z., Zhuang, S., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
6. Gim, I., Chen, G., Lee, S., Sarda, N., Khandelwal, A. y Zhong, L. (2024). *Prompt Cache: Modular Attention Reuse for Low-Latency Inference.* MLSys 2024. [arXiv:2311.04934](https://arxiv.org/abs/2311.04934)
7. Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* Transactions of the ACL (TACL). [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
8. Hsieh, C.-P., Sun, S., Kriman, S., et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
9. Es, S., James, J., Espinosa-Anke, L. y Schockaert, S. (2024). *RAGAs: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024, System Demonstrations, pp. 150–158. [ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/)
10. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023, Datasets and Benchmarks. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
11. Ollama. *FAQ* (tamaño del contexto, `keep_alive`, peticiones simultáneas): [docs.ollama.com/faq](https://docs.ollama.com/faq) · *Structured outputs*: [docs.ollama.com/capabilities/structured-outputs](https://docs.ollama.com/capabilities/structured-outputs) · *API*: [github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)
12. Ji, Z., Lee, N., Frieske, R., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys. [arXiv:2202.03629](https://arxiv.org/abs/2202.03629)
13. Anthropic. *Prompt caching* (documentación oficial, ejemplo comercial de la misma idea): [platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
