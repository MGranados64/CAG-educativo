# 📖 Teoría de CAG en 12 minutos

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
- **No le creas al juez sin pruebas.** El juez de veracidad debe copiar la
  **cita textual** que respalda cada dato, y Python la comprueba: la mayor parte
  de la cita (el 70 % de sus pares de palabras) y **todas sus cifras** deben
  existir en los documentos, y la cita debe tener que ver con la afirmación. En
  nuestras pruebas, sin esta comprobación el juez aprobó una vez *"La capital de
  Australia es Canberra"* copiando como "evidencia" la etiqueta de un documento.
  Con ella no volvió a pasar. No exigimos una copia letra por letra porque los
  modelos casi nunca la hacen: unen renglones de tablas, quitan negritas o
  cambian comillas. Esa exigencia rechazó muchas respuestas correctas y nos
  obligó a afinarla. Si el juez describe en vez de citar, el dato cuenta como no
  respaldado: preferimos ese error, que puedes revisar, a aprobar una
  alucinación.
- **LLM-as-a-Judge** [10]: usar un modelo como evaluador. Los jueces fuertes
  coinciden con las personas más del 80 % de las veces, pero tienen **sesgos**:
  prefieren respuestas largas, se dejan influir por el orden y tienden a
  favorecer sus propias respuestas (*autopromoción*). En este proyecto el juez
  es el mismo modelo que respondió, así que **muestra su evidencia y tú tienes
  la última palabra**.

Cuando además **conocemos la respuesta correcta** (como en el banco de pruebas de
la sección 9), RAGAS propone dos métricas más que usamos aquí: **exactitud**
(*answer correctness*: ¿coincide con la respuesta correcta?) y **recuperación**
(*context recall*: ¿el buscador de RAG trajo la información necesaria?).

> 🧩 **Reto para quien quiera más:** tienes instalado `nomic-embed-text`.
> Implementa la relevancia "como en el artículo": genera las preguntas
> inversas, conviértelas en *embeddings* y calcula su similitud con la pregunta
> original.

---

## 9. CAG vs RAG cara a cara

💡 *Para que una comparación enseñe algo, todo debe ser igual excepto lo que
quieres medir.*

**RAG por dentro** (en `rag_core.py`):

1. **Trocear:** los documentos se cortan en fichas de ~1,200 caracteres, sin
   mezclar secciones. A cada ficha se le antepone su documento y su sección.
2. **Embeddings:** `nomic-embed-text` [14] convierte cada ficha en una lista de
   768 números que representa su significado. Su documentación exige anteponer
   `search_document:` a las fichas y `search_query:` a las preguntas [15].
3. **Buscar:** se compara la pregunta con todas las fichas (*similitud coseno*)
   y se toman las *k* más parecidas (4 por defecto).
4. **Responder:** el mismo modelo de chat recibe solo esas fichas.

**Las reglas de la pelea justa:** mismos documentos, mismas preguntas (con su
respuesta correcta conocida), mismo modelo, misma mesa (`num_ctx`), mismo
razonamiento, temperatura 0 y los mismos jueces. La única diferencia es qué
texto ve el modelo: todo (CAG) o *k* fichas (RAG).

**Lo que encontramos** (corpus de la cooperativa, `gpt-oss:20b`, k = 4, RTX 5070 Ti):

| 16 preguntas de referencia | 🧠 CAG | 🔎 RAG |
|---|---|---|
| Exactitud | **100 %** | 81 % |
| Veracidad (nada inventado) | **97 %** | 88 % |
| Preguntas que necesitan 2 documentos (3) | **100 %** | 33 % |
| Tokens leídos por pregunta | 4,637 | **916** |
| Tiempo de lectura del prompt | **0.04 s** (desde la caché) | 0.10 s |
| Tiempo total por pregunta | 4.1 s | **1.3 s** |

- **Empate** en datos puntuales, cálculos, cifras de distintos años, paráfrasis y
  preguntas trampa (ambos dijeron "no aparece en los documentos").
- **RAG falla cuando la respuesta está repartida.** En 2 de las 3 preguntas que
  necesitaban dos documentos, y en la que pedía juntar nombres dispersos, el
  buscador trajo solo una parte (recuperación del 50 %). En una de ellas RAG
  **rellenó el hueco inventando**: dijo que una variedad "se registró en 2022",
  cuando 2022 era el año de unas plantaciones. Es justo el riesgo de la sección 3.
- **La paradoja de la velocidad.** RAG lee 5 veces menos texto, pero tarda más
  del doble en leerlo: su prompt cambia en cada pregunta y no aprovecha la
  caché, mientras que CAG lee todo desde sus apuntes. Aun así, RAG terminó antes
  porque el modelo escribió mucho menos (196 tokens de media frente a 638,
  razonamiento incluido): con más contexto, el modelo razonó más.
- **Los jueces también se equivocan.** En la primera corrida, el juez de
  exactitud reprobó una respuesta correcta de RAG por añadir un dato extra (que
  era cierto), y el de veracidad aprobó una vez un dato inventado (sección 8).
  Ajustamos ambos y repetimos todo. Moraleja: evalúa también al evaluador.
- **Los números se mueven un poco.** En todas las corridas CAG acertó el 100 %,
  el RAG clásico entre el 81 % y el 84 %, y el tiempo de CAG fue de 2.2 a 4.1 s:
  ni el razonamiento ni los jueces son 100 % deterministas.

> ⚠️ Son 16 preguntas, un modelo y un corpus pequeño (~4,600 tokens). Sirve para
> entender, no como verdad estadística. Con bibliotecas más grandes, la ventaja
> de RAG al leer menos crece, y también los límites de CAG (sección 7).

**El idioma importa (y CAG no lo sufre):** `nomic-embed-text` se entrenó en
inglés [14]. En nuestras pruebas, con preguntas y textos en inglés el buscador
acertó siempre y con márgenes claros. En español falló 2 de 5 veces (por
ejemplo, con *"si me lastimo"* no encontró *"accidente"*). CAG no tiene ese
problema porque el modelo lee todo el texto. Por eso las preguntas del corpus
del paper están en inglés, igual que el paper.

**Sobre los documentos de prueba: citar no es lo mismo que tener permiso.**
Una cita reconoce la autoría, pero no da permiso para copiar ni redistribuir un
texto: eso lo decide su **licencia**. El paper de CAG usa CC BY-SA 4.0 [1] [16], que
permite compartirlo con atribución y con la misma licencia. Aun así, este
proyecto no lo incluye: lo descarga de arXiv cuando lo pides. El corpus de la
cooperativa es ficticio y no tiene ese problema; además, como es inventado, el
modelo no puede "saber" las respuestas de memoria.

---

## 10. Cuatro bibliotecarios: estrategias de RAG

💡 *RAG no es una sola técnica: hay muchas formas de buscar, y cada una arregla
un fallo distinto (y a veces crea otro).*

El RAG de la sección 9 es el **clásico**. En la página ⚔️ puedes elegir otras
tres estrategias, todas sobre el mismo fichero:

| Estrategia | Cómo busca | Qué arregla | Su precio |
|---|---|---|---|
| 🔎 **Clásico** | Embeddings: las *k* fichas más parecidas en significado | — (es la base) | Falla con nombres, cifras y respuestas repartidas |
| 🔀 **Híbrido** | Embeddings + **BM25**, que busca palabras exactas y da más peso a las raras [17]. Las dos listas se unen con **RRF** [18] y el modelo reordena las candidatas (*reranking*) [19] | Términos exactos ("Taza de Excelencia", "Nube Morada") | Una llamada extra al modelo (~1.4 s) |
| 👪 **Padre-hijo** | Busca fichas pequeñas (precisas), pero entrega su documento o su sección (más contexto) [20] | Datos que están en otra parte del mismo documento | Lee mucho más texto: con documentos pequeños se parece a CAG |
| 🤖 **Agéntico** | El modelo divide la pregunta en búsquedas [21], busca, revisa si le falta algo y vuelve a buscar [22] | Preguntas que piden varios datos | Varias llamadas al modelo (~1.3 s); depende de que planee bien |

**Lo que aprendimos al construirlas** (antes de escribir el código, medimos):

- **Más técnica no siempre es mejor.** El híbrido sin *reranker* encontraba los
  nombres exactos, pero perdía la pregunta parafraseada (*"si me lastimo"*):
  BM25 empujaba fuera la ficha correcta. El *reranker* lo resolvió.
- **El tamaño del "padre" importa.** Entregar la *sección* no mejoró nada aquí,
  porque nuestras fichas ya son secciones; entregar el *documento* sí. Por eso la
  estrategia es adaptativa: documento completo si es corto (≤ 6,000 caracteres),
  solo la sección si es largo.
- **La agregación sigue siendo difícil.** Ninguna resolvió *"todas las personas
  con cargo"*: el agente notó que le faltaba algo, pero su búsqueda extra no
  encontró los nombres dispersos. Ese es el terreno de **Graph RAG** [23], que
  antes de buscar extrae entidades y relaciones (persona → cargo).
- **Un modelo que razona puede quedarse dando vueltas.** Con padre-hijo (tres
  documentos completos) y la pregunta de agregación, el modelo razonó una vez
  12,800 tokens durante 82 segundos, llenó la mesa y no llegó a responder. No
  fue determinista: al repetirla respondió en ~22 s. Por eso el proyecto pone un
  **tope de escritura** (6,144 tokens) y avisa cuando se alcanza, en vez de
  mostrar una respuesta vacía sin explicación.

**La liga** (las 16 preguntas, cada estrategia contra CAG):

| 16 preguntas | 🧠 CAG | 🔎 Clásico | 🔀 Híbrido | 👪 Padre-hijo | 🤖 Agéntico |
|---|---|---|---|---|---|
| **Exactitud** | **100 %** | 84 % | 97 % | 94 % | 97 % |
| Preguntas de 2 documentos (3) | **100 %** | 33 % | **100 %** | **100 %** | **100 %** |
| Agregación (1) | **100 %** | 50 % | 50 % | 0 % | 50 % |
| Recuperación del buscador | — | 89 % | 96 % | 96 % | 96 % |
| Tokens leídos por pregunta | 4,637 | 916 | 913 | 2,969 | 1,046 |
| Tiempo por pregunta | 2.2 s | **1.3 s** | 2.6 s | 3.9 s | 2.6 s |

- **Las tres estrategias nuevas cierran casi toda la brecha** (94–97 % frente al
  84 % del clásico) y resuelven las preguntas de dos documentos. El precio es el
  tiempo: el híbrido y el agente gastan ~1.3 s extra en llamadas al modelo, y CAG
  termina antes que ellos.
- **La agregación sigue siendo de CAG.** Cuando la respuesta está repartida por
  todo el corpus, leerlo todo gana. Ninguna estrategia de RAG la resolvió.
- **Padre-hijo está a medio camino.** Lee 3 veces más que el clásico (el 64 % de
  lo que lee CAG). En las dos corridas de la liga, se quedó dando vueltas en la
  pregunta de agregación; el tope de escritura la cortó a los 40 s.
- **La veracidad no se muestra en la tabla porque engaña aquí.** CAG sacó 93 % y
  el híbrido 100 %, pero revisamos las 4 afirmaciones de CAG que el juez rechazó y
  **las 4 eran ciertas**: el juez describió o parafraseó en vez de citar. Las
  respuestas más largas le dan más ocasiones de citar mal.

---

## 11. Dónde ver cada idea en este proyecto

| Concepto | En el código | En la app |
|---|---|---|
| Precarga del conocimiento | `MotorCAG.cargar_conocimiento()` en `cag_core.py` | Panel ① *Inyectar* |
| Regla del prefijo | `MotorCAG.prompt_de_sistema()` (el libro va primero) | Panel ③ *Tiempo de lectura* |
| Tamaño de la mesa | `calcular_tamano_de_mesa()` → `num_ctx` | Panel ① *Mesa* |
| Caducidad de la caché | `keep_alive` en `parametros_base()` | Panel ① *Duración de la caché* |
| Invalidador silencioso | `romper_cache=True` | Barra lateral → 🧨 *Romper la caché* |
| Truncado silencioso | `RespuestaDelModelo.prompt_recortado` | Panel ③, alerta roja |
| Fidelidad | `evaluar_veracidad()` en `evals.py` | Panel ③ ⚖️ |
| Relevancia | `evaluar_relevancia()` en `evals.py` | Panel ③ 🎯 |
| Troceado de RAG | `trocear_documento()` en `rag_core.py` | ⚔️ CAG vs RAG → *Fichas en el índice* |
| Búsqueda por similitud | `MotorRAG.buscar()` | ⚔️ → *Las fichas que trajo el buscador* |
| Exactitud y recuperación | `evaluar_exactitud()` y `evaluar_recuperacion()` | ⚔️ → *Banco de pruebas* |
| La pelea justa | `preparar_contendientes()` en `comparacion.py` | ⚔️ → *Prepara a los rivales* |
| BM25 y fusión RRF | `IndiceBM25` y `fusion_rrf()` en `rag_core.py` | ⚔️ → 🔀 Híbrido |
| *Reranking* con el modelo | `MotorRAG._reordenar()` | ⚔️ → 🔀 Híbrido → *Lo que hizo el buscador* |
| Padre-hijo | `MotorRAG._buscar_padre_hijo()` | ⚔️ → 👪 Padre-hijo |
| Agente que planea y revisa | `MotorRAG._buscar_agentico()` | ⚔️ → 🤖 Agéntico → *paso a paso* |
| La liga de estrategias | `correr_liga()` en `comparacion.py` | ⚔️ → 🏆 *Liga de estrategias* |

---

## 12. Glosario exprés

- **Caché KV:** apuntes internos (claves y valores) que el modelo guarda de lo que ya leyó.
- **Prefill / lectura:** la fase en la que el modelo procesa el prompt. Es lo que la caché acelera.
- **Generación:** la fase en la que escribe la respuesta, token por token. La caché no la acelera.
- **`num_ctx`:** tamaño de la ventana de contexto en Ollama (la mesa).
- **`keep_alive`:** cuánto tiempo mantiene Ollama el modelo, y su caché, en memoria.
- **VRAM:** la memoria de la tarjeta gráfica, donde viven el modelo y su caché.
- **Embedding:** un texto convertido en una lista de números que representa su significado; RAG los usa para buscar.
- **Fragmento o ficha (*chunk*):** cada pedazo en que RAG corta los documentos.
- **Similitud coseno:** qué tan parecidos son dos embeddings (1 = idénticos en significado).
- **Top-k:** cuántas fichas trae el buscador de RAG en cada pregunta.
- **BM25:** buscador por palabras exactas; las palabras raras pesan más que las comunes.
- **RRF (*Reciprocal Rank Fusion*):** forma de unir varias listas ordenadas premiando lo que sale arriba en todas.
- ***Reranking*:** segunda pasada en la que el modelo lee las candidatas y elige las más útiles.
- **Agente:** un modelo que decide qué hacer (buscar, revisar, volver a buscar) antes de responder.
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
14. Nussbaum, Z., Morris, J. X., Duderstadt, B. y Mulyar, A. (2024). *Nomic Embed: Training a Reproducible Long Context Text Embedder.* Transactions on Machine Learning Research (TMLR). [arXiv:2402.01613](https://arxiv.org/abs/2402.01613)
15. Nomic AI. *Ficha del modelo nomic-embed-text-v1.5* (prefijos de tarea obligatorios): [huggingface.co/nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
16. Creative Commons. *Atribución-CompartirIgual 4.0 Internacional (CC BY-SA 4.0)*: [creativecommons.org/licenses/by-sa/4.0/deed.es](https://creativecommons.org/licenses/by-sa/4.0/deed.es)
17. Robertson, S. y Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389. DOI: [10.1561/1500000019](https://doi.org/10.1561/1500000019)
18. Cormack, G. V., Clarke, C. L. A. y Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009, pp. 758–759. DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
19. Sun, W., Yan, L., Ma, X., et al. (2023). *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.* EMNLP 2023. [arXiv:2304.09542](https://arxiv.org/abs/2304.09542)
20. Chen, T., Wang, H., Chen, S., et al. (2024). *Dense X Retrieval: What Retrieval Granularity Should We Use?* EMNLP 2024, pp. 15159–15177. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.845/)
21. Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A. y Lewis, M. (2023). *Measuring and Narrowing the Compositionality Gap in Language Models* (método *Self-Ask*). Findings of EMNLP 2023. [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)
22. Asai, A., Wu, Z., Wang, Y., Sil, A. y Hajishirzi, H. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
23. Edge, D., Trinh, H., Cheng, N., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Preprint. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
