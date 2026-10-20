# Corpus: paper de CAG

Este corpus usa el artículo que propuso CAG:

> Chan, B. J., Chen, C.-T., Cheng, J.-H. y Huang, H.-H. (2025). *Don't Do RAG:
> When Cache-Augmented Generation is All You Need for Knowledge Tasks.* The ACM
> Web Conference 2025 (WWW '25), artículo corto. https://arxiv.org/abs/2412.15605

**Licencia del artículo:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Permite compartirlo y adaptarlo **con atribución** y **bajo la misma licencia**.

**Por qué no está en el repositorio:** citar un trabajo no da permiso para
redistribuirlo; lo que manda es su licencia. Esta lo permitiría, pero para mantener
el repositorio ligero y con una sola licencia (MIT), el texto se descarga de arXiv
cuando lo pides:

- en la app: página **⚔️ CAG vs RAG** → *Descargar el paper desde arXiv*;
- en la terminal: `python comparacion.py paper_cag --descargar`.

El archivo descargado (`paper_cag.md`) incluye la atribución y conserva la licencia
CC BY-SA 4.0 del original. Está excluido de Git en `.gitignore`.

`preguntas.json` contiene preguntas y respuestas breves escritas para este proyecto
(paráfrasis, no copias del texto).
