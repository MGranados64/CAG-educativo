"""
pruebas/test_sin_modelo.py — Pruebas automáticas que NO necesitan Ollama
=========================================================================

Comprueban las piezas "mecánicas" del proyecto (trocear, ordenar, calcular,
convertir) en menos de un segundo. Si cambias algo y una prueba falla, sabrás
exactamente qué se rompió antes de gastar minutos con el modelo.

Ejecuta desde la carpeta del proyecto:

    python -m unittest discover -s pruebas -t . -v
"""

import json
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cag_core import (
    RespuestaEnVivo,
    Documento,
    MedicionDeTiempos,
    calcular_tamano_de_mesa,
    cargar_documentos_de_carpeta,
    construir_conocimiento_en_cache,
)
from comparacion import CORPUS, _ExtractorDeTexto
from evals import PUNTAJE_DE_EXACTITUD, PUNTAJE_DE_RECUPERACION, cita_verificada, semaforo
from rag_core import (
    IndiceBM25,
    MotorRAG,
    fusion_rrf,
    palabras_para_buscar,
    partir_texto_largo,
    prefijos_para,
    trocear_documento,
)


class PruebasDelConocimientoEnCache(unittest.TestCase):
    def test_el_orden_de_los_documentos_no_cambia_el_prefijo(self):
        # Si el texto cambiara con el orden, la caché nunca se reutilizaría.
        a, b = Documento("a.md", "Texto A"), Documento("b.md", "Texto B")
        self.assertEqual(construir_conocimiento_en_cache([a, b]), construir_conocimiento_en_cache([b, a]))

    def test_la_mesa_tiene_margen_y_es_multiplo_de_4096(self):
        for caracteres in (0, 100, 20_000, 150_000):
            mesa = calcular_tamano_de_mesa(caracteres)
            self.assertEqual(mesa % 4096, 0)
            self.assertGreaterEqual(mesa, 8192)
            self.assertGreater(mesa, caracteres / 2.5)  # cabe incluso en el peor caso


class PruebasDeLaCarpetaDeDocumentos(unittest.TestCase):
    def test_ignora_preguntas_y_archivos_con_guion_bajo(self):
        with tempfile.TemporaryDirectory() as carpeta:
            Path(carpeta, "manual.md").write_text("conocimiento", encoding="utf-8")
            Path(carpeta, "_LEEME.md").write_text("nota de licencia", encoding="utf-8")
            Path(carpeta, "preguntas.json").write_text("{}", encoding="utf-8")
            nombres = [documento.nombre for documento in cargar_documentos_de_carpeta(Path(carpeta))]
        self.assertEqual(nombres, ["manual.md"])

    def test_los_corpus_tienen_preguntas_bien_formadas(self):
        campos = {"id", "tipo", "pregunta", "respuesta_esperada", "en_documentos", "documentos_fuente"}
        for corpus in CORPUS.values():
            preguntas = corpus.preguntas()["preguntas"]
            self.assertTrue(preguntas, corpus.clave)
            self.assertEqual(len({p["id"] for p in preguntas}), len(preguntas), "ids repetidos")
            for pregunta in preguntas:
                self.assertTrue(campos <= set(pregunta), pregunta["id"])

    def test_las_fuentes_de_la_cooperativa_existen(self):
        corpus = CORPUS["cooperativa"]
        nombres = {documento.nombre for documento in corpus.documentos()}
        for pregunta in corpus.preguntas()["preguntas"]:
            self.assertTrue(set(pregunta["documentos_fuente"]) <= nombres, pregunta["id"])


class PruebasDelTroceado(unittest.TestCase):
    TEXTO = (
        "# Manual\n\nIntroducción breve.\n\n"
        "## Vacaciones\n\nEl primer año son 14 días.\n\nCada año se suman 2 días.\n\n"
        "## Seguridad\n\nUsa casco en el beneficio seco."
    )

    def test_nunca_mezcla_secciones(self):
        fichas = trocear_documento(Documento("manual.md", self.TEXTO), tamano=1_000)
        self.assertEqual([f.seccion for f in fichas], ["Manual", "Vacaciones", "Seguridad"])
        self.assertIn("14 días", fichas[1].texto)
        self.assertNotIn("casco", fichas[1].texto)

    def test_descarta_las_fichas_que_solo_tienen_un_titulo(self):
        texto = "## 3. Acuerdos\n\n### Acuerdo 1\n\nSe aprueba el precio."
        fichas = trocear_documento(Documento("acta.md", texto))
        self.assertEqual(len(fichas), 1)
        self.assertEqual(fichas[0].seccion, "Acuerdo 1")

    def test_la_ficha_lleva_su_contexto(self):
        ficha = trocear_documento(Documento("manual.md", self.TEXTO))[1]
        self.assertTrue(ficha.texto_con_contexto.startswith("[manual.md › Vacaciones]"))

    def test_un_parrafo_largo_se_parte_sin_perder_texto(self):
        palabras = " ".join(f"palabra{i}" for i in range(400))
        trozos = partir_texto_largo(palabras, tamano=300, solapamiento=50)
        self.assertGreater(len(trozos), 1)
        self.assertTrue(all(len(trozo) <= 300 for trozo in trozos))
        # Todas las palabras aparecen en algún trozo (nada se pierde al cortar).
        juntas = " ".join(trozos)
        self.assertTrue(all(f"palabra{i}" in juntas for i in range(400)))

    def test_prefijos_solo_para_nomic(self):
        self.assertEqual(prefijos_para("nomic-embed-text:latest"), ("search_document: ", "search_query: "))
        self.assertEqual(prefijos_para("otro-modelo"), ("", ""))


class PruebasDeMedicionesYPuntajes(unittest.TestCase):
    def test_la_medicion_convierte_nanosegundos(self):
        respuesta = SimpleNamespace(
            prompt_eval_count=1000,
            prompt_eval_duration=500_000_000,
            eval_count=50,
            eval_duration=1_000_000_000,
            load_duration=0,
        )
        medicion = MedicionDeTiempos.desde_respuesta(respuesta)
        self.assertAlmostEqual(medicion.segundos_leyendo, 0.5)
        self.assertAlmostEqual(medicion.velocidad_de_lectura, 2000)
        self.assertAlmostEqual(medicion.segundos_sin_cache(velocidad_en_frio=1000), 1.0)
        self.assertFalse(medicion.modelo_recien_cargado)

    DOCUMENTO = (
        "## Vacaciones\n\n**Vacaciones:** el primer año corresponden **14 días hábiles**.\n\n"
        "| Calidad | 2024 | 2025 |\n|---|---|---|\n| Madura de primera | $84 MXN/kg | $92 MXN/kg |\n\n"
        "Se exportó a:\n\n- **Japón:** 95 toneladas.\n- **Chile:** 35 toneladas.\n\n"
        "Registró su variedad llamada **\"Nube Morada\"** en 2011."
    )

    def test_acepta_citas_reales_aunque_cambie_el_formato(self):
        casos = [
            ("Tiene 14 días hábiles el primer año", '"el primer año corresponden 14 días hábiles."'),
            ("Exportó 95 toneladas a Japón", "Se exportó a: Japón: 95 toneladas. Chile: 35 toneladas."),
            ("En 2024 se pagó $84 por kilo", "Cereza madura de primera | $84 MXN/kg | $92 MXN/kg"),
            ("Se registró en 2011", "registró su variedad llamada 'Nube Morada' en 2011"),
        ]
        for afirmacion, cita in casos:
            self.assertTrue(cita_verificada(cita, self.DOCUMENTO, afirmacion), cita)

    def test_rechaza_citas_inventadas_o_que_no_vienen_al_caso(self):
        casos = [
            ("Fermenta 72 horas", "la fermentación dura 72 horas en tanques de acero"),  # no existe
            ("Exportó 200 toneladas a Japón", "Se exportó a: Japón: 200 toneladas."),  # cifra inventada
            ("La capital de Australia es Canberra", "el primer año corresponden 14 días hábiles"),  # no viene al caso
            ("La capital de Australia es Canberra", '<documento nombre="m.md">'),  # una etiqueta no es prueba
            ("Algo", ""),
        ]
        for afirmacion, cita in casos:
            self.assertFalse(cita_verificada(cita, self.DOCUMENTO, afirmacion), cita)

    def test_puntajes_y_semaforo(self):
        self.assertEqual(PUNTAJE_DE_EXACTITUD, {"correcta": 1.0, "parcial": 0.5, "incorrecta": 0.0})
        self.assertEqual(PUNTAJE_DE_RECUPERACION["nada"], 0.0)
        self.assertTrue(semaforo(0.95).startswith("🟢"))
        self.assertTrue(semaforo(0.7).startswith("🟡"))
        self.assertTrue(semaforo(0.2).startswith("🔴"))


class PruebasDelExtractorHTML(unittest.TestCase):
    def test_conserva_titulos_y_descarta_bibliografia(self):
        html = """
        <html><head><script>no()</script></head><body><article>
          <h2 class="ltx_title">1 Introduction</h2>
          <div class="ltx_para"><p>CAG preloads <math alttext="k=4"><mi>k</mi></math> documents.</p></div>
          <ul><li>first</li><li>second</li></ul>
          <section class="ltx_bibliography"><p>[1] A reference.</p></section>
        </article></body></html>
        """
        extractor = _ExtractorDeTexto()
        extractor.feed(html)
        texto = extractor.texto()
        self.assertIn("## 1 Introduction", texto)
        self.assertIn("CAG preloads k=4 documents.", texto)
        self.assertIn("- first", texto)
        self.assertNotIn("A reference", texto)
        self.assertNotIn("no()", texto)


class PruebasDeLaBusquedaPorPalabras(unittest.TestCase):
    def test_quita_acentos_mayusculas_y_palabras_vacias(self):
        self.assertEqual(palabras_para_buscar("¿Cuántas BECAS de la Cooperativa?"), ["becas", "cooperativa"])

    def test_bm25_premia_la_palabra_rara(self):
        indice = IndiceBM25(["café café café", "café y glifosato", "café de altura"])
        puntajes = indice.puntajes("glifosato café")
        self.assertEqual(int(np.argmax(puntajes)), 1)  # "glifosato" solo está en una ficha

    def test_rrf_premia_estar_arriba_en_las_dos_listas(self):
        # La ficha 2 es segunda en ambas listas; la 1 es primera en una y no aparece en la otra.
        self.assertEqual(fusion_rrf([[1, 2, 3], [4, 2, 5]])[0], 2)


class ClienteSimulado:
    """
    Imita a Ollama sin modelo: embeddings deterministas (por palabras) y
    respuestas JSON programables, para probar las estrategias de RAG.
    """

    def __init__(self, respuestas_json: list[str] | None = None):
        self.respuestas_json = list(respuestas_json or [])
        self.llamadas_al_chat = 0

    def show(self, nombre):
        return SimpleNamespace(
            details=SimpleNamespace(family="gptoss", parameter_size="20B", quantization_level="Q4"),
            modelinfo={"general.architecture": "gptoss", "gptoss.context_length": 131072},
            capabilities=["completion", "thinking"],
        )

    def embed(self, model, input, keep_alive=None):
        vectores = []
        for texto in input:
            vector = np.zeros(64)
            for palabra in palabras_para_buscar(texto):
                vector[zlib.crc32(palabra.encode()) % 64] += 1
            vectores.append(vector + 1e-6)
        return SimpleNamespace(embeddings=vectores)

    def chat(self, **parametros):
        self.llamadas_al_chat += 1
        contenido = self.respuestas_json.pop(0) if self.respuestas_json else "{}"
        return SimpleNamespace(message=SimpleNamespace(content=contenido))


class PruebasDeLasEstrategiasDeRAG(unittest.TestCase):
    DOCUMENTOS = [
        Documento("corto.md", "# Corto\n\n## Fundación\n\nSe fundó en 1987.\n\n## Variedad\n\nLa variedad Nube Morada se registró en 2011."),
        Documento("largo.md", "# Largo\n\n## Producción\n\n" + "Se produjeron 83 toneladas de café. " * 200
                  + "\n\n## Tienda\n\nLa tienda cierra los martes."),
    ]

    def motor(self, respuestas_json=None, estrategia="clasico"):
        motor = MotorRAG(cliente=ClienteSimulado(respuestas_json), estrategia=estrategia, fragmentos_a_recuperar=2)
        motor.indexar(self.DOCUMENTOS)
        return motor

    def test_padre_hijo_entrega_documento_si_es_corto_y_seccion_si_es_largo(self):
        motor = self.motor(estrategia="padre_hijo")
        padres = motor.buscar("¿Cuántas toneladas de café se produjeron?").recuperados
        por_documento = {p.fragmento.documento: p.fragmento for p in padres}
        self.assertEqual(por_documento["largo.md"].seccion, "Producción")  # documento largo: solo su sección
        self.assertNotIn("martes", por_documento["largo.md"].texto)
        if "corto.md" in por_documento:
            self.assertEqual(por_documento["corto.md"].seccion, "documento completo")

    def test_pedir_json_reintenta_si_no_cumple_el_esquema(self):
        motor = self.motor(['{"fichas": [0]}', '{"elegidas": [1, 0]}'])
        elegidas = motor._reordenar("pregunta", [5, 6, 7], cuantas=2)
        self.assertEqual(elegidas, [6, 5])  # usó la segunda respuesta (la que sí cumple)
        self.assertEqual(motor.cliente.llamadas_al_chat, 2)

    def test_reordenar_completa_si_el_modelo_elige_pocas(self):
        motor = self.motor(['{"elegidas": [2]}'])
        self.assertEqual(motor._reordenar("pregunta", [5, 6, 7], cuantas=3), [7, 5, 6])

    def test_reordenar_sobrevive_si_el_modelo_falla_siempre(self):
        motor = self.motor(["no es json", "tampoco"])
        self.assertEqual(motor._reordenar("pregunta", [5, 6, 7], cuantas=2), [5, 6])

    def test_el_agente_planea_busca_revisa_y_no_repite_fichas(self):
        motor = self.motor(
            [
                json.dumps({"busquedas": ["año registro Nube Morada", "registro variedad Nube Morada"]}),
                json.dumps({"falta": "", "suficiente": True, "nueva_busqueda": ""}),
            ],
            estrategia="agentico",
        )
        busqueda = motor.buscar("¿En qué año se registró la variedad Nube Morada?")
        numeros = [r.fragmento.numero for r in busqueda.recuperados]
        self.assertEqual(len(numeros), len(set(numeros)))  # sin fichas repetidas
        self.assertTrue(any("2011" in r.fragmento.texto for r in busqueda.recuperados))
        self.assertTrue(busqueda.pasos[0].startswith("🧭"))
        self.assertTrue(any(paso.startswith("🤔") for paso in busqueda.pasos))

    def test_cambiar_de_estrategia_no_cambia_la_mesa(self):
        motor = self.motor()
        mesa = motor.num_ctx
        for estrategia in ("hibrido", "padre_hijo", "agentico"):
            motor.estrategia = estrategia
            self.assertEqual(motor.num_ctx, mesa)  # otra mesa obligaría a Ollama a recargar el modelo



class PruebasDeLaRespuestaEnVivo(unittest.TestCase):
    @staticmethod
    def fragmento(pensamiento="", texto="", fin=None, generados=None):
        return SimpleNamespace(
            message=SimpleNamespace(thinking=pensamiento, content=texto),
            done_reason=fin,
            prompt_eval_count=100 if fin else None,
            prompt_eval_duration=1_000_000 if fin else None,
            eval_count=generados,
            eval_duration=1_000_000_000 if fin else None,
            load_duration=0,
        )

    def respuesta(self, fragmentos):
        cliente = SimpleNamespace(chat=lambda **parametros: iter(fragmentos))
        en_vivo = RespuestaEnVivo(cliente, {"model": "modelo-de-prueba"}, "¿pregunta?")
        texto = "".join(en_vivo)
        return texto, en_vivo.resultado

    def test_marca_cuando_el_modelo_se_queda_razonando_sin_responder(self):
        texto, resultado = self.respuesta(
            [self.fragmento(pensamiento="mmm… "), self.fragmento(pensamiento="mmm… ", fin="length", generados=6144)]
        )
        self.assertEqual(texto, "")
        self.assertTrue(resultado.se_quedo_sin_espacio)
        self.assertEqual(resultado.razonamiento, "mmm… mmm… ")

    def test_una_respuesta_normal_no_se_marca(self):
        texto, resultado = self.respuesta(
            [self.fragmento(pensamiento="pienso"), self.fragmento(texto="Hola"), self.fragmento(fin="stop", generados=5)]
        )
        self.assertEqual(texto, "Hola")
        self.assertFalse(resultado.se_quedo_sin_espacio)


if __name__ == "__main__":
    unittest.main()
