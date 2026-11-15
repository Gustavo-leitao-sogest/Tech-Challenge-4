"""Testes do grafo sem rede: rotas puras e fluxo com nós falsos."""
import unittest

from src import config
from src.rag.graph import (answer, build_graph, mermaid, rota_escopo,
                           rota_panorama, rota_suficiencia)

EVID = [{"n": 1, "review_id": "a", "score": 0.9, "nota": 1, "data": "2018-01-01",
         "categoria": "x", "atraso": None, "texto": "ruim"}]


def _nos(tipo="especifica", em_escopo=True, suficiente=True, evidencias=EVID, chamadas=None):
    ch = chamadas if chamadas is not None else []

    def analisar(s):
        ch.append("analisar")
        out = {"analise": {"tipo": tipo}, "consulta": "c0", "filtros": {"nota_max": 2}, "tentativas": 0}
        if not em_escopo:
            out.update(status="fora_de_escopo", resposta="fora", citacoes=[])
        return out

    def rec(s):
        ch.append("recuperar")
        return {"evidencias": evidencias}

    def pan(s):
        ch.append("panorama")
        return {"evidencias": evidencias, "contagens": None, "panorama_amostra": len(evidencias)}

    def reesc(s):
        ch.append("reescrever")
        return {"consulta": "hyde", "tentativas": s.get("tentativas", 0) + 1}

    def gerar(s):
        ch.append("gerar")
        return {"status": "ok", "resposta": "r [1]", "citacoes": [1]}

    def recusar(s):
        ch.append("recusar")
        return {"status": "insuficiente", "resposta": "nao", "citacoes": []}

    return {
        "analisar": analisar, "recuperar": rec, "recuperar_panorama": pan,
        "avaliar": lambda s: {"suficiente": suficiente},
        "reescrever": reesc, "gerar": gerar, "recusar": recusar,
    }


class TestRotas(unittest.TestCase):
    def test_escopo(self):
        self.assertEqual(rota_escopo({"status": "fora_de_escopo"}), "fim")
        self.assertEqual(rota_escopo({"analise": {"tipo": "panoramica"}}), "panoramica")
        self.assertEqual(rota_escopo({"analise": {"tipo": "especifica"}}), "especifica")

    def test_panorama(self):
        self.assertEqual(rota_panorama({"evidencias": EVID}), "gerar")
        self.assertEqual(rota_panorama({"evidencias": []}), "recusar")

    def test_suficiencia(self):
        self.assertEqual(rota_suficiencia({"suficiente": True}), "gerar")
        self.assertEqual(rota_suficiencia({"suficiente": False, "tentativas": 0}), "reescrever")
        self.assertEqual(
            rota_suficiencia({"suficiente": False, "tentativas": config.MAX_RETRIES}), "recusar")


class TestGrafo(unittest.TestCase):
    def _rodar(self, **kw):
        ch: list[str] = []
        g = build_graph(_nos(chamadas=ch, **kw))
        return answer("p", grafo=g), ch

    def test_fora_de_escopo(self):
        r, ch = self._rodar(em_escopo=False)
        self.assertEqual(r.status, "fora_de_escopo")
        self.assertEqual(ch, ["analisar"])

    def test_especifica_suficiente(self):
        r, ch = self._rodar()
        self.assertEqual(r.status, "ok")
        self.assertEqual(ch, ["analisar", "recuperar", "gerar"])
        self.assertEqual(r.citacoes, [1])
        self.assertEqual(r.evidencias[0].review_id, "a")
        self.assertEqual(r.filtros, {"nota_max": 2})

    def test_insuficiente_retry_e_recusa(self):
        r, ch = self._rodar(suficiente=False)
        self.assertEqual(ch, ["analisar", "recuperar", "reescrever", "recuperar", "recusar"])
        self.assertEqual(r.status, "insuficiente")
        self.assertEqual(r.consulta_transformada, "hyde")

    def test_panoramica(self):
        r, ch = self._rodar(tipo="panoramica", suficiente=False)  # limiar não vale na panorâmica
        self.assertEqual(ch, ["analisar", "panorama", "gerar"])
        self.assertEqual(r.filtros["panorama_amostra"], 1)

    def test_panoramica_sem_evidencias(self):
        r, ch = self._rodar(tipo="panoramica", evidencias=[])
        self.assertEqual(ch, ["analisar", "panorama", "recusar"])
        self.assertEqual(r.status, "insuficiente")

    def test_mermaid(self):
        self.assertIn("analisar", mermaid())


if __name__ == "__main__":
    unittest.main()
