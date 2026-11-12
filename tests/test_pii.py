"""Testes de src/data/pii.py."""
import unittest

from src.data.pii import cpf_valido, mascarar_pii


def gerar_cpf(base: str) -> str:
    """Gera um CPF válido a partir de 9 dígitos, pelo algoritmo oficial."""
    d = [int(c) for c in base]
    for n in (9, 10):
        soma = sum(x * p for x, p in zip(d, range(n + 1, 1, -1)))
        d.append((soma * 10 % 11) % 10)
    return "".join(map(str, d))


class PiiTest(unittest.TestCase):
    def test_email(self):
        self.assertEqual(mascarar_pii("escreva para joao.silva+x@mail.com.br ok"),
                         "escreva para [EMAIL] ok")

    def test_telefones(self):
        casos = [
            "(11) 98765-4321", "11 98765 4321", "11987654321", "(11) 3456-7890",
            "+55 11 98765-4321", "98765-4321", "9876-5432", "98765 4321",
            "9876 5432", "(11) 22-33-44-55", "11.98765.4321",
            "011 98765-4321", "11 9876-5432", "11 8765-4321", "(11) 9 8765-4321", "98765 - 4321",
        ]
        for c in casos:
            with self.subTest(c=c):
                self.assertEqual(mascarar_pii(f"ligue {c} agora"), "ligue [TELEFONE] agora")

    def test_cpf_valido_e_invalido(self):
        cpf = gerar_cpf("123456789")
        self.assertTrue(cpf_valido(cpf))
        fmt = f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
        self.assertEqual(mascarar_pii(f"cpf {fmt}"), "cpf [CPF]")
        self.assertEqual(mascarar_pii(f"cpf {cpf}"), "cpf [CPF]")
        errado = cpf[:-1] + str((int(cpf[-1]) + 1) % 10)
        self.assertEqual(mascarar_pii(f"cpf {errado}"), f"cpf {errado}")
        self.assertEqual(mascarar_pii("cpf 111.111.111-11"), "cpf 111.111.111-11")

    def test_onze_digitos_colados(self):
        # sem dígito verificador válido, vira telefone; com ele válido, vira CPF
        self.assertFalse(cpf_valido("11987654321"))
        self.assertEqual(mascarar_pii("11987654321"), "[TELEFONE]")
        cpf = gerar_cpf("119876543")
        self.assertEqual(mascarar_pii(cpf), "[CPF]")

    def test_pedido_nao_mascara(self):
        for s in ["pedido 12-34567890", "pedido: 11 98765 4321", "Pedidos 9876-5432",
                  "nº 12-34567890", "n° 9876 5432", "número do pedido 11 98765-4321"]:
            with self.subTest(s=s):
                self.assertEqual(mascarar_pii(s), s)

    def test_contexto_nao_e_pedido(self):
        casos = {
            "meu numero 11 98765-4321": "meu numero [TELEFONE]",
            "houve engano. 3456-7890": "houve engano. [TELEFONE]",
            "despedido 3456-7890": "despedido [TELEFONE]",
            "Aguardo retorno. (11) 98765-4321": "Aguardo retorno. [TELEFONE]",
        }
        for s, esperado in casos.items():
            with self.subTest(s=s):
                self.assertEqual(mascarar_pii(s), esperado)

    def test_pedido_com_ddd_entre_parenteses_nao_vaza_cauda(self):
        self.assertEqual(mascarar_pii("pedido (11) 98765-4321"), "pedido (11) 98765-4321")

    def test_anos_e_sequencias_longas(self):
        for s in ["2017-2018", "de 2017 2018 foi", "1234-5678-9012"]:
            with self.subTest(s=s):
                self.assertEqual(mascarar_pii(s), s)

    def test_nao_string(self):
        self.assertEqual(mascarar_pii(None), "")
        self.assertEqual(mascarar_pii(float("nan")), "")

    def test_oito_digitos_colados(self):
        self.assertEqual(mascarar_pii("codigo 12345678 aqui"), "codigo 12345678 aqui")

    def test_data_valor_cep(self):
        for s in ["entregue em 12/11/2017", "paguei R$ 1.234,56", "cep 12345-678",
                  "R$ 129,90 em 10/01/2018", "2017-11-12"]:
            with self.subTest(s=s):
                self.assertEqual(mascarar_pii(s), s)

    def test_idempotente(self):
        s = "a@b.com (11) 98765-4321 " + gerar_cpf("987654321") + " 9876-5432"
        uma = mascarar_pii(s)
        self.assertEqual(mascarar_pii(uma), uma)
        self.assertEqual(uma.count("["), 4)

    def test_vazio(self):
        self.assertEqual(mascarar_pii(""), "")


if __name__ == "__main__":
    unittest.main()
