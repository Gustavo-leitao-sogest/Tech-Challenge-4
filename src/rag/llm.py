"""Fábrica de LLM por provedor.

Importa só o pacote do provedor escolhido em LLM_PROVIDER. Modelo sempre vindo do .env.
"""


def get_llm(temperature: float = 0.0):
    raise NotImplementedError
