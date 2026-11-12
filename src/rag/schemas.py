"""Contratos de dados do pipeline (saídas estruturadas e resultado final)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["ok", "insuficiente", "fora_de_escopo"]


class AnalisePergunta(BaseModel):
    """Saída estruturada da análise da pergunta."""

    em_escopo: bool = Field(description="A pergunta é sobre a experiência de compra relatada em avaliações?")
    tipo: Literal["especifica", "panoramica"] = Field(
        description="'panoramica' se pede principais/mais frequentes/padrões/visão geral"
    )
    consulta_reescrita: str = Field(description="Reformulação em linguagem de cliente, para a busca")
    nota_min: Optional[int] = Field(default=None, ge=1, le=5)
    nota_max: Optional[int] = Field(default=None, ge=1, le=5)
    categoria: Optional[str] = None
    apenas_atrasados: Optional[bool] = None


class RespostaLLM(BaseModel):
    """Saída estruturada da geração."""

    resposta: str = Field(description="Texto com citações [n] após cada afirmação")
    citacoes: list[int] = Field(description="Índices das evidências efetivamente usadas")
    suficiente: bool = Field(description="False se as evidências não sustentam uma resposta")


class Evidencia(BaseModel):
    n: int
    review_id: str
    score: float
    nota: int
    data: str
    categoria: str
    atraso: bool | None
    texto: str


class RagResult(BaseModel):
    """Resultado final devolvido à UI e ao eval."""

    pergunta: str
    status: Status
    resposta: str
    citacoes: list[int] = []
    evidencias: list[Evidencia] = []
    consulta_transformada: str | None = None
    filtros: dict = {}
    contagens: dict | None = None  # rota panorâmica: frequências calculadas no código
