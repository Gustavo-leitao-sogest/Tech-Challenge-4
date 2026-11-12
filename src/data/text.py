"""Limpezas de texto.

- limpar_leve: para embedding, LLM e exibição. Mantém acentos, pontuação e negações.
- limpar_bm25: para o índice lexical. Lowercase, sem acentos, abreviações expandidas,
  stopwords removidas EXCETO negações. Aplicar a mesma função à pergunta.
"""
import html
import re
import unicodedata
from functools import lru_cache

import nltk
from nltk.stem import RSLPStemmer

from src import config

NEGACOES = {"não", "nao", "nem", "nunca", "jamais", "nenhum", "nenhuma", "sem"}

# Abreviações comuns de avaliações; expandidas ANTES das stopwords (ex.: "vc" -> "voce").
# Valores já sem acento, pois a expansão roda depois da normalização.
_ABREVIACOES = {
    "vc": ("voce",), "vcs": ("voces",), "pq": ("por", "que"), "tb": ("tambem",),
    "tbm": ("tambem",), "q": ("que",), "n": ("nao",), "naum": ("nao",),
    "msm": ("mesmo",), "qdo": ("quando",), "td": ("tudo",), "mt": ("muito",),
    "mto": ("muito",), "blz": ("beleza",),
}

_RE_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
# Não engole pontuação final: "(http://x.com)." deixa ")." no texto.
_RE_URL = re.compile(r"(?:https?://|www\.)\S*[^\s).,;]", re.IGNORECASE)
_RE_ESPACOS = re.compile(r"\s+")
# "p/" -> para, "s/" -> sem (mantém a negação); precisa vir antes de a pontuação sumir.
_RE_P_BARRA = re.compile(r"(?<!\w)p/")
_RE_S_BARRA = re.compile(r"(?<!\w)s/")
# "n°", "nº", "n." antes de número/"do pedido" é indicador numérico, não a negação "n".
_RE_N_NUMERO = re.compile(r"(?<!\w)n\s?[°º.](?=\s*(?:\d|do\b))")
_RE_TOKEN = re.compile(r"[a-z0-9]+")


def _sem_acento(s: str) -> str:
    """Remove diacríticos (NFKD e descarta marcas combinantes)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def limpar_leve(texto: str) -> str:
    """Remove HTML e URLs, normaliza unicode (NFC) e colapsa espaços.

    E-mails ficam intactos: quem os trata é o mascaramento de PII.
    """
    if not isinstance(texto, str):  # None, NaN
        return ""
    # Entidades aninhadas (&amp;lt;) pedem mais de uma passada para estabilizar.
    for _ in range(3):
        novo = html.unescape(texto)
        if novo == texto:
            break
        texto = novo
    texto = _RE_URL.sub(" ", _RE_HTML.sub(" ", texto))
    # NFC por último: o unescape pode gerar combinantes (a&#771;).
    return unicodedata.normalize("NFC", _RE_ESPACOS.sub(" ", texto).strip())


def _garantir_recurso(caminho: str, nome: str) -> None:
    """Baixa o recurso NLTK em silêncio se ainda não existir."""
    try:
        nltk.data.find(caminho)
    except LookupError:
        if not nltk.download(nome, quiet=True):
            raise RuntimeError(
                f"Recurso NLTK '{nome}' ausente e o download falhou (offline?). "
                f"Rode nltk.download('{nome}') com internet."
            )


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """Stopwords PT do NLTK sem acento, exceto negações (que mudam o sentido)."""
    _garantir_recurso("corpora/stopwords", "stopwords")
    from nltk.corpus import stopwords
    negacoes = {_sem_acento(n) for n in NEGACOES}
    return frozenset(_sem_acento(w) for w in stopwords.words("portuguese")) - negacoes


@lru_cache(maxsize=1)
def _stemmer() -> RSLPStemmer:
    """Stemmer RSLP (criado uma única vez)."""
    _garantir_recurso("stemmers/rslp", "rslp")
    return RSLPStemmer()


@lru_cache(maxsize=200_000)
def _radical(token: str) -> str:
    """Radical RSLP com cache; tokens numéricos ficam como estão."""
    return token if token.isdigit() else _stemmer().stem(token)


def limpar_bm25(texto: str) -> list[str]:
    """Retorna tokens normalizados para o BM25."""
    t = limpar_leve(texto).lower()
    # "p/" tem barra: precisa ser tratado antes de a pontuação sumir.
    t = _RE_N_NUMERO.sub(" numero ", t)
    t = _RE_S_BARRA.sub(" sem ", _RE_P_BARRA.sub(" para ", t))
    stop = _stopwords()
    tokens: list[str] = []
    for tok in _RE_TOKEN.findall(_sem_acento(t)):
        for sub in _ABREVIACOES.get(tok, (tok,)):
            if sub in stop or (len(sub) == 1 and not sub.isdigit()):
                continue
            tokens.append(sub)
    if config.BM25_USE_STEMMING:
        tokens = [_radical(tok) for tok in tokens]
    return tokens
