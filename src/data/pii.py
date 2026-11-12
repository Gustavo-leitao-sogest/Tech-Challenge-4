"""Mascaramento de PII (LGPD — minimização). Melhor esforço, orientado por regex."""
import re

# DDD brasileiro: dois dígitos (zero de tronco opcional, ex.: 011).
_DDD = r"0?[1-9]\d"
# Separador: espaço, hífen ou ponto, com espaços opcionais ao redor do hífen/ponto.
_SEP = r"(?:\s*[.-]\s*|\s+)"
_SEPO = rf"(?:{_SEP})?"
_PREFIXO_PAIS = rf"(?:\+?55{_SEPO})?"
_MOVEL = rf"9[\s.]?\d{{4}}{_SEPO}\d{{4}}"          # 9XXXX-XXXX ou 9 XXXX-XXXX
_FIXO = rf"[2-9]\d{{3}}{_SEPO}\d{{4}}"              # XXXX-XXXX (também celulares antigos)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Telefone com DDD. Todas as formas ancoradas em dígitos para não pegar trechos
# de números maiores (pedido, protocolo etc.); o DDD sempre entra no trecho mascarado.
_TEL_DDD = re.compile(
    r"(?<!\d)(?:"
    # (DD) com parênteses: móvel, fixo (colado ou não) ou XX-XX-XX-XX
    rf"{_PREFIXO_PAIS}\(\s*{_DDD}\s*\)\s*(?:{_MOVEL}|{_FIXO}|\d{{2}}{_SEP}\d{{2}}{_SEP}\d{{2}}{_SEP}\d{{2}})"
    # celular sem parênteses; aceita colado (11 dígitos)
    rf"|{_PREFIXO_PAIS}{_DDD}{_SEPO}{_MOVEL}"
    # fixo sem parênteses: exige separador após o DDD (10 dígitos colados são ambíguos)
    rf"|{_PREFIXO_PAIS}{_DDD}{_SEP}{_FIXO}"
    r")(?!\d)"
)

# Sem DDD: exige separador no meio. 8 dígitos colados não entram (falso positivo da EDA).
# Lookbehinds: não remascarar a cauda de um número com DDD poupado (pedido).
# Lookaheads: pares de anos (2017-2018) e sequências maiores (1234-5678-9012) não são telefone.
# CEP (5-3) não casa porque o bloco final precisa de 4 dígitos.
_TEL_LOCAL = re.compile(
    r"(?<!\d)(?<!\d[\s.-])(?<!\))(?<!\)\s)"
    r"(?!(?:19|20)\d{2}[\s.-](?:19|20)\d{2}(?!\d))"
    rf"(?:9\d{{4}}{_SEP}\d{{4}}|\d{{4}}{_SEP}\d{{4}})"
    r"(?!\d)(?![\s.-]?\d)"
)

_CPF = re.compile(r"(?<!\d)(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})(?!\d)")

# Número de pedido nunca é telefone: olha o contexto imediatamente anterior.
# Ancorado no início da palavra para não casar "despedido" nem o "no." de "engano.".
_CTX_PEDIDO = re.compile(
    r"(?<![a-zà-ú])(?:pedidos?|ped\.|n[º°]|n[uú]mero\s+do\s+pedido)\s*[:#-]?\s*$",
    re.IGNORECASE,
)


def _apos_pedido(texto: str, inicio: int) -> bool:
    """Indica se o trecho vem logo depois de 'pedido', 'nº' etc."""
    return bool(_CTX_PEDIDO.search(texto[max(0, inicio - 30):inicio]))


def _substituir_tel(m: re.Match) -> str:
    """Troca o telefone por [TELEFONE], exceto se for número de pedido."""
    return m.group(0) if _apos_pedido(m.string, m.start()) else "[TELEFONE]"


def cpf_valido(digitos: str) -> bool:
    """Valida CPF (11 dígitos) pelos dois dígitos verificadores."""
    if len(digitos) != 11 or not digitos.isdigit() or len(set(digitos)) == 1:
        return False
    for n in (9, 10):
        soma = sum(int(d) * p for d, p in zip(digitos[:n], range(n + 1, 1, -1)))
        if int(digitos[n]) != (soma * 10 % 11) % 10:
            return False
    return True


def _substituir_cpf(m: re.Match) -> str:
    so_digitos = re.sub(r"\D", "", m.group(0))
    return "[CPF]" if cpf_valido(so_digitos) else m.group(0)


def mascarar_pii(texto: str) -> str:
    """Substitui e-mail por [EMAIL], telefone BR por [TELEFONE] e CPF por [CPF]."""
    if not isinstance(texto, str):  # None, NaN
        return ""
    # CPF válido antes do telefone: 11 dígitos colados com dígito verificador correto
    # são CPF; os que não validam ainda podem ser celular com DDD. Local por último.
    texto = _EMAIL.sub("[EMAIL]", texto)
    texto = _CPF.sub(_substituir_cpf, texto)
    texto = _TEL_DDD.sub(_substituir_tel, texto)
    return _TEL_LOCAL.sub(_substituir_tel, texto)
