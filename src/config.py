"""Configuração central: caminhos, parâmetros e variáveis de ambiente."""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INDEX = ROOT / "data" / "index"
REVIEWS_PARQUET = DATA_PROCESSED / "reviews.parquet"
# No Windows o Chroma exige caminho ASCII (ver chroma_path); pode ser trocado via .env.
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or DATA_INDEX / "chroma")
BM25_PATH = DATA_INDEX / "bm25.pkl"
CHROMA_COLLECTION = "olist_reviews"

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3     # novas tentativas do cliente em erros transitórios (ex.: 503)
LLM_TIMEOUT = 60        # segundos por chamada ao LLM

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_BATCH_SIZE = 64
CHROMA_INSERT_BATCH = 5000

# Recuperação
TOP_K_DENSE = 50       # candidatos da busca densa
TOP_K_BM25 = 50        # candidatos do BM25
RRF_K = 60             # constante do Reciprocal Rank Fusion (IA Gen Aula 2)
TOP_K_CONTEXT = 15     # evidências enviadas ao LLM
RETRIEVAL_MODE = "hibrida"  # "densa" | "hibrida" (o eval compara os dois)
PANORAMA_FETCH_K = 80  # candidatos da busca densa antes do MMR (rota panorâmica)
PANORAMA_K = 25        # evidências diversas devolvidas pela rota panorâmica
MMR_LAMBDA = 0.5       # 1 = só relevância, 0 = só diversidade
# Limites do período da base (Olist), para expandir intervalos de ano_mes abertos
ANO_MES_LIMITE_INF = "2016-01"
ANO_MES_LIMITE_SUP = "2018-12"

# Guardrails (calibrar no eval; NÃO chutar)
MIN_SCORE = 0.90       # provisório: separa 0,81–0,89 (sem evidência) de 0,93–0,96 (boas) em amostra inicial; calibrar no eval
MIN_EVIDENCES = 3      # nº mínimo de evidências acima do limiar
MAX_RETRIES = 1        # novas tentativas com HyDE antes de recusar

# Preparação
MIN_TEXT_ALNUM = 3     # mínimo de caracteres alfanuméricos; "..." e "ok" saem (decidido na EDA)
DESCONHECIDO = "desconhecido"  # valor para metadado ausente (Chroma não aceita None)

# Texto / BM25
BM25_USE_STEMMING = False  # stemming RSLP nos tokens do BM25; o eval decide se liga


def chroma_path(caminho: Path | str | None = None) -> str:
    """Caminho do Chroma pronto para uso (todo código que abrir o índice deve usá-lo).

    No Windows, o backend em Rust não grava/lê os arquivos do HNSW em pastas com
    caracteres não-ASCII. Converte para o nome curto 8.3; se ainda sobrar não-ASCII,
    levanta erro orientando a definir CHROMA_DIR em uma pasta só ASCII.
    """
    caminho = Path(caminho if caminho is not None else CHROMA_DIR)
    caminho.mkdir(parents=True, exist_ok=True)
    texto = str(caminho.resolve())
    if sys.platform != "win32" or texto.isascii():
        return texto
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(texto, buf, 1024) and buf.value.isascii():
        return buf.value
    raise RuntimeError(
        f"O Chroma no Windows exige caminho ASCII e o nome curto 8.3 não está disponível "
        f"para '{texto}'. Defina CHROMA_DIR no .env apontando para uma pasta só ASCII "
        r"(ex.: C:\chroma_olist)."
    )
