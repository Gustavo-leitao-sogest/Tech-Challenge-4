"""Configuração central: caminhos, parâmetros e variáveis de ambiente."""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# Caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INDEX = ROOT / "data" / "index"
REVIEWS_PARQUET = DATA_PROCESSED / "reviews.parquet"
CHROMA_DIR = DATA_INDEX / "chroma"
BM25_PATH = DATA_INDEX / "bm25.pkl"
CHROMA_COLLECTION = "olist_reviews"

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = 0.0

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_BATCH_SIZE = 64
CHROMA_INSERT_BATCH = 5000

# Recuperação
TOP_K_DENSE = 50       # candidatos da busca densa
TOP_K_BM25 = 50        # candidatos do BM25
RRF_K = 60             # constante do Reciprocal Rank Fusion (IA Gen Aula 2)
TOP_K_CONTEXT = 15     # evidências enviadas ao LLM

# Guardrails (calibrar no eval; NÃO chutar)
MIN_SCORE = 0.80       # placeholder: similaridade mínima da melhor evidência
MIN_EVIDENCES = 3      # nº mínimo de evidências acima do limiar
MAX_RETRIES = 1        # novas tentativas com HyDE antes de recusar

# Preparação
MIN_TEXT_ALNUM = 3     # mínimo de caracteres alfanuméricos; "..." e "ok" saem (decidido na EDA)
DESCONHECIDO = "desconhecido"  # valor para metadado ausente (Chroma não aceita None)

# Texto / BM25
BM25_USE_STEMMING = False  # stemming RSLP nos tokens do BM25; o eval decide se liga
