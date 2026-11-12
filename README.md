# Voice of Customer Intelligence com RAG — Olist

> README em construção.

## Execução rápida
```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash no Windows
pip install -r requirements.txt
cp .env.example .env            # preencher
# baixar o dataset do Kaggle para data/raw/
python -m src.data.prepare
python -m src.index.build_index
streamlit run app/streamlit_app.py
```
