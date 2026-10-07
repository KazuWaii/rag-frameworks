# RAG frameworks comparison

The same RAG pipeline over FinSolve's internal documents (from ds-rpc-01), built three times with LlamaIndex,
LangChain and Haystack, to compare what each framework does by default.

## Shared setup

- Data: 9 Markdown files and 1 CSV (100 employee rows) in `data/`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Vector index: FAISS `IndexFlatL2` (exact search)
- LLM: Groq `openai/gpt-oss-20b`, same prompt in all three versions (`common.py`)
- Retrieval: top 4 chunks
- Chunking: split on Markdown headers, then 512 tokens with 128 overlap

## Findings

Test question: "What is Aadhya Patel's salary?" (expected answer: 1332478.37).

| | LlamaIndex | LangChain | Haystack |
|---|---|---|---|
| Chunks | 368 | 309 | 309 |
| Header-only chunks | 59 | 0 | 0 |
| Embedded text | metadata + text | text | text |
| Rank of the correct row | 3 | 1 | 1 |
| Answer | ₹1,332,478.37 | 1332478.37 | 1332478.37 |
| Reported score | squared L2 distance (lower is better) | squared L2 distance (lower is better) | 1 / (1 + squared L2) (higher is better) |
| Default chunk unit | tokens (cl100k_base) | characters | words (tokens use o200k_base) |

- With the same configuration, LangChain and Haystack retrieve the same chunks in the same order. LlamaIndex differs
  because of two defaults:
  - `MarkdownNodeParser` creates a node for every header, including 59 headers with no text of their own.
  - Metadata (`department`, `file_name`, `header_path`) is embedded with the text. This prefix is identical on all
    100 CSV rows and lowers the cosine similarity of the correct row from 0.575 to 0.396. It can be turned off with
    `excluded_embed_metadata_keys`.
- LlamaIndex added a currency (₹) that is not in the source row. Its context included a handbook chunk mentioning
  ₹50,000; the other two versions, whose context did not, returned the raw value.
- Scores are not comparable across frameworks, even with the same FAISS index.
- Data note: two employees are named Ishaan Patel (FINEMP1012 and FINEMP1026) and share the same email address.

These are observations from one question and one run per framework, to be confirmed by the evaluation.

## Web demo

`streamlit_app.py` asks one framework, or all three side by side, and shows each answer with the 4 chunks it was
built from, each labelled with its source file. The *Documents* page shows the full source files, to check an answer.
Demo account: `demo` / `finsolve`. Each session is limited to 30 LLM calls to protect the Groq quota.

Run it locally (needs `GROQ_API_KEY` in `.env`):

```bash
uv run streamlit run streamlit_app.py
```

The first question sent to a framework also builds its index, which takes about a minute.

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io/), create an app from the repository with `streamlit_app.py`
   as the main file. Community Cloud installs the dependencies from `uv.lock`; torch comes from PyTorch's CPU-only
   index (see `pyproject.toml`), which avoids several GB of CUDA packages on Linux.
3. In *Advanced settings*, choose Python 3.12 and add the secret `GROQ_API_KEY = "..."`.

Apps go to sleep after 12 hours without traffic; the next visitor wakes them up with one click.
