import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 4

SYSTEM_PROMPT = (
    "You are FinSolve Technologies' internal assistant. Answer the question "
    "using ONLY the context provided. If the context doesn't contain the answer, "
    "say you don't have access to that information rather than guessing. "
    "Keep answers concise."
)

def load_csv_rows(data_dir=DATA_DIR):
    rows = []
    for csv_path in data_dir.rglob("*.csv"):
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                text = "; ".join(f"{col} : {value}" for col, value in row.items())
                rows.append(
                    (text,{
                            "department": csv_path.parent.name,
                            "file_name": csv_path.name
                        }
                    )
                )
    return rows