import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import haystack_rag
import langchain_rag
import llamaindex_rag
from common import ALL_DEPARTMENTS

FRAMEWORKS = {
    "llamaindex": (llamaindex_rag.ask, llamaindex_rag.get_index),
    "langchain": (langchain_rag.ask, langchain_rag.get_vectorstore),
    "haystack": (haystack_rag.ask, haystack_rag.get_pipelines),
}
TESTSET = Path(__file__).parent / "testset.json"
OUTPUT = Path(__file__).parent / "answers.json"

def load_answers():
    if OUTPUT.exists():
        return json.loads(OUTPUT.read_text(encoding="utf-8"))
    return []

def save_answers(answers):
    OUTPUT.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    answers = load_answers()
    done = {(a["id"], a["framework"]) for a in answers}

    for name, (ask, build) in FRAMEWORKS.items():
        build()
        for item in testset:
            if (item["id"], name) in done:
                continue
            start = time.perf_counter()
            result = ask(item["question"], ALL_DEPARTMENTS)
            latency = time.perf_counter() - start
            answers.append({
                "id": item["id"],
                "framework": name,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "latency_s": round(latency, 2),
            })
            save_answers(answers)
            print(f"[{name}] #{item['id']} ({latency:.1f}s) {result['answer'][:60]!r}")
            time.sleep(15)