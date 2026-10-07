import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from groq import APIError, Groq

import langchain_rag

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
PER_DEPARTMENT = 4
OUTPUT = Path(__file__).parent / "testset.json"

PROMPT = (
    "You write test questions for an internal company assistant. From the text below, write ONE question "
    "an employee could realistically ask that can be answered using only this text, and its short answer. "
    "Do not mention 'the text' or 'the document'; the question must make sense on its own. "
    'Return JSON only: {"question": "...", "answer": "..."}'
)

def sample_chunks():
    chunks = langchain_rag.split_document(langchain_rag.load_files())
    by_department = defaultdict(list)
    for chunk in chunks:
        if len(chunk.page_content.split()) >= 30:
            by_department[chunk.metadata["department"]].append(chunk)
    rng = random.Random(42)
    return [chunk for dept in sorted(by_department) for chunk in rng.sample(by_department[dept], PER_DEPARTMENT)]

def generate(client, context):
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": PROMPT + "\n\nText:\n" + context}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    client = Groq()
    testset = []
    for i, chunk in enumerate(sample_chunks()):
        try:
            qa = generate(client, chunk.page_content)
        except (json.JSONDecodeError, APIError) as error:
            print(f"skip {i}: {error}")
            continue
        testset.append({
            "id": i,
            "department": chunk.metadata["department"],
            "file_name": chunk.metadata["file_name"],
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "reference_context": chunk.page_content,
        })
        print(f"[{chunk.metadata['department']}] {qa['question']}")
        time.sleep(8)
    OUTPUT.write_text(json.dumps(testset, indent=2, ensure_ascii=False), encoding="utf-8")