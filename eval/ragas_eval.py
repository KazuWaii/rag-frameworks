# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ragas==0.4.3",
#     "langchain-community<0.4",
#     "python-dotenv",
# ]
# ///
import asyncio
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy, ContextRecall, Faithfulness

load_dotenv()

EVAL_DIR = Path(__file__).parent
OUTPUT = EVAL_DIR / "ragas_scores.json"
JUDGE_MODEL = "openai/gpt-oss-120b"
METRICS = ["context_recall", "faithfulness", "answer_accuracy"]

client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1",
                     api_key=os.environ["GROQ_API_KEY"], max_retries=5)
llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=4096)
context_recall = ContextRecall(llm=llm)
faithfulness = Faithfulness(llm=llm)
answer_accuracy = AnswerAccuracy(llm=llm)

async def score(item, answer):
    question, reference = item["question"], item["reference_answer"]
    response, contexts = answer["answer"], answer["contexts"]
    recall = await context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference)
    faithful = await faithfulness.ascore(user_input=question, response=response, retrieved_contexts=contexts)
    accuracy = await answer_accuracy.ascore(user_input=question, response=response, reference=reference)
    return {"context_recall": recall.value, "faithfulness": faithful.value, "answer_accuracy": accuracy.value}

async def main(framework, limit):
    testset = {q["id"]: q for q in json.loads((EVAL_DIR / "testset.json").read_text(encoding="utf-8"))}
    answers = [a for a in json.loads((EVAL_DIR / "answers.json").read_text(encoding="utf-8"))
               if a["framework"] == framework]
    results = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else []
    done = {(r["id"], r["framework"]) for r in results}

    for answer in answers[:limit]:
        if (answer["id"], framework) in done:
            continue
        scores = await score(testset[answer["id"]], answer)
        results.append({"id": answer["id"], "framework": framework, **scores})
        OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"#{answer['id']:>2} " + "  ".join(f"{name}={value:.2f}" for name, value in scores.items()))
        await asyncio.sleep(60)

    rows = [r for r in results if r["framework"] == framework]
    for metric in METRICS:
        values = [r[metric] for r in rows if not math.isnan(r[metric])]
        mean = sum(values) / len(values) if values else float("nan")
        print(f"{metric:<16} {mean:.2f}  (n={len(values)}, nan={len(rows) - len(values)})")

if __name__ == "__main__":
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(main(sys.argv[1], limit))