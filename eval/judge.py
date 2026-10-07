import json
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import APIError, Groq

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
EVAL_DIR = Path(__file__).parent
OUTPUT = EVAL_DIR / "judgments.json"
FRAMEWORKS = ["llamaindex", "langchain", "haystack"]

JUDGE_PROMPT = """You are evaluating the answer of a company's internal assistant.

Question: {question}
Reference answer: {reference}
Retrieved context:
{context}
Assistant's answer: {answer}

Return JSON with these fields:
- "correctness": "correct", "partial" or "incorrect", compared with the reference answer.
  "correct": same facts as the reference; wording and extra correct details do not matter.
  If the question is ambiguous and the answer lists every valid option, it is "correct".
  "partial": some facts right, some missing or wrong. "incorrect": wrong, or a refusal to answer.
- "correctness_reason": one sentence.
- "faithful": true if every fact in the answer is supported by the retrieved context, else false.
  A refusal to answer is faithful.
- "unsupported_claims": list of the facts in the answer that the context does not support.
"""

def judge(client, item, answer):
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        reference=item["reference_answer"],
        context="\n\n---\n\n".join(answer["contexts"]),
        answer=answer["answer"],
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    testset = {q["id"]: q for q in json.loads((EVAL_DIR / "testset.json").read_text(encoding="utf-8"))}
    answers = json.loads((EVAL_DIR / "answers.json").read_text(encoding="utf-8"))
    judgments = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else []
    done = {(j["id"], j["framework"]) for j in judgments}
    client = Groq()

    for answer in answers:
        if (answer["id"], answer["framework"]) in done:
            continue
        try:
            verdict = judge(client, testset[answer["id"]], answer)
        except (json.JSONDecodeError, APIError) as error:
            print(f"skip #{answer['id']} {answer['framework']}: {error}")
            continue
        judgments.append({"id": answer["id"], "framework": answer["framework"], **verdict})
        OUTPUT.write_text(json.dumps(judgments, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"#{answer['id']:>2} {answer['framework']:<11} {verdict['correctness']:<10} faithful={verdict['faithful']}")
        time.sleep(15)

    print()
    for fw in FRAMEWORKS:
        rows = [j for j in judgments if j["framework"] == fw]
        n = len(rows)
        correct = sum(j["correctness"] == "correct" for j in rows)
        partial = sum(j["correctness"] == "partial" for j in rows)
        faithful = sum(j["faithful"] in (True, "true") for j in rows)
        print(f"{fw:<12} correct {correct}/{n}  partial {partial}/{n}  faithful {faithful}/{n}")