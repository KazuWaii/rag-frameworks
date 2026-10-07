import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
LABELS = EVAL_DIR / "human_labels.json"
N_RANDOM_OK = 13

def load(name):
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))

def is_flagged(j):
    return j["correctness"] != "correct" or j["faithful"] not in (True, "true")

def build_sample(judgments):
    flagged = [(j["id"], j["framework"]) for j in judgments if is_flagged(j)]
    ok = [(j["id"], j["framework"]) for j in judgments if not is_flagged(j)]
    rng = random.Random(42)
    sample = flagged + rng.sample(ok, N_RANDOM_OK)
    rng.shuffle(sample)
    return sample

def ask_choice(prompt, choices):
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return choices[value]

def label():
    testset = {q["id"]: q for q in load("testset.json")}
    answers = {(a["id"], a["framework"]): a for a in load("answers.json")}
    labels = load("human_labels.json") if LABELS.exists() else []
    done = {(l["id"], l["framework"]) for l in labels}
    sample = build_sample(load("judgments.json"))

    for n, key in enumerate(sample, 1):
        if key in done:
            continue
        item, answer = testset[key[0]], answers[key]
        print("\n" + "=" * 80)
        print(f"[{n}/{len(sample)}] Question: {item['question']}")
        print(f"Reference: {item['reference_answer']}")
        for i, context in enumerate(answer["contexts"], 1):
            print(f"\n--- Context {i} ---\n{context}")
        print(f"\n>>> Answer: {answer['answer']}")
        correctness = ask_choice("Correctness [c]orrect / [p]artial / [i]ncorrect: ",
                                 {"c": "correct", "p": "partial", "i": "incorrect"})
        faithful = ask_choice("Faithful [y]es / [n]o: ", {"y": True, "n": False})
        note = input("Note (optional): ")
        labels.append({"id": key[0], "framework": key[1], "correctness": correctness,
                       "faithful": faithful, "note": note})
        LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

def compare():
    judgments = {(j["id"], j["framework"]): j for j in load("judgments.json")}
    groups = {"flagged by judge": [], "ok for judge": []}
    for l in load("human_labels.json"):
        j = judgments[(l["id"], l["framework"])]
        groups["flagged by judge" if is_flagged(j) else "ok for judge"].append((l, j))

    for name, pairs in groups.items():
        same_c = sum(l["correctness"] == j["correctness"] for l, j in pairs)
        same_f = sum(l["faithful"] == (j["faithful"] in (True, "true")) for l, j in pairs)
        print(f"{name:<18} n={len(pairs):<3} correctness agree {same_c}/{len(pairs)}  faithful agree {same_f}/{len(pairs)}")

    print("\nDisagreements:")
    for pairs in groups.values():
        for l, j in pairs:
            judge_faithful = j["faithful"] in (True, "true")
            if l["correctness"] != j["correctness"] or l["faithful"] != judge_faithful:
                print(f"\n#{l['id']} {l['framework']}: you={l['correctness']}/{l['faithful']}  judge={j['correctness']}/{judge_faithful}")
                print(f"   judge: {j['correctness_reason']} | unsupported: {j['unsupported_claims']}")
                if l["note"]:
                    print(f"   you:   {l['note']}")

if __name__ == "__main__":
    if sys.argv[1:] == ["compare"]:
        compare()
    else:
        label()