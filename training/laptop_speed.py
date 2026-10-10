"""Time the finalist student models on THIS laptop, one question at a time (as
the app would run them), on a fixed mixed sample of the bake-off questions.
Speed only: no judge, no cost.

    python -m training.laptop_speed --models qwen3.5:9b-q4_K_M qwen3:14b-q4_K_M gemma4:12b-it-q4_K_M --n 20
"""
import argparse
import json
import random
import statistics
import time

from training.student_bakeoff import OUT, play, sample

GIVE_UP_S = 300  # a model averaging 5 min/answer after 5 questions is unusable; stop rather than burn an hour on it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    items = random.Random(11).sample(sample(200), args.n)  # same 20 for every model, traps and free-form mixed
    path = OUT / "laptop_speed.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for model in args.models:
        play(args.ollama, model, {"category": "warmup", "question": "Hi"})  # load the model into memory; not timed
        eps = []
        for i, it in enumerate(items):
            eps.append(play(args.ollama, model, it))
            print(f"{model} {i + 1}/{len(items)}: {eps[-1]['wall_s']}s, {len(eps[-1]['calls'])} lookups", flush=True)
            if i == 4 and statistics.mean(e["wall_s"] for e in eps) > GIVE_UP_S:
                print(f"{model}: over {GIVE_UP_S}s per answer, stopped", flush=True)
                break
        walls = sorted(e["wall_s"] for e in eps)
        results[model] = {"questions": len(eps), "errors": sum(bool(e["error"]) for e in eps),
                          "median_s": statistics.median(walls), "p90_s": walls[int(0.9 * (len(walls) - 1))], "max_s": walls[-1],
                          "gen_tokens_per_s": round(sum(e["gen_tokens"] for e in eps) / max(1e-9, sum(e["gen_s"] for e in eps)), 1)}
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"{model}: {json.dumps(results[model])}", flush=True)


if __name__ == "__main__":
    main()
