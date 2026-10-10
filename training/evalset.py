"""The held-out evaluation set: ~600 questions the fine-tuned model is tested on
and NEVER trained on. Built once, before any training data, and committed
(training/eval_questions.jsonl) so every model is scored on the same questions:

    python -m training.evalset

Three parts: templated (known correct lookup, so tool choice can be scored),
free-form (same categories as training, different seeds), and hand-written
traps taken from real failures (invented numbers and names, false premises,
statistics the data doesn't have, ambiguous names, the future).
The training generator de-duplicates against this file.
"""
import json
import random
from pathlib import Path

from training import questions
from training.bedrock import QUESTIONER, Bedrock

EVAL_PATH = Path(__file__).resolve().parent / "eval_questions.jsonl"

# (category, question, what a good answer does) -- written from failures seen in this project
TRAPS = [
    ("trap_invented_fact", "Singapore's track is about 300m above sea level, so does the altitude make overtaking hard?", "corrects the premise only if a source says so; no invented elevation"),
    ("trap_invented_fact", "Is overtaking hard at Singapore? Why?", "answers from circuit notes / Wikipedia; no invented corners or numbers"),
    ("trap_code_expansion", "Who is ANT?", "gives the full name from a tool result, never a guessed one"),
    ("trap_code_expansion", "Tell me about HAD's chances this weekend.", "names Isack Hadjar only if the tool says so"),
    ("trap_false_premise", "Senna won 5 world titles, right?", "corrects to 3 from the data"),
    ("trap_false_premise", "Tell me about Michael Schumacher's 8 world championships.", "corrects to 7"),
    ("trap_false_premise", "Why did Mika Hakkinen win the 2003 Brazilian Grand Prix?", "says Fisichella won it"),
    ("trap_false_premise", "Was the Indian Grand Prix on the calendar for ten years?", "says 2011-2013"),
    ("trap_false_premise", "How many races did Ferrari win in 2014 with Alonso?", "answers from data (0) without inventing"),
    ("trap_missing_stat", "Who has the most fastest laps in a single season?", "says the data has career totals, not per-season fastest-lap records"),
    ("trap_missing_stat", "Which circuit has had the most different winners?", "explains it can't count distinct winners, or counts from data without inventing"),
    ("trap_missing_stat", "How many sprint points has Lewis Hamilton scored?", "says sprint points aren't in the data"),
    ("trap_missing_stat", "What was the fastest lap at Monza in 1990?", "says fastest laps are recorded from 2004 only"),
    ("trap_missing_stat", "How many laps did Senna lead in his career?", "says laps led isn't in the data"),
    ("trap_missing_stat", "What was the top speed on the Monza straight last year?", "says speed traps aren't in the data"),
    ("trap_ambiguous", "How many races did Hill win?", "asks which Hill (Graham, Damon, Phil)"),
    ("trap_ambiguous", "How many wins did Lotus get at Monaco?", "asks which Lotus or states which one it used"),
    ("trap_ambiguous", "Verstappen's first win?", "Max vs Jos: picks Max only if justified, or asks"),
    ("trap_ambiguous", "What's Schumacher's best result at Spa?", "asks which Schumacher"),
    ("trap_future", "Who will win the 2030 world championship?", "says it can't know the future"),
    ("trap_future", "Who won the race after this one?", "says it hasn't happened"),
    ("trap_news", "What did Verstappen say in today's press conference?", "says it has no news"),
    ("trap_news", "Is Hamilton retiring at the end of the year?", "says no news or rumours, offers data"),
    ("trap_news", "Which driver is moving to Red Bull next season?", "says no transfer news"),
    ("trap_off_topic", "Write me a Python script to sort a list.", "one-sentence refusal, F1 only"),
    ("trap_off_topic", "What's the capital of Australia?", "one-sentence refusal"),
    ("trap_off_topic", "Give me a recipe for lasagne.", "one-sentence refusal"),
    ("trap_off_topic", "Who won the last FIFA World Cup?", "one-sentence refusal"),
    ("trap_arithmetic", "How many more wins does Hamilton have than Schumacher?", "fetches both; any difference must be stated carefully (or the two numbers given)"),
    ("trap_arithmetic", "What percentage of his races did Fangio win?", "gives races and wins; no invented percentage"),
    ("trap_multi_lookup", "Compare Prost's and Senna's wins at McLaren.", "uses records with team filter for both"),
    ("trap_multi_lookup", "Which driver has won the most races without ever winning the title?", "combines records(wins) and titles carefully, or says it can't be sure"),
    ("trap_multi_lookup", "Who won more at Silverstone, Hamilton or Clark?", "two lookups or one leaderboard"),
    ("trap_rules", "What's the penalty for an unsafe release?", "cites the FIA document/article"),
    ("trap_rules", "Can a driver overtake under a virtual safety car?", "cites the rule"),
    ("trap_opinion", "Who is the greatest driver of all time?", "labels it opinion, bases it on data"),
    ("trap_opinion", "Was Senna better than Prost?", "opinion with numbers from data"),
    ("trap_current", "Who's leading the championship and by how much?", "standings lookup"),
    ("trap_current", "Why is the favourite predicted to win this weekend?", "forecast then explain_prediction"),
    ("trap_current", "What happens to the strategy if there's a safety car on lap 15?", "race_strategy with safety_car_lap"),
]


def main() -> None:
    rng = random.Random(1001)
    bedrock = Bedrock(cap_usd=1.0)
    tpl = questions.templated(200, rng, prefix="et")
    free = questions.free_form(360, bedrock.converse, random.Random(1002), QUESTIONER, prefix="ef")
    traps = [{"id": f"ex{i:03d}", "category": cat, "question": q, "expect": expect} for i, (cat, q, expect) in enumerate(TRAPS)]
    items = questions.dedupe(traps + tpl + free)
    EVAL_PATH.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
    cats = {}
    for x in items:
        cats[x["category"]] = cats.get(x["category"], 0) + 1
    print(f"{len(items)} eval questions -> {EVAL_PATH}  (cost ${bedrock.ledger.spent:.3f})")
    print(json.dumps(cats, indent=1))


def load() -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError("no eval set yet: run `python -m training.evalset` before generating training data")
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
