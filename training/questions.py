"""The question bank for the training conversations.

Two sources:
  templated   built from the real history tables and forecasts, so the right
              first lookup is known ("gold"): the episode starts from it and the
              writer only reads the result and answers. Cheap and always correct.
  free-form   written by a model per category, steered by a persona and a few
              seed names so the bank isn't a thousand questions about Senna:
              background, explanations, comparisons, and the awkward ones a safe
              assistant must handle (not F1, news, false premises, ambiguity).

Both are de-duplicated (exact and near-duplicate), and against the held-out
evaluation set, so the model is never tested on what it was trained on.

A question is {"id", "category", "question", "gold"?: [{"name", "arguments"}],
"snapshot"?: {"season", "round", "today"}}.
"""
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import date

from src.agent import history as h
from src.agent import context, tools

# category -> (weight among free-form, what to ask the writer for)
FREE_FORM = {
    "history_stats": (14, "statistics and records from F1 history (most wins, poles, titles, best seasons, streaks, eras), phrased the way fans ask"),
    "bio_background": (11, "who a driver is, their career story, rivalries, what they are known for"),
    "team_story": (7, "a team's history: rise and fall, famous seasons, ownership, rivalries, name changes"),
    "circuit_story": (7, "a circuit's history, character, famous races, why it is special or hard"),
    "rules_explained": (8, "how the sport's rules work: DRS, points, qualifying format, tyres, flags, safety car, sprint, penalties"),
    "technical": (7, "how the cars and engineering work: aerodynamics, ground effect, hybrid power units, tyres, setup"),
    "current_season": (9, "this season and this race weekend: the next race, who is predicted to win, the championship fight, tyre strategy, how accurate the predictions are"),
    "comparisons": (7, "comparing two drivers or teams across their careers or a specific period (needs several lookups)"),
    "out_of_scope": (6, "everyday questions that are NOT about Formula 1 at all"),
    "unanswerable_news": (6, "things no historical data can answer: rumours, transfers, retirements, future results, what someone said yesterday, live news"),
    "false_premise": (6, "questions that contain a wrong fact the asker believes (a driver credited with titles or wins they never had, a race that never happened, a wrong year or team)"),
    "vague_ambiguous": (5, "questions that name only a surname shared by several F1 drivers or a team name that meant several different teams"),
    "opinion": (5, "subjective questions: greatest driver, best car, most underrated, best race, who should a team sign"),
}

PERSONAS = ["a casual fan typing on a phone, short and lowercase", "a stats nerd who loves exact numbers", "a newcomer who started watching last year",
            "an older fan who remembers the 1980s and 90s", "someone whose English is their second language", "a journalist checking facts",
            "a fantasy-league player", "a teenager using slang", "a polite, detailed asker", "an impatient asker who skips punctuation"]
RULE_TOPICS = ["DRS", "points system", "sprint format", "qualifying knockout format", "blue flags", "yellow flags", "red flag restarts", "safety car",
               "virtual safety car", "track limits", "unsafe release", "grid penalties for engine parts", "parc ferme", "tyre compound rules",
               "minimum car weight", "fuel flow limits", "cost cap", "superlicence points", "pit lane speed limit", "five-second penalties",
               "stop-go penalties", "formation lap", "jump starts", "overtaking under safety car", "team orders", "reprimands"]
TECH_TOPICS = ["ground effect", "porpoising", "hybrid power units", "MGU-K energy recovery", "tyre degradation", "undercut and overcut", "slipstream and dirty air",
               "brake cooling", "downforce vs drag", "active aerodynamics", "the halo", "car setup for Monaco", "wet tyres vs intermediates", "pit stop choreography"]
SHARED_SURNAMES = ["Schumacher", "Hill", "Verstappen", "Rosberg", "Brabham", "Fittipaldi", "Villeneuve", "Andretti", "Stuck", "Winkelhock",
                   "Magnussen", "Sainz", "Piquet", "Unser", "Whitehead"]
AMBIGUOUS_TEAMS = ["Lotus", "Alfa Romeo", "Renault", "Brabham", "March", "Toyota", "Honda"]
OFF_TOPICS = ["cooking", "maths homework", "coding", "travel", "football", "tennis", "the weather", "movies", "health", "money", "history of Rome", "cars for sale"]

GENERATE_PROMPT = """Write {n} different questions a user might type into a Formula 1 chat assistant.
Category: {desc}.
Write them as {persona}.
For inspiration you may use: {seeds}. Also include other names, eras and angles of your own; don't make every question about the inspiration names.
Vary length and specificity. Do not number them. Output ONLY a JSON array of {n} strings."""

_STOP = set("a an and are as at be by did do does for from has have how i in is it its me of on or the this to was were what when where which who why will with you your".split())


# --- entity pools from the real data ---

def _driver_pool() -> list[dict]:
    d = h._data()
    starts, wins = Counter(), Counter()
    for r in d["rows"]:
        starts[r["driver"]] += 1
        wins[r["driver"]] += int(r["win"])
    out = []
    for did, n in starts.items():
        if n >= 10:
            p = d["people"]["drivers"][did]
            out.append({"id": did, "name": p["name"], "family": p["family"], "starts": n, "wins": wins[did],
                        "unique_family": h.find_driver(p["family"])[0] == did})
    return out


def _weighted(rng: random.Random, pool: list[dict], key: str = "starts") -> dict:
    return rng.choices(pool, weights=[math.sqrt(x[key]) for x in pool])[0]


def _circuits() -> list[dict]:
    d = h._data()
    n = Counter(r["circuit"] for r in d["races"])
    return [{"id": c, "name": d["people"]["circuits"][c]["name"], "locality": d["people"]["circuits"][c]["locality"]}
            for c, k in n.items() if k >= 3 and c in d["people"]["circuits"]]


def _gps() -> list[str]:
    n = Counter(r["name"] for r in h._data()["races"])
    return [g for g, k in n.items() if k >= 5]


def _teams() -> list[str]:
    d = h._data()
    entries = Counter(r["team"] for r in d["rows"])
    return [d["people"]["constructors"][t]["name"] for t, k in entries.items() if k >= 40]


def _casual(rng: random.Random, q: str) -> str:
    """Real users rarely type tidy questions."""
    r = rng.random()
    if r < 0.2:
        return q.lower().rstrip("?")
    if r < 0.3:
        return q.rstrip("?")
    return q


def _say(rng: random.Random, driver: dict) -> str:
    return driver["family"] if driver["unique_family"] and rng.random() < 0.45 else driver["name"]


def _gp_phrase(rng: random.Random, gp: str) -> str:
    return rng.choice([gp, gp, gp.replace(" Grand Prix", " GP"), gp.replace(" Grand Prix", "")])


# --- templated questions ---

PLAN = {"winner": 9, "podium": 4, "finish_of": 4, "career": 12, "driver_at_circuit": 7, "circuit_leaders": 7, "team_at_circuit": 3, "era": 6,
        "champion": 6, "team": 7, "team_drivers": 3, "season_count": 3,
        "forecast": 5, "forecast_driver": 5, "explain": 6, "h2h": 3, "standings": 4, "title_maths": 3, "strategy": 3, "schedule": 3, "timeline": 2,
        "last_race_model": 2, "track_record": 2, "rules": 5}


def _snapshot_drivers(snap: dict | None) -> list[dict]:
    with (context.as_of(snap["season"], snap["round"], date.fromisoformat(snap["today"]), snap.get("variant", "")) if snap else nullcontext()):
        f = tools.forecast()
    return [{"code": d["driver"], "name": d.get("name") or d["driver"]} for d in f["drivers"]]


def templated(n: int, rng: random.Random, snapshots: list[dict] | None = None, prefix: str = "t", only: set[str] | None = None) -> list[dict]:
    """`snapshots` (from training.generate) each carry a weight; the current
    weekend is used when none are given. `only` limits the kinds generated."""
    drivers, circuits, gps, teams = _driver_pool(), _circuits(), _gps(), _teams()
    snaps = snapshots or [None]
    weights = [s["weight"] for s in snapshots] if snapshots else [1]
    field = {json.dumps(s, sort_keys=True): _snapshot_drivers(s) for s in snaps}
    kinds = [k for k, w in PLAN.items() for _ in range(w) if not only or k in only]
    out = []
    for i in range(n):
        kind = rng.choice(kinds)
        snap = rng.choices(snaps, weights=weights)[0]
        cur = field[json.dumps(snap, sort_keys=True)]
        d, c = _weighted(rng, drivers), rng.choice(circuits)
        gp = rng.choice(gps)
        year = rng.randint(1950, (snap["season"] - 1) if snap else 2025)
        metric = rng.choice(["wins", "poles", "podiums"])
        say = lambda x: _say(rng, x)  # noqa: E731
        if kind == "winner":
            q = rng.choice(["Who won the {y} {g}?", "Who took the win at the {g} in {y}?", "{g} {y} winner?", "who won the {g} back in {y}",
                            "Which driver won the {y} {g}?", "Remind me who won the {g} in {y}"]).format(y=year, g=_gp_phrase(rng, gp))
            gold = [("history_results", {"season": year, "grand_prix": gp})]
        elif kind == "podium":
            q = rng.choice(["Who was on the podium at the {y} {g}?", "Top three at the {g} {y}?", "Who finished second at the {y} {g}?"]).format(y=year, g=_gp_phrase(rng, gp))
            gold = [("history_results", {"season": year, "grand_prix": gp})]
        elif kind == "finish_of":
            rs = [r for r in h._data()["rows"] if r["driver"] == d["id"] and r["season"] < 2026]  # never a race a replayed weekend hasn't reached
            if not rs:
                continue
            r0 = rng.choice(rs)
            q = rng.choice(["Where did {d} finish at the {y} {g}?", "How did {d} do at the {g} in {y}?", "What was {d}'s result in the {y} {g}?"]).format(
                d=say(d), y=r0["season"], g=_gp_phrase(rng, r0["race"]))
            gold = [("history_results", {"season": r0["season"], "round": r0["round"]})]
        elif kind == "career":
            q = rng.choice(["How many races did {d} win?", "How many world championships did {d} win?", "Give me {d}'s career numbers.",
                            "How many pole positions did {d} get?", "How many podiums did {d} have?", "Which teams did {d} drive for?",
                            "When did {d} make their F1 debut?", "How many seasons did {d} race in F1?", "What's {d}'s best championship finish?"]).format(d=say(d))
            gold = [("driver_career", {"driver": d["name"]})]
        elif kind == "driver_at_circuit":
            q = rng.choice(["How many times did {d} win at {c}?", "Did {d} ever win at {c}?", "{d}'s record at {c}?", "How many poles did {d} take at {c}?"]).format(d=say(d), c=c["locality"])
            gold = [("records", {"metric": "poles" if "poles" in q else "wins", "driver": d["name"], "circuit": c["name"]})]
        elif kind == "circuit_leaders":
            q = rng.choice(["Who has the most {m} at {c}?", "Which driver is the most successful at {c} by {m}?", "most {m} at {c}?", "Top 3 drivers for {m} at {c}?"]).format(m=metric, c=c["locality"])
            gold = [("records", {"metric": metric, "circuit": c["name"], "n": 3})]
        elif kind == "team_at_circuit":
            q = rng.choice(["Which team has won the most at {c}?", "Most successful constructor at {c}?"]).format(c=c["locality"])
            gold = [("records", {"metric": "wins", "by": "team", "circuit": c["name"], "n": 3})]
        elif kind == "era":
            y1 = rng.randint(1955, 2018)
            y2 = min(y1 + rng.choice([4, 9, 14]), 2025)
            q = rng.choice(["Who won the most races between {a} and {b}?", "Most {m} from {a} to {b}?", "Which driver dominated {a}-{b} by {m}?",
                            "Which team won the most races from {a} to {b}?"]).format(a=y1, b=y2, m=metric)
            by = "team" if "team" in q else "driver"
            gold = [("records", {"metric": "wins" if "races" in q else metric, "by": by, "from_year": y1, "to_year": y2, "n": 3})]
        elif kind == "champion":
            y = rng.randint(1958, 2025)
            q = rng.choice(["Who won the {y} world championship?", "Who was champion in {y}?", "Who won the {y} constructors' title?",
                            "How close was the {y} title fight?"]).format(y=y)
            gold = [("season_summary", {"year": y})]
        elif kind == "season_count":
            y = rng.randint(1950, 2025)
            q = rng.choice(["How many races were there in {y}?", "Who won the most races in {y}?", "How many different winners were there in {y}?"]).format(y=y)
            gold = [("season_summary", {"year": y})]
        elif kind == "team":
            t = rng.choice(teams)
            q = rng.choice(["How many wins does {t} have?", "When did {t} win the constructors' championship?", "Give me {t}'s history in numbers.",
                            "How long did {t} race in F1?"]).format(t=t)
            gold = [("team_history", {"team": t})]
        elif kind == "team_drivers":
            t = rng.choice(teams)
            q = rng.choice(["Who won the most races for {t}?", "Which driver has the most podiums with {t}?"]).format(t=t)
            gold = [("records", {"metric": "podiums" if "podiums" in q else "wins", "team": t, "n": 3})]
        elif kind == "forecast":
            q = rng.choice(["Who is predicted to win this weekend?", "What does the forecast say for this race?", "Who's the favourite for the race?",
                            "Give me the predicted top 3 for this weekend.", "Who's on pole according to the predictions?", "Which race is this weekend and who's favourite?"])
            gold = [("forecast", {})]
        elif kind == "forecast_driver":
            x = rng.choice(cur)
            q = rng.choice(["Where is {n} expected to finish this weekend?", "What are {n}'s chances this weekend?", "Can {c} get a podium this race?",
                            "What's {n}'s chance of pole?"]).format(n=x["name"].split()[-1] if rng.random() < 0.5 else x["name"], c=x["code"])
            gold = [("forecast", {"driver": x["code"]})]
        elif kind == "explain":
            x = rng.choice(cur)
            target = rng.choice([None, None, "qualifying", "quali_delta"])
            q = rng.choice({None: ["Why is {c} predicted where he is?", "What's driving the prediction for {n}?", "Explain {c}'s forecast to me."],
                            "qualifying": ["Why is {n} expected to qualify there?"], "quali_delta": ["Why does the model think {c} will gain or lose places?"]}[target]).format(
                c=x["code"], n=x["name"])
            gold = [("explain_prediction", {"driver": x["code"], **({"target": target} if target else {})})]
        elif kind == "h2h":
            a, b = rng.sample(cur, 2)
            q = rng.choice(["{a} vs {b}: who finishes ahead?", "Will {a} beat {b} this weekend?", "Who's more likely to finish ahead, {a} or {b}?"]).format(a=a["code"], b=b["code"])
            gold = [("head_to_head", {"driver_a": a["code"], "driver_b": b["code"]})]
        elif kind == "standings":
            ctor = rng.random() < 0.35
            q = "Who is leading the constructors' championship?" if ctor else rng.choice(["Who is leading the championship?", "What are the drivers' standings?",
                                                                                         "Who can still win the title?", "How many points does the leader have?"])
            gold = [("championship", {"kind": "constructors" if ctor else "drivers"})]
        elif kind == "title_maths":
            x = rng.choice(cur[:8])
            q = rng.choice(["What does {n} need to win the title?", "Can {n} still win the championship?"]).format(n=x["name"])
            gold = [("title_scenario_for", {"driver": x["code"]})]
        elif kind == "strategy":
            lap = rng.randint(8, 45)
            q = rng.choice(["What's the best tyre strategy for this race?", "When should they pit this weekend?", f"What changes if there's a safety car on lap {lap}?",
                            "One stop or two this weekend?"])
            gold = [("race_strategy", {"safety_car_lap": lap} if "safety car" in q else {})]
        elif kind == "schedule":
            q = rng.choice(["When is the next race?", "What's left on the calendar this season?", "Which are the sprint weekends this year?", "How many races are left?"])
            gold = [("season_schedule", {})]
        elif kind == "timeline":
            x = rng.choice(cur)
            q = rng.choice(["How has the forecast changed through the weekend?", "Has {c}'s prediction moved since practice?"]).format(c=x["code"])
            gold = [("forecast_timeline", {"driver": x["code"]} if x["code"] in q else {})]
        elif kind == "last_race_model":
            if not snap or snap["round"] < 3:
                continue
            r_ = rng.randint(max(1, snap["round"] - 4), snap["round"] - 1)
            q = rng.choice(["How did your predictions do at round {r} this season?", "Did the model get round {r} right?"]).format(r=r_)
            gold = [("past_race_prediction", {"season": snap["season"], "round": r_})]
        elif kind == "track_record":
            q = rng.choice(["How accurate are your predictions?", "Can I trust the forecast?", "How good is the model at picking winners?"])
            gold = [("model_track_record", {})]
        else:
            topic = rng.choice(RULE_TOPICS)
            q = rng.choice(["What do the regulations say about {t}?", "What's the rule on {t}?", "Explain {t} according to the rules."]).format(t=topic)
            gold = [("search_knowledge", {"query": topic, "only": "rules"})]
        out.append({"id": f"{prefix}{i:05d}", "category": f"tpl_{kind}", "question": _casual(rng, q),
                    "gold": [{"name": n_, "arguments": a_} for n_, a_ in gold], **({"snapshot": _plain_snap(snap)} if snap else {})})
    return out


def _plain_snap(snap: dict) -> dict:
    return {k: snap[k] for k in ("season", "round", "today", "variant") if k in snap}


# --- free-form questions ---

def _seeds(rng: random.Random, cat: str, drivers: list[dict], circuits: list[dict], teams: list[str]) -> str:
    pick = lambda k: ", ".join(_weighted(rng, drivers)["name"] for _ in range(k))  # noqa: E731
    return {
        "history_stats": f"{pick(2)}; {rng.choice(circuits)['name']}; the {rng.choice(range(1950, 2030, 10))}s",
        "bio_background": pick(3),
        "team_story": ", ".join(rng.sample(teams, 2)),
        "circuit_story": ", ".join(c["name"] for c in rng.sample(circuits, 2)),
        "rules_explained": ", ".join(rng.sample(RULE_TOPICS, 3)),
        "technical": ", ".join(rng.sample(TECH_TOPICS, 3)),
        "current_season": "this weekend's race, the title fight, strategy, the forecast",
        "comparisons": f"{pick(2)}; or the teams {', '.join(rng.sample(teams, 2))}",
        "out_of_scope": ", ".join(rng.sample(OFF_TOPICS, 3)),
        "unanswerable_news": pick(2),
        "false_premise": f"{pick(2)}; {rng.choice(teams)}",
        "vague_ambiguous": ", ".join(rng.sample(SHARED_SURNAMES, 3) + rng.sample(AMBIGUOUS_TEAMS, 1)),
        "opinion": f"{pick(2)}; the {rng.choice(range(1950, 2030, 10))}s",
    }[cat]


def free_form(n: int, writer, rng: random.Random, model: str, per_call: int = 10, workers: int = 6, prefix: str = "f",
              boost: dict[str, float] | None = None) -> list[dict]:
    """`writer(model, system, messages, ...)` is Bedrock.converse. Asks ~35% more
    than needed per category, then de-duplicates. `boost` multiplies a category's
    share (used to ask for more of the categories that pass the gates less often)."""
    drivers, circuits, teams = _driver_pool(), _circuits(), _teams()
    weights = {cat: w * (boost or {}).get(cat, 1.0) for cat, (w, _) in FREE_FORM.items()}
    total_w = sum(weights.values())
    jobs = []
    for cat, (_, desc) in FREE_FORM.items():
        want = max(2, round(n * weights[cat] / total_w))
        for _ in range(math.ceil(want * 1.35 / per_call)):
            jobs.append((cat, desc, PERSONAS[rng.randrange(len(PERSONAS))], _seeds(rng, cat, drivers, circuits, teams)))

    def ask(job):
        cat, desc, persona, seeds = job
        resp = writer(model, "You write realistic user questions for an F1 chat assistant. Reply with JSON only.",
                      [{"role": "user", "content": [{"text": GENERATE_PROMPT.format(n=per_call, desc=desc, persona=persona, seeds=seeds)}]}],
                      max_tokens=1500, temperature=1.0)
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        m = re.search(r"\[.*\]", text, re.S)
        try:
            return cat, [q.strip() for q in json.loads(m.group()) if isinstance(q, str) and 8 < len(q.strip()) < 300] if m else []
        except json.JSONDecodeError:
            return cat, []

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(ask, jobs))
    by_cat = defaultdict(list)
    for cat, qs in results:
        by_cat[cat] += qs
    out = []
    for cat in FREE_FORM:
        want = max(2, round(n * weights[cat] / total_w))
        qs = dedupe([{"question": q} for q in by_cat[cat]])
        rng.shuffle(qs)
        out += [{"id": f"{prefix}{len(out) + k:05d}", "category": cat, "question": q["question"]} for k, q in enumerate(qs[:want])]
    return out


# --- de-duplication ---

def _tokens(q: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in _STOP]


def dedupe(items: list[dict], against: list[dict] = (), threshold: float = 0.8) -> list[dict]:
    """Drop exact and near-duplicate questions (Jaccard of content words >=
    `threshold`), within `items` and against `against` (e.g. the eval set).
    Candidates are found through each question's two rarest words, so this
    stays fast for tens of thousands of questions."""
    all_q = [x["question"] for x in list(against) + list(items)]
    df = Counter(t for q in all_q for t in set(_tokens(q)))
    seen_exact: set[str] = set()
    buckets: dict[str, list[set]] = defaultdict(list)

    def key_tokens(toks: set) -> list[str]:
        return sorted(toks, key=lambda t: (df[t], t))[:2]

    def is_dup(toks: set) -> bool:
        for t in key_tokens(toks):
            for other in buckets[t]:
                if toks and len(toks & other) / len(toks | other) >= threshold:
                    return True
        return False

    def add(toks: set) -> None:
        for t in key_tokens(toks):
            buckets[t].append(toks)

    for x in against:
        toks = set(_tokens(x["question"]))
        seen_exact.add(" ".join(sorted(toks)))
        add(toks)
    out = []
    for x in items:
        toks = set(_tokens(x["question"]))
        exact = " ".join(sorted(toks))
        if not toks or exact in seen_exact or is_dup(toks):
            continue
        seen_exact.add(exact)
        add(toks)
        out.append(x)
    return out
