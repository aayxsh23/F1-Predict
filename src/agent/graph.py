"""Phase 6: LangGraph live-standings agent.

Architecture note (see LEARNING.md for the full reasoning): a real test of
the local Llama 3.2 1B Instruct's LangChain tool-calling (bind_tools via
ChatHuggingFace) came back with zero tool calls and no tool schema even
reaching the model's prompt -- not reliable enough to build a ReAct-style
"let the LLM pick which tool to call" agent on. So this graph does the
multi-step orchestration itself (classify -> fetch/compute -> synthesize) and
only asks the LLM for the two things it's actually good at: picking one of a
few known intents, and writing the final answer from real, already-computed
data. It's still a genuine LangGraph StateGraph -- multiple nodes, conditional
routing, checkpointed memory for follow-ups -- not manual pipeline glue.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.f1_api import get_constructor_standings, get_current_round, get_driver_standings, get_season_schedule
from src.agent.scenarios import closest_rival, remaining_rounds, title_scenario
from src.rag.llm import generate

INTENTS = ("driver_standings", "constructor_standings", "title_scenario", "general")

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user's F1 question into exactly one category. Reply with "
    "ONLY the category word, nothing else. Categories:\n"
    "- driver_standings: asking about a driver's current points/position/championship standing\n"
    "- constructor_standings: asking about a team/constructor's current points/position\n"
    "- title_scenario: asking what a driver needs to do to win/clinch the championship, "
    "or whether they're still mathematically in contention\n"
    "- general: anything else"
)

ANSWER_SYSTEM_PROMPT = (
    "You are an F1 live-standings assistant. Answer the user's question using ONLY "
    "the data provided below. Do not invent points, positions, or facts not given to you. "
    "Be concise -- 2-4 sentences."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str | None
    data: dict | None


def _extract_driver_code(question: str, standings: list[dict]) -> str | None:
    """Match a driver named in the question against real current standings
    (3-letter code or family name) -- deterministic, not LLM-guessed, since
    an invented or misheard driver code would silently compute a scenario
    for the wrong person with no way to tell."""
    q = question.lower()
    words = q.split()
    for d in standings:
        if d["code"].lower() in words or d["family_name"].lower() in q:
            return d["code"]
    return None


def classify_node(state: AgentState) -> dict:
    """Heuristic keyword match first, LLM classification only as a last
    resort -- flipped from an earlier LLM-first design after a real test
    showed the LLM route misclassifying "who is leading the constructors
    championship" as title_scenario (the word "championship" alone is a weak
    signal the model over-weighted). The keyword rules below are deterministic
    and directly test-covered; a 1B model's single-word classification isn't,
    so it only gets used when nothing else applies."""
    question = state["messages"][-1].content
    ql = question.lower()

    if any(w in ql for w in ("constructor", "team standings", "team's points", "team points", "team in")):
        return {"intent": "constructor_standings"}
    if any(w in ql for w in ("need to do to win", "clinch", "mathematically", "still in contention",
                              "win the title", "win the championship")):
        return {"intent": "title_scenario"}
    if any(w in ql for w in ("driver", "points", "standing", "position", "leading", "champion")):
        return {"intent": "driver_standings"}

    # Inherit the previous turn's intent ONLY for an explicit continuation
    # ("and what about...", "how about...") -- an earlier version inherited
    # for ANY unmatched question, which wrongly answered a completely
    # unrelated question ("when did F1 start using turbo hybrid engines?")
    # with leftover constructor-standings data from two turns earlier. A
    # continuation marker at the START of the sentence is a real, specific
    # discourse signal; "didn't match any keyword" is not.
    previous_intent = state.get("intent")
    continuation_markers = ("and ", "also ", "what about", "how about")
    if previous_intent and ql.strip().startswith(continuation_markers):
        return {"intent": previous_intent}

    raw = generate(CLASSIFY_SYSTEM_PROMPT, question, max_new_tokens=10).strip().lower()
    intent = next((i for i in INTENTS if i in raw), "general")
    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    return {
        "driver_standings": "fetch_driver_standings",
        "constructor_standings": "fetch_constructor_standings",
        "title_scenario": "compute_title_scenario",
        "general": "synthesize_answer",
    }[state["intent"]]


def fetch_driver_standings_node(state: AgentState) -> dict:
    return {"data": {"driver_standings": get_driver_standings()}}


def fetch_constructor_standings_node(state: AgentState) -> dict:
    return {"data": {"constructor_standings": get_constructor_standings()}}


def compute_title_scenario_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    standings = get_driver_standings()
    driver_code = _extract_driver_code(question, standings)
    if driver_code is None:
        # no driver named -- default to the current championship leader
        driver_code = min(standings, key=lambda d: d["position"])["code"]

    driver = next(d for d in standings if d["code"] == driver_code)
    rival = closest_rival(standings, driver_code)
    remaining = remaining_rounds(get_season_schedule(), get_current_round())
    scenario = title_scenario(driver["points"], rival["points"], remaining)

    return {"data": {"driver": driver, "rival": rival, "scenario": scenario}}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_GAP_WORDS = ("gap", "behind", "ahead", "difference")


def _format_driver_standings_answer(data: dict, question: str) -> str:
    standings = data["driver_standings"]
    ql = question.lower()
    code = _extract_driver_code(question, standings)
    if code:
        d = next(x for x in standings if x["code"] == code)
        wins = f"{d['wins']} win{'s' if d['wins'] != 1 else ''}"
        return f"{d['given_name']} {d['family_name']} ({d['team']}) is {_ordinal(d['position'])} with {d['points']:.0f} points and {wins}."
    if any(w in ql for w in _GAP_WORDS):
        first, second = standings[0], standings[1]
        return (f"{first['family_name']} leads {second['family_name']} by "
                f"{first['points'] - second['points']:.0f} points ({first['points']:.0f} vs {second['points']:.0f}).")
    lines = "; ".join(f"{d['position']}. {d['family_name']} ({d['team']}) - {d['points']:.0f} pts" for d in standings[:5])
    return f"Current driver standings (top 5): {lines}."


def _format_constructor_standings_answer(data: dict, question: str) -> str:
    standings = data["constructor_standings"]
    ql = question.lower()
    named = next((c for c in standings if c["name"].lower() in ql), None)
    if named:
        wins = f"{named['wins']} win{'s' if named['wins'] != 1 else ''}"
        return f"{named['name']} is {_ordinal(named['position'])} in the constructors' championship with {named['points']:.0f} points and {wins}."
    if any(w in ql for w in _GAP_WORDS):
        first, second = standings[0], standings[1]
        return (f"{first['name']} leads {second['name']} by "
                f"{first['points'] - second['points']:.0f} points ({first['points']:.0f} vs {second['points']:.0f}).")
    lines = "; ".join(f"{c['position']}. {c['name']} - {c['points']:.0f} pts" for c in standings[:5])
    return f"Current constructor standings (top 5): {lines}."


def _format_title_scenario_answer(data: dict) -> str:
    driver, rival, scenario = data["driver"], data["rival"], data["scenario"]
    name, rival_name = f"{driver['given_name']} {driver['family_name']}", f"{rival['given_name']} {rival['family_name']}"
    races, sprints, pool = scenario["remaining_races"], scenario["remaining_sprints"], scenario["max_points_available"]

    if not scenario["still_mathematically_in_contention"]:
        return (f"{name} is mathematically out of contention against {rival_name}: even winning every "
                f"remaining race ({races} races, {sprints} sprints, {pool} points available) wouldn't be "
                f"enough to catch {rival_name}'s current {rival['points']:.0f} points.")
    gap = scenario["points_behind"]
    if gap <= 0:
        return (f"{name} currently leads {rival_name} by {abs(gap):.0f} points, with {races} races "
                f"({sprints} sprints) and {pool} points still on the table.")
    return (f"{name} is {gap:.0f} points behind {rival_name}, with {races} races ({sprints} sprints, "
            f"{pool} points available) left. Assuming {rival_name} scores nothing more, {name} would need "
            f"{scenario['points_needed_if_rival_scores_zero']:.0f} points to draw level -- in practice, "
            f"that means consistently out-scoring {rival_name} across the remaining rounds.")


def synthesize_answer_node(state: AgentState) -> dict:
    """Structured intents (driver/constructor standings, title scenarios) get
    a deterministic answer templated directly from real computed data -- no
    LLM in that path. A real test of the free-form version showed the base
    model hallucinating badly even when handed correct data verbatim (a
    fabricated "red-carbon fiber wall", an invented driver code, numbers with
    no basis in the actual JSON) -- the same lesson as the tool-calling
    decision this graph's docstring explains: use the LLM only where it's
    actually reliable. Only `general` (no structured data exists to template)
    still goes through the LLM, with the same honest caveat chat.py's
    open-ended format already carries -- answers there are noticeably
    rougher than the templated ones, since there's no deterministic fallback
    for a genuinely open-ended question."""
    question = state["messages"][-1].content
    intent = state.get("intent")
    data = state.get("data")

    if intent == "driver_standings" and data:
        answer = _format_driver_standings_answer(data, question)
    elif intent == "constructor_standings" and data:
        answer = _format_constructor_standings_answer(data, question)
    elif intent == "title_scenario" and data:
        answer = _format_title_scenario_answer(data)
    else:
        prompt = f"Question: {question}\n\n(No live F1 data applies to this question -- answer briefly and say so if it needs data you don't have.)"
        answer = generate(ANSWER_SYSTEM_PROMPT, prompt)

    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("fetch_driver_standings", fetch_driver_standings_node)
    graph.add_node("fetch_constructor_standings", fetch_constructor_standings_node)
    graph.add_node("compute_title_scenario", compute_title_scenario_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_intent)
    graph.add_edge("fetch_driver_standings", "synthesize_answer")
    graph.add_edge("fetch_constructor_standings", "synthesize_answer")
    graph.add_edge("compute_title_scenario", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    # MemorySaver + a thread_id at invoke time is how follow-up questions in
    # the same conversation see the earlier messages (project plan's "add
    # basic state/memory so it can handle follow-up questions")
    return graph.compile(checkpointer=MemorySaver())


def ask(graph, question: str, thread_id: str = "default") -> str:
    """Only `messages` is passed as input -- NOT a fresh `intent`/`data` reset
    every call. The checkpointer carries the previous turn's `intent`/`data`
    forward as this call's starting state (LangGraph merges input into
    checkpointed state, it doesn't replace it), which is exactly what lets
    classify_node's follow-up-inherits-previous-intent logic see it."""
    from langchain_core.messages import HumanMessage
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content
