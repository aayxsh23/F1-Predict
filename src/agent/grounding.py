"""The grounding check: numbers and names in an answer that appear nowhere in
what the writer was given (the question, tool results, instructions).
AGENTS.md: every number in a chat answer comes from Python and every name
should too. Used on the live chat's model answers (src/agent/chat.py) and as
the first quality gate on generated training conversations (training/quality.py).

It is a heuristic with a known ceiling: it catches invented specifics (a
figure, a driver, a place) but not a wrong claim in ordinary words, and it
rejects honest arithmetic (a sum of two results). Real failures it caught: a
circuit "over 300m above sea level", and ANT expanded to "António Félix da
Costa" (a Formula E driver).
"""
import math
import re

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# in an answer, digits glued to letters are labels (Q1, FP2, F1, R14), not claims; finishing positions (P9) are still checked
_ANSWER_NUMBER = re.compile(r"(?:(?<=\b[Pp])|(?<![A-Za-z]))\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-zÀ-ÿ][\wÀ-ÿ'’-]*")
_SEGMENT = re.compile(r"[.!?:;\n|]+|\s[-•*]\s|\*\*|\(\s*|—")
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
MONTH_ABBREVIATIONS = {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
# capitalised words that are fine without appearing in the data
SHORTHAND = re.compile(r"^(P|Q|FP|R|T|S)\d{1,2}$")  # P9, Q3, FP2, R14, T1 (turn), S1 (sector)
ACRONYMS = {"dnf", "dns", "dsq", "dnq", "sc", "vsc", "kers", "ers", "mgu", "mgu-k", "mgu-h", "pu", "ice", "drs", "fia", "gp", "f1", "f2", "f3", "tv", "uk", "usa", "us"}
GENERIC = MONTHS | MONTH_ABBREVIATIONS | DAYS | ACRONYMS | {"formula", "one", "grand", "prix", "gp", "fia", "f1", "wikipedia", "pit", "radio", "jolpica", "ergast", "drs", "utc", "gmt",
                           "world", "championship", "championships", "champion", "champions", "constructors", "drivers", "related",
                           "i", "if", "sorry", "yes", "no", "mph", "kph", "km", "kg"}


def _norm(word: str) -> str:
    w = word.lower().replace("’", "'")
    return w[:-2] if w.endswith("'s") else w.rstrip("'")


def _given_words(given: str) -> set[str]:
    words = set()
    for w in re.findall(r"[\wÀ-ÿ'’-]+", given):
        n = _norm(w)
        words.add(n)
        words.update(p for p in n.split("-") if p)  # "Spa-Francorchamps" also gives "spa" and "francorchamps"
    return words


def _given_numbers(given: str) -> set[float]:
    """Every number in the text as a float, so "76" matches "76.0", plus its
    roundings to 0-2 decimals; a probability like 0.191 also allows its
    percentage forms 19.1 and 19. (Signs are ignored: the regex sees 2.68.)"""
    numbers = {float(n) for n in _NUMBER.findall(given)} | {0.0}  # "0 wins" is what an empty result means
    for x in list(numbers):
        numbers.update({float(math.floor(x)), float(math.ceil(x))})  # "around P9" for 9.5
        for d in (0, 1, 2, 3):  # honest rounding: 2.6812 quoted as 2.681, 2.68, 2.7 or 3
            numbers.add(round(x, d))
            if 0 < x <= 1:
                numbers.add(round(x * 100, d))
    return numbers


def _known_word(w: str, words: set[str]) -> bool:
    n = _norm(w)
    return n in words or n.removesuffix("s") in words or n in GENERIC or bool(SHORTHAND.match(w)) or n.isdigit()


def ungrounded(answer: str, given: str) -> list[str]:
    """Numbers, then names, in `answer` that are not in `given`. Empty means grounded."""
    numbers, words = _given_numbers(given), _given_words(given)
    bad = []
    for m in _ANSWER_NUMBER.finditer(answer):
        n = m.group()
        if float(n) in numbers:
            continue
        # "the 1990s": a decade is grounded if the given text has a year in it
        if re.fullmatch(r"\d{3}0", n) and answer[m.end(): m.end() + 1] == "s" and any(int(n) <= x < int(n) + 10 for x in numbers):
            continue
        bad.append(n)
    for segment in _SEGMENT.split(answer):
        found = _WORD.findall(segment)
        for w in found[1:]:  # the first word of a sentence is capitalised anyway
            if not w[0].isupper() or len(w) < 2 or _known_word(w, words):
                continue
            if "-" in w and all(_known_word(part, words) for part in w.split("-") if part):  # "F1-related", "Grand-Prix"
                continue
            bad.append(w)
    return bad
