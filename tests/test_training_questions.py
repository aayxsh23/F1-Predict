"""Question-bank de-duplication (training/questions.py): exact and near
duplicates are dropped, including against the held-out eval set, so the model
is never tested on a question it was trained on. No AWS, no model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.questions import dedupe


def _q(*texts):
    return [{"question": t} for t in texts]


def test_exact_and_near_duplicates_are_dropped():
    kept = dedupe(_q("Who won the 1988 Monaco GP?", "who won the 1988 monaco gp", "Who won the 1989 Monaco GP?", "Who won the Monaco GP in 1988?"))
    assert [k["question"] for k in kept] == ["Who won the 1988 Monaco GP?", "Who won the 1989 Monaco GP?"]


def test_nothing_overlapping_the_eval_set_survives():
    kept = dedupe(_q("How many races did Ayrton Senna win?", "How many poles did Senna get?"), against=_q("how many races did ayrton senna win"))
    assert [k["question"] for k in kept] == ["How many poles did Senna get?"]
