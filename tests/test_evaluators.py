import pytest

from vtb.core.registry import EVALUATORS
from vtb.evaluators.mme import parse_pred
from vtb.evaluators.multiple_choice import parse_choice
from vtb.evaluators.vqa_accuracy import normalize_answer, vqa_score


@pytest.mark.parametrize("raw,norm", [
    ("Two", "2"), ("the dog.", "dog"), ("Yes!", "yes"), ("dont", "don't"),
    ("3.5", "3.5"), ("a red car", "red car"), ("1,000", "1000"),
])
def test_vqa_normalize(raw, norm):
    assert normalize_answer(raw) == norm


def test_vqa_leave_one_out():
    gts = ["dakota"] * 2 + ["other"] * 8
    assert vqa_score("dakota", gts) == pytest.approx(0.6)  # (2 * 1/3 + 8 * 2/3) / 10
    gts = ["dakota"] * 4 + ["other"] * 6
    assert vqa_score("dakota", gts) == pytest.approx(1.0)
    assert vqa_score("nope", gts) == 0.0


def test_vqa_three_matches_exact():
    gts = ["dakota"] * 3 + ["other"] * 7
    # removing a matching answer leaves 2 -> 2/3 (3 times); otherwise 3 -> 1 (7 times)
    assert vqa_score("dakota", gts) == pytest.approx((3 * 2 / 3 + 7) / 10)


@pytest.mark.parametrize("text,letter", [
    ("B", "B"), ("B.", "B"), ("(C) a cat", "C"), ("A: dog", "A"),
    ("The answer is D.", "D"), ("Answer: b", "B"), ("dog", "B"), ("I am not sure", None),
])
def test_parse_choice(text, letter):
    assert parse_choice(text, ["cat", "dog", "bird", "fish"]) == letter


def test_parse_choice_respects_option_count():
    assert parse_choice("D", ["x", "y"]) is None


def test_multiple_choice_circular():
    rec = lambda out, ans, g: {"output": out, "answers": [ans], "choices": ["x", "y", "z"],  # noqa: E731
                               "meta": {"circular_group": g, "category": "c"}}
    recs = [rec("A", "A", 1), rec("B", "B", 1), rec("A", "A", 2), rec("A", "C", 2)]
    s = EVALUATORS.build("multiple_choice").score(recs)
    assert s["accuracy"] == pytest.approx(75.0)
    assert s["circular_accuracy"] == pytest.approx(50.0)


@pytest.mark.parametrize("text,ans", [("Yes", "yes"), ("no.", "no"), ("Yes, it is.", "yes"), ("Maybe", "other")])
def test_mme_parse(text, ans):
    assert parse_pred(text) == ans


def test_mme_scores():
    rec = lambda out, ans, cat, img: {"output": out, "answers": [ans], "meta": {"category": cat, "image_id": img}}  # noqa: E731
    recs = [
        rec("Yes", "Yes", "existence", "1"), rec("No", "No", "existence", "1"),   # both right
        rec("Yes", "Yes", "existence", "2"), rec("Yes", "No", "existence", "2"),  # one right
        rec("Yes", "Yes", "code_reasoning", "3"), rec("No", "No", "code_reasoning", "3"),
    ]
    ev = EVALUATORS.build("mme")
    s = ev.score(recs)
    assert s["existence"] == pytest.approx(75 + 50)
    assert s["perception"] == pytest.approx(125)
    assert s["cognition"] == pytest.approx(200)
    h = ev.headline(s)
    assert h["primary_metric"] == "perception" and h["score_pct"] == pytest.approx(125 / 20)
    assert ev.headline(s, "total")["score"] == pytest.approx(325)


def test_exact_match():
    recs = [{"output": "Yes.", "answers": ["yes"], "meta": {}}, {"output": "red", "answers": ["blue"], "meta": {}}]
    assert EVALUATORS.build("exact_match").score(recs)["accuracy"] == pytest.approx(50.0)
