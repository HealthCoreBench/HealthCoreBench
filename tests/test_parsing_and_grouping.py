"""Unit tests for multiple-choice parsing, evaluators, and grouping."""

from healthcorebench.benchmarks.answer_parsing import (
    parse_label,
    parse_multiple_choice_letter,
    parse_multiple_choice_letters,
    parse_yes_no_maybe,
)
from healthcorebench.evaluators import get_evaluator
from healthcorebench.aggregation.grouping import group_scores


def test_mc_parse_edge_cases():
    assert parse_multiple_choice_letter("The answer is B", ["A", "B", "C", "D"]) == "B"
    assert parse_multiple_choice_letter("\\boxed{C}", ["A", "B", "C", "D"]) == "C"
    assert parse_multiple_choice_letter("D", ["A", "B", "C", "D"]) == "D"
    assert parse_multiple_choice_letter("A) because ...", ["A", "B", "C", "D"]) == "A"
    # option text mentioning other letters should not fool it when a marker exists
    assert parse_multiple_choice_letter("Answer: A", ["A", "B", "C", "D"]) == "A"
    # undecidable -> None (never a guess)
    assert parse_multiple_choice_letter("I am not sure about this question", ["A", "B", "C", "D"]) is None


def test_multi_answer_parse():
    V = ["A", "B", "C", "D", "E"]
    # single answer degrades to a one-element list
    assert parse_multiple_choice_letters("B", V) == ["B"]
    assert parse_multiple_choice_letters("D", V) == ["D"]
    # separated multi-answers, various forms
    assert parse_multiple_choice_letters("B, D", V) == ["B", "D"]
    assert parse_multiple_choice_letters("The answer is A and C.", V) == ["A", "C"]
    assert parse_multiple_choice_letters("answers: A,C,E", V) == ["A", "C", "E"]
    assert parse_multiple_choice_letters("answer is (A) and (D)", V) == ["A", "D"]
    assert parse_multiple_choice_letters("The correct options are B, C and E.", V) == ["B", "C", "E"]
    # dense boxed run
    assert parse_multiple_choice_letters("\\boxed{BD}", V) == ["B", "D"]
    # joiner/prose words must not leak their own letters (and->a,d ; because->b,c)
    assert parse_multiple_choice_letters("The answer is A because B is clearly wrong", V) == ["A"]
    # Chinese joiners and separators
    assert parse_multiple_choice_letters("正确答案是 A、C", V) == ["A", "C"]
    assert parse_multiple_choice_letters("答案：A和C", V) == ["A", "C"]
    # undecidable -> None
    assert parse_multiple_choice_letters("I am not sure", V) is None


def test_multi_answer_evaluator_set_match():
    ev = get_evaluator("multiple_answer")

    def correct(pred, ref):
        norm = ev.normalize(pred, {})
        return ev.score(norm, {"reference_answer_normalized": ref})[2]

    assert correct(["B", "D"], "D,B") is True       # order-independent set equality
    assert correct(["B", "D"], "B,D") is True
    assert correct(["B"], "B,D") is False            # missing a required letter
    assert correct(["B", "D", "E"], "B,D") is False  # extra letter
    # a parse failure is not a wrong answer: it is unscorable, so it leaves the denominator
    # instead of entering it as a zero (was `is False`, which conflated the two).
    assert correct(None, "B") is None
    # parse_failed flag distinguishes None from confidently-wrong
    _, _, _, parsed = ev.score(ev.normalize(None, {}), {"reference_answer_normalized": "B"})
    assert parsed["parse_failed"] is True
    assert parsed["unscorable_reason"] == "unparsed_answer"


def test_numeric_tolerance_evaluator():
    ev = get_evaluator("numeric_tolerance")

    def correct(pred, ref, lo=None, hi=None, ot=None):
        s = {"reference_answer_normalized": ref,
             "metadata": {"lower_limit": lo, "upper_limit": hi, "output_type": ot}}
        return ev.score(ev.normalize(pred, s), s)[2]

    # accepted range
    assert correct("25.2", "25.2381", "23.97619", "26.50001") is True
    assert correct("Answer: 25.0", "25.2381", "23.97619", "26.50001") is True  # number extracted from text
    assert correct("30", "25.2381", "23.97619", "26.50001") is False           # outside range
    # relative tolerance fallback (default 5%) when no explicit range
    assert correct("52", "50") is True
    assert correct("60", "50") is False
    # date-typed exact match
    assert correct("01/02/2020", "01/02/2020", ot="date") is True
    assert correct("01/03/2020", "01/02/2020", ot="date") is False
    # unparseable numeric prediction -> incorrect + flagged
    s = {"reference_answer_normalized": "50", "metadata": {}}
    _, _, ok, parsed = ev.score(ev.normalize("no idea", s), s)
    assert ok is False and parsed["parse_failed"] is True


def test_likert_credit_evaluator():
    ev = get_evaluator("likert_credit")
    # expert credit distribution (max-normalized in the evaluator): modal option is "1"
    credit = {"-2": 0.0, "-1": 0.4, "0": 0.8, "1": 1.0, "2": 0.2}

    def credit_of(pick):
        s = {"metadata": {"credit": credit}}
        raw, norm, correct, parsed = ev.score(ev.normalize(pick, s), s)
        return raw, correct

    assert credit_of(1) == (1.0, True)     # modal expert option -> full credit, "correct"
    assert credit_of(0) == (0.8, False)    # partial credit
    assert credit_of(-1) == (0.4, False)
    assert credit_of(-2) == (0.0, False)
    # unparseable -> unscorable, not zero credit: a response that never named an option is
    # not evidence that the model picked the least-endorsed one.
    s = {"metadata": {"credit": credit}}
    raw, norm, correct, parsed = ev.score(ev.normalize(None, s), s)
    assert (raw, norm, correct) == (None, None, None)
    assert parsed["unscorable_reason"] == "unparsed_answer"
    # parse_failed is still carried: summarize.py's legacy branch counts num_parsing_errors off it.
    assert parsed["parse_failed"] is True


def test_parse_label():
    from healthcorebench.benchmarks.answer_parsing import parse_label
    labels = ["entailment", "contradiction", "neutral"]
    aliases = {"entails": "entailment", "contradicts": "contradiction"}
    assert parse_label("entailment", labels, aliases) == "entailment"
    assert parse_label("The answer is contradiction.", labels, aliases) == "contradiction"
    assert parse_label("This clearly contradicts the premise", labels, aliases) == "contradiction"
    assert parse_label("neutral", labels, aliases) == "neutral"
    # ambiguous (two labels present) -> None, never a guess
    assert parse_label("could be entailment or contradiction", labels, aliases) is None
    assert parse_label("no idea here", labels, aliases) is None


def test_parsers_accept_reasoning_followed_by_final_answer():
    reasoning = "A is unlikely.\n\nTherefore, the final answer is:\n\nB. definitive option"
    assert parse_multiple_choice_letter(reasoning, ["A", "B", "C", "D"]) == "B"
    diagnosis = (
        "The findings could be cholecystitis or pancreatitis. "
        "The most likely diagnosis is **appendicitis**."
    )
    assert parse_label(
        diagnosis, ["appendicitis", "cholecystitis", "pancreatitis", "diverticulitis"]
    ) == "appendicitis"


def test_discrete_parsers_prefer_answer_after_closed_thinking_block():
    reasoning = "A may fit, while B and C are also discussed."
    assert parse_multiple_choice_letter(
        f"{reasoning}\n</think>\n\nB", list("ABCD")
    ) == "B"
    assert parse_multiple_choice_letters(
        f"{reasoning}\n</think>\n\nB, D", list("ABCD")
    ) == ["B", "D"]
    assert parse_yes_no_maybe("Both yes and no were considered.\n</think>\nNo") == "no"
    assert parse_label(
        "kidney and liver were compared.\n</think>\nliver", ["kidney", "liver"]
    ) == "liver"
    assert parse_multiple_choice_letter("A was considered.\n</think>\n", list("ABCD")) is None


def test_mcqa_parser_accepts_a_bare_final_line_without_thinking_delimiter():
    response = (
        "A, B, and C were considered in the analysis.\n"
        "The requested final answer follows.\n\nB"
    )
    assert parse_multiple_choice_letter(response, list("ABCD")) == "B"


def test_grouping():
    rows = [
        {"value": "Easy", "score": 1.0, "correct": True},
        {"value": "Easy", "score": 0.0, "correct": False},
        {"value": "Hard", "score": 1.0, "correct": True},
        {"value": None, "score": 1.0, "correct": True},
    ]
    g = group_scores(rows, "difficulty")
    assert g["Easy"]["n"] == 2 and abs(g["Easy"]["score"] - 0.5) < 1e-9
    assert g["Hard"]["score"] == 1.0
    assert "__unspecified__" in g
