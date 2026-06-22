# -*- coding: utf-8 -*-
"""Validate the curated RAGAS evaluation set (structure only, no services)."""

import json
from pathlib import Path

EVAL_SET = Path(__file__).resolve().parents[2] / "eval" / "eval_set.json"


def _load():
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


def test_eval_set_has_two_documents_and_80_samples():
    data = _load()
    groups = data["groups"]
    assert len(groups) == 2
    assert {"康宁终身保险条款", "平安短期综合意外伤害保险条款"} == {g["document"] for g in groups}
    assert sum(len(g["samples"]) for g in groups) == 80


def test_every_sample_has_question_ground_truth_and_clause():
    for group in _load()["groups"]:
        assert group["collection"]
        assert len(group["samples"]) == 40
        for sample in group["samples"]:
            assert sample["question"].strip()
            assert sample["ground_truth"].strip()
            assert sample["clause"].strip()


def test_questions_are_unique():
    questions = [s["question"] for g in _load()["groups"] for s in g["samples"]]
    assert len(questions) == len(set(questions))
