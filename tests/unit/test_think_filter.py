# -*- coding: utf-8 -*-

from stage4_generate import remove_thinking_text


def test_complete_think_block_is_removed():
    text = "<think>推理过程</think>最终答案"

    assert remove_thinking_text(text) == "最终答案"


def test_incomplete_think_tag_is_safe():
    text = "最终答案 <think>未闭合推理"

    assert remove_thinking_text(text) == "最终答案"
