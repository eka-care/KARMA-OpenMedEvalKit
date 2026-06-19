"""
Shared MCQA answer-extraction helpers.

This module intentionally has zero imports from sibling dataset modules so
that MCQA dataset classes can import ``extract_wrapped_mcq_answer`` without
creating a cross-dataset dependency graph. Only stdlib imports are allowed.
"""

from typing import Tuple
import re

MCQ_FINAL_ANSWER_EDGE_CHARS = " \t\r\n*_`~"
MCQ_FINAL_ANSWER_WRAPPER_PAIRS = {"(": ")", "[": "]"}


def extract_wrapped_mcq_answer(
    response: str, valid_letters: str = "ABCDE"
) -> Tuple[str, bool]:
    """
    Extract a single MCQ option using minimal edge stripping.

    Bedrock responses sometimes render the requested final answer as variants
    like ``**(A)**`` or ``** (C)`` instead of a bare ``(A)``. Some residual
    outputs also repeat the ``Final Answer:`` marker on the next line before
    the wrapped option. This helper strips leading and trailing markdown
    emphasis, unwraps one surrounding bracket pair when present, tolerates one
    duplicated leading ``Final Answer:`` marker, and then accepts only a
    single answer letter.

    Args:
        response: Raw model response text.
        valid_letters: Allowed answer letters, case-insensitive. Defaults to
            ``"ABCDE"`` so callers relying on 5-option datasets (e.g. NFI,
            MedXpertQA where letters extend past D) keep working; callers with
            stricter alphabets can pass a shorter string.

    Returns:
        ``(letter, success)`` where ``letter`` is an uppercase single
        character from ``valid_letters`` on success, or ``("", False)`` when
        no valid answer could be parsed.
    """
    raw_answer = (
        response.split("Final Answer:", 1)[1].strip()
        if "Final Answer:" in response
        else response.strip()
    )
    if not raw_answer:
        return "", False

    candidate = ""
    for line in raw_answer.splitlines():
        candidate = line.strip(MCQ_FINAL_ANSWER_EDGE_CHARS)
        while candidate.startswith("Final Answer:"):
            candidate = candidate[len("Final Answer:") :].strip(
                MCQ_FINAL_ANSWER_EDGE_CHARS
            )
        if candidate:
            break

    if not candidate:
        return "", False

    expected_closer = MCQ_FINAL_ANSWER_WRAPPER_PAIRS.get(candidate[:1])
    if expected_closer and candidate.endswith(expected_closer):
        candidate = candidate[1:-1].strip(MCQ_FINAL_ANSWER_EDGE_CHARS)

    if len(candidate) == 1:
        answer = candidate.upper()
        if answer in valid_letters.upper():
            return answer, True

    letter_match = re.match(
        rf"^([{re.escape(valid_letters)}])[\.\)\:\s]",
        candidate,
        re.IGNORECASE,
    )
    if letter_match:
        return letter_match.group(1).upper(), True

    return "", False
