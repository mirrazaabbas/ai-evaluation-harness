"""Optional cross-platform LLM judging for evaluation cases."""
from __future__ import annotations

import json
from typing import Any

from ai_platform import AIClient


def judge_case(case: dict[str, Any], client: AIClient) -> str:
    if not isinstance(case, dict) or not isinstance(case.get("output"), str):
        raise ValueError("Case must contain an output string.")
    system = (
        "You are an evaluation judge. Assess the candidate output against the supplied context and "
        "requirements. Be concise, identify unsupported claims, and do not invent missing evidence."
    )
    user = json.dumps(
        {
            "output": case["output"],
            "context": case.get("context", ""),
            "required_terms": case.get("required_terms", []),
            "expected_citations": case.get("expected_citations", []),
        },
        ensure_ascii=False,
    )
    return client.generate(system, user)
