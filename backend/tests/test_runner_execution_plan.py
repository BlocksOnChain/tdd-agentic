"""Tests for Lead execution_plan parsing in runner."""
from __future__ import annotations

from backend.agents.runner import _parse_execution_plan, _parse_structured_summary, _format_structured_summary


SAMPLE_LEAD_JSON = """{
  "execution_plan": {
    "ticket_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "subtasks": [
      {
        "title": "Scaffold infrastructure",
        "description": "Docker, CI, env files",
        "required_functionality": "Runnable dev environment",
        "test_cases": [],
        "assigned_to": "devops",
        "order_index": 0
      },
      {
        "title": "Create JWT Service",
        "description": "Issue and verify tokens",
        "required_functionality": "JWT auth",
        "test_cases": [
          {
            "given": "valid credentials",
            "should": "return a signed JWT",
            "expected": "eyJ",
            "test_type": "unit"
          }
        ],
        "assigned_to": "backend_dev",
        "order_index": 1
      }
    ]
  }
}"""


def test_parse_execution_plan_plain_json() -> None:
    plan = _parse_execution_plan(SAMPLE_LEAD_JSON)
    assert plan is not None
    assert plan.ticket_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].title == "Scaffold infrastructure"
    assert plan.subtasks[0].assigned_to == "devops"
    assert plan.subtasks[1].assigned_to == "backend_dev"
    assert plan.subtasks[1].test_cases is not None
    assert plan.subtasks[1].test_cases[0].expected == "eyJ"


def test_parse_execution_plan_markdown_fence() -> None:
    text = f"```json\n{SAMPLE_LEAD_JSON}\n```"
    plan = _parse_execution_plan(text)
    assert plan is not None
    assert len(plan.subtasks) == 2


def test_parse_execution_plan_returns_none_on_garbage() -> None:
    assert _parse_execution_plan("just prose, no JSON") is None


def test_parse_execution_plan_strips_order_index() -> None:
    """order_index is not part of SubtaskPlan — must not break validation."""
    plan = _parse_execution_plan(SAMPLE_LEAD_JSON)
    assert plan is not None
    assert not hasattr(plan.subtasks[0], "order_index")


def test_parse_structured_summary() -> None:
    text = '{"docs_written": ["docs/arch.md"], "rag_ingested": ["arch"], "next_steps": ["plan tickets"]}'
    data = _parse_structured_summary(text)
    assert data is not None
    formatted = _format_structured_summary(data)
    assert "docs/arch.md" in formatted
    assert "RAG ingested" in formatted
    assert "plan tickets" in formatted


def test_parse_structured_summary_returns_none_on_prose() -> None:
    assert _parse_structured_summary("Finished research on FastAPI.") is None
