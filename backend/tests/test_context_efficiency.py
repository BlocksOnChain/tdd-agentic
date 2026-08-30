"""Token-efficiency and truncation behaviour of the agent runtime."""
from __future__ import annotations

import json

import pytest

from backend.agents.llm import cacheable_system_message, supports_prompt_caching
from backend.agents.runner import MAX_TOOL_RESULT_CHARS, _truncate_tool_result
from backend.tools import persistence_tools, ticket_tools


class TestCompactToolResults:
    """Tool results are the highest-frequency payload; whitespace is billed."""

    def test_no_pretty_printing(self) -> None:
        payload = {"tickets": [{"id": "a", "subtasks": [{"title": "t", "order_index": 0}]}]}
        for dump in (ticket_tools._dump, persistence_tools._dump):
            out = dump(payload)
            assert "\n" not in out
            assert ": " not in out and ", " not in out
            assert json.loads(out) == payload

    def test_compact_is_materially_smaller(self) -> None:
        payload = {"tickets": [{"id": f"id-{i}", "title": "x", "status": "todo"} for i in range(20)]}
        compact = len(ticket_tools._dump(payload))
        pretty = len(json.dumps(payload, indent=2))
        assert compact < pretty * 0.7

    def test_pydantic_objects_still_serialize(self) -> None:
        from backend.agents.state import SubtaskPlan

        out = ticket_tools._dump(SubtaskPlan(title="T", assigned_to="qa"))
        assert json.loads(out)["title"] == "T"


class TestStructuralTruncation:
    """Truncation must not hand the model unparseable JSON."""

    def test_truncated_object_is_still_valid_json(self) -> None:
        payload = {
            "id": "ticket-1",
            "title": "Auth",
            "subtasks": [
                {"title": f"Subtask {i}", "description": "x" * 200} for i in range(40)
            ],
        }
        raw = json.dumps(payload, separators=(",", ":"))
        assert len(raw) > 500

        out = _truncate_tool_result(raw, 500)
        assert len(out) <= 700  # cap plus the truncation note
        parsed = json.loads(out)  # the whole point: still parseable
        assert parsed["id"] == "ticket-1"
        assert parsed["title"] == "Auth"
        assert 0 < len(parsed["subtasks"]) < 40
        assert "_truncated" in parsed

    def test_truncated_array_is_still_valid_json(self) -> None:
        raw = json.dumps([{"id": i, "blob": "y" * 100} for i in range(50)], separators=(",", ":"))
        out = _truncate_tool_result(raw, 400)
        parsed = json.loads(out)
        assert parsed["_truncated_items"] > 0
        assert len(parsed["items"]) > 0

    def test_short_results_pass_through_untouched(self) -> None:
        raw = json.dumps({"ok": True})
        assert _truncate_tool_result(raw, MAX_TOOL_RESULT_CHARS) == raw

    def test_non_json_falls_back_to_character_truncation(self) -> None:
        raw = "TOOL_ERROR: " + "z" * 5000
        out = _truncate_tool_result(raw, 100)
        assert out.startswith("TOOL_ERROR:")
        assert "truncated" in out

    def test_object_with_no_lists_falls_back_gracefully(self) -> None:
        raw = json.dumps({"content": "q" * 5000})
        out = _truncate_tool_result(raw, 200)
        assert "truncated" in out


class TestPromptCaching:
    """Anthropic gets an explicit cache breakpoint; others must not see the key."""

    def test_anthropic_system_prompt_is_marked_cacheable(self) -> None:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-sonnet-4-6", api_key="test")
        assert supports_prompt_caching(model)

        msg = cacheable_system_message("SYSTEM PROMPT", model)
        assert isinstance(msg.content, list)
        assert msg.content[0]["text"] == "SYSTEM PROMPT"
        assert msg.content[0]["cache_control"] == {"type": "ephemeral"}

    def test_openai_compatible_gets_a_plain_string(self) -> None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o", api_key="test")
        assert not supports_prompt_caching(model)

        msg = cacheable_system_message("SYSTEM PROMPT", model)
        # A local llama.cpp / LM Studio server rejects the unknown key.
        assert msg.content == "SYSTEM PROMPT"


class TestModelMemoization:
    """Each specialist turn calls llm_factory(); building a client each time
    opened a new connection pool and TLS handshake per turn."""

    def test_same_slug_and_temperature_returns_one_instance(self) -> None:
        from backend.agents.llm import get_chat_model

        a = get_chat_model("anthropic/claude-sonnet-4-6", temperature=0.0)
        b = get_chat_model("anthropic/claude-sonnet-4-6", temperature=0.0)
        assert a is b

    def test_temperature_is_part_of_the_key(self) -> None:
        from backend.agents.llm import get_chat_model

        a = get_chat_model("anthropic/claude-sonnet-4-6", temperature=0.0)
        b = get_chat_model("anthropic/claude-sonnet-4-6", temperature=0.7)
        assert a is not b

    def test_openai_roles_receive_the_output_cap(self) -> None:
        """max_output_tokens used to be popped for openai/openrouter, leaving
        every non-Anthropic role uncapped."""
        from backend.agents.llm import get_chat_model
        from backend.config import get_settings

        model = get_chat_model("openai/gpt-4o", temperature=0.0)
        assert model.max_tokens == get_settings().max_output_tokens


class TestSkillRegistryHygiene:
    def test_legacy_lead_roles_are_remapped(self, tmp_path, monkeypatch) -> None:
        """backend_lead / frontend_lead were merged into `lead`; registries
        written before the merge would otherwise reach nobody."""
        from backend.agents.skills import registry
        from backend.config import get_settings

        skills_dir = get_settings().workspace_root / "_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "registry.json").write_text(
            json.dumps(
                {
                    "skills": {
                        "old": {
                            "name": "old",
                            "description": "d",
                            "roles": ["backend_lead", "frontend_lead", "qa"],
                            "path": "/nonexistent/SKILL.md",
                        }
                    }
                }
            )
        )
        registry.invalidate_registry_cache()

        roles = registry.list_skills()[0]["roles"]
        assert "backend_lead" not in roles
        assert "frontend_lead" not in roles
        assert roles.count("lead") == 1
        assert [s["name"] for s in registry.get_skills_for_role("lead")] == ["old"]

    def test_registry_is_not_reread_when_unchanged(self, monkeypatch) -> None:
        from backend.agents.skills import registry
        from backend.config import get_settings

        skills_dir = get_settings().workspace_root / "_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "registry.json").write_text(json.dumps({"skills": {}}))
        registry.invalidate_registry_cache()

        reads = 0
        real_read = type(skills_dir).read_text

        def _counting_read(self, *a, **kw):
            nonlocal reads
            if self.name == "registry.json":
                reads += 1
            return real_read(self, *a, **kw)

        monkeypatch.setattr(type(skills_dir), "read_text", _counting_read)

        for _ in range(5):
            registry.list_skills()

        assert reads == 1, "registry.json should be parsed once, not per lookup"
