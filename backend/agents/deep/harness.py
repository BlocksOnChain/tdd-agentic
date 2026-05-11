"""Register Deep Agents harness profiles for tdd-agentic models."""
from __future__ import annotations

from deepagents import HarnessProfile, register_harness_profile
from deepagents._models import get_model_identifier, get_model_provider
from langchain_core.language_models.chat_models import BaseChatModel


def register_harness_exclusions_for_model(
    model: BaseChatModel,
    *,
    excluded_tools: frozenset[str],
) -> None:
    """Layer tool exclusions onto the harness profile key for this chat model.

    Keys follow Deep Agents lookup: ``provider:identifier`` (see
    ``_harness_profile_for_model``).
    """
    prov = get_model_provider(model)
    ident = get_model_identifier(model)
    if not prov or ident is None:
        register_harness_profile(
            "tdd-agentic:default",
            HarnessProfile(excluded_tools=excluded_tools),
        )
        return
    key = ident if ":" in ident else f"{prov}:{ident}"
    register_harness_profile(key, HarnessProfile(excluded_tools=excluded_tools))
