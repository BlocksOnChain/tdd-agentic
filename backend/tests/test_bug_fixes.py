"""Tests for the three bug fixes.

Bug 1: PM/leads must create infrastructure subtasks (devops scaffolding)
Bug 2: PM/leads must create multiple detailed tickets (granularity)
Bug 3: Dev agents must handle test failures properly (diagnosis)
"""
import pytest


# ====================================================================
# Bug 1: Infrastructure scaffolding in PM/lead prompts
# ====================================================================


class TestBug1InfrastructureScaffolding:
    """Verify the PM and lead prompts mandate infrastructure scaffolding."""

    def test_pm_prompt_has_infrastructure_step(self):
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        # PM must have a step that dispatches devops for infrastructure
        assert "devops" in PROJECT_MANAGER_SYSTEM
        assert "scaffold" in PROJECT_MANAGER_SYSTEM.lower()

        # Must mention specific infrastructure deliverables
        assert "package.json" in PROJECT_MANAGER_SYSTEM
        assert "Dockerfile" in PROJECT_MANAGER_SYSTEM
        assert ".env.example" in PROJECT_MANAGER_SYSTEM

        # Must mention npm install / pip install
        assert "npm install" in PROJECT_MANAGER_SYSTEM
        assert "pip install" in PROJECT_MANAGER_SYSTEM

    def test_pm_prompt_parallel_dispatch(self):
        """PM must dispatch TWO agents in parallel at step 1."""
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        assert "TWO agents in parallel" in PROJECT_MANAGER_SYSTEM

    def test_pm_prompt_infrastructure_as_readiness_gate(self):
        """PM must not move ticket to TODO without infrastructure."""
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        # The PM must have infrastructure as a readiness check
        assert "infrastructure" in PROJECT_MANAGER_SYSTEM.lower()

    def test_lead_infrastructure_first(self):
        """Lead must create infrastructure subtask at order_index 0 via devops."""
        from backend.agents.prompts import LEAD_SYSTEM

        assert "infrastructure" in LEAD_SYSTEM.lower()
        assert "order_index" in LEAD_SYSTEM
        assert "devops" in LEAD_SYSTEM

    def test_lead_covers_both_domains(self):
        """Merged lead must cover both backend and frontend infrastructure."""
        from backend.agents.prompts import LEAD_SYSTEM

        assert "infrastructure" in LEAD_SYSTEM.lower()
        assert "backend" in LEAD_SYSTEM.lower()
        assert "frontend" in LEAD_SYSTEM.lower()

    def test_devops_system_prompt_expands_deliverables(self):
        """DevOps system prompt must list specific scaffolding deliverables."""
        from backend.agents.prompts import DEVOPS_SYSTEM

        assert "package.json" in DEVOPS_SYSTEM
        assert "Dockerfile" in DEVOPS_SYSTEM
        assert "docker-compose" in DEVOPS_SYSTEM
        assert ".env.example" in DEVOPS_SYSTEM
        assert "CI" in DEVOPS_SYSTEM


# ====================================================================
# Bug 2: Multiple detailed tickets / granularity constraints
# ====================================================================


class TestBug2TicketGranularity:
    """Verify the PM and lead prompts force fine-grained decomposition."""

    def test_pm_prompt_forces_multiple_tickets(self):
        """PM must explicitly say NEVER bundle into one ticket."""
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        assert "NEVER bundle" in PROJECT_MANAGER_SYSTEM
        assert "MULTIPLE" in PROJECT_MANAGER_SYSTEM

    def test_pm_prompt_minimum_ticket_count(self):
        """PM must have a minimum ticket count."""
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        assert "8-12" in PROJECT_MANAGER_SYSTEM

    def test_pm_prompt_has_granularity_constraints(self):
        """PM CONSTRAINTS must include granularity rules."""
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        constraints = PROJECT_MANAGER_SYSTEM
        # Must mention minimum subtask count
        assert "Minimum subtask count" in constraints
        # Must mention maximum subtask size
        assert "2 hours" in constraints

    def test_pm_review_step_checks_decomposition(self):
        """PM step 5 must review subtask count per ticket."""
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM

        # PM must check for minimum subtasks
        assert "4 for full-stack" in PROJECT_MANAGER_SYSTEM or ">= 4" in PROJECT_MANAGER_SYSTEM
        assert "3 for single-domain" in PROJECT_MANAGER_SYSTEM or ">= 3" in PROJECT_MANAGER_SYSTEM

    def test_lead_minimum_subtasks(self):
        """Merged lead must have minimum subtask count constraint and decomposition guidance."""
        from backend.agents.prompts import LEAD_SYSTEM

        assert "Minimum subtask count" in LEAD_SYSTEM
        assert "4" in LEAD_SYSTEM
        assert "decompose further" in LEAD_SYSTEM.lower()

    def test_supervisor_enforces_min_subtasks(self):
        """_ticket_ready_for_todo must enforce minimum subtask count (via source inspection)."""
        import pathlib
        supervisor_path = pathlib.Path(__file__).parent.parent / "agents" / "project_manager" / "supervisor.py"
        source = supervisor_path.read_text()
        assert "len(subs) < 4" in source
        assert "len(subs) < 3" in source


# ====================================================================
# Bug 3: Dev agents must handle test failures properly
# ====================================================================


class TestBug3TestFailureHandling:
    """Verify dev prompts include test failure diagnosis instructions."""

    def test_dev_prompt_has_failure_handling(self):
        """Dev system prompt must have a TEST FAILURE HANDLING section."""
        from backend.agents.prompts import DEV_SYSTEM_BASE

        assert "TEST FAILURE HANDLING" in DEV_SYSTEM_BASE
        assert "MUST NOT ignore it" in DEV_SYSTEM_BASE

    def test_dev_prompt_has_shell_run_diagnosis(self):
        """Dev prompt must instruct using shell_run for diagnosis."""
        from backend.agents.prompts import DEV_SYSTEM_BASE

        assert "shell_run" in DEV_SYSTEM_BASE
        assert "node --version" in DEV_SYSTEM_BASE
        assert "npm --version" in DEV_SYSTEM_BASE
        assert "which" in DEV_SYSTEM_BASE
        assert "npm install" in DEV_SYSTEM_BASE

    def test_dev_prompt_has_repeated_failure_stop(self):
        """Dev prompt must stop after 3 identical failures."""
        from backend.agents.prompts import DEV_SYSTEM_BASE

        assert "3+ times" in DEV_SYSTEM_BASE
        assert "STOP" in DEV_SYSTEM_BASE

    def test_dev_prompt_has_environment_check_steps(self):
        """Dev prompt must check environment before running tests."""
        from backend.agents.prompts import DEV_SYSTEM_BASE

        assert "command not found" in DEV_SYSTEM_BASE.lower()
        assert "diagnose" in DEV_SYSTEM_BASE.lower()

    def test_dev_prompt_never_ignore_errors(self):
        """Dev prompt must say NEVER ignore test runner errors."""
        from backend.agents.prompts import DEV_SYSTEM_BASE

        assert "NEVER ignore" in DEV_SYSTEM_BASE

    def test_all_dev_roles_include_failure_handling(self):
        """All dev role prompts must include failure handling (inherited from base)."""
        from backend.agents.prompts import (
            BACKEND_DEV_SYSTEM,
            FRONTEND_DEV_SYSTEM,
            DEVOPS_SYSTEM,
        )

        for prompt_name, prompt in [
            ("BACKEND_DEV_SYSTEM", BACKEND_DEV_SYSTEM),
            ("FRONTEND_DEV_SYSTEM", FRONTEND_DEV_SYSTEM),
            ("DEVOPS_SYSTEM", DEVOPS_SYSTEM),
        ]:
            assert "TEST FAILURE HANDLING" in prompt, f"{prompt_name} missing failure handling"
            assert "shell_run" in prompt, f"{prompt_name} missing shell_run diagnosis"


# ====================================================================
# Regression: Verify existing functionality not broken
# ====================================================================


class TestRegressionNoRegresion:
    """Ensure the fixes don't break existing prompt behavior."""

    def test_pm_still_has_routing_protocol(self):
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM
        assert "next_agent" in PROJECT_MANAGER_SYSTEM
        assert "rationale" in PROJECT_MANAGER_SYSTEM

    def test_pm_still_has_resume_safety(self):
        from backend.agents.prompts import PROJECT_MANAGER_SYSTEM
        assert "Resume safety" in PROJECT_MANAGER_SYSTEM
        assert "list_tickets" in PROJECT_MANAGER_SYSTEM

    def test_leads_still_have_rite_contract(self):
        from backend.agents.prompts import LEAD_PLANNING_APPENDIX
        assert "RITE TEST-CASE FORMAT" in LEAD_PLANNING_APPENDIX
        assert "given" in LEAD_PLANNING_APPENDIX
        assert "should" in LEAD_PLANNING_APPENDIX
        assert "expected" in LEAD_PLANNING_APPENDIX

    def test_lead_has_scope_definition(self):
        from backend.agents.prompts import LEAD_SYSTEM
        assert "backend" in LEAD_SYSTEM.lower() and "frontend" in LEAD_SYSTEM.lower()
        assert "do not call any tools" in LEAD_SYSTEM.lower() or "no tool calls" in LEAD_SYSTEM.lower() or "do not call" in LEAD_SYSTEM.lower()

    def test_dev_still_has_tdd_contract(self):
        from backend.agents.prompts import DEV_SYSTEM_BASE
        # Core TDD contract must still be present
        assert "ONE subtask per graph invocation" in DEV_SYSTEM_BASE
        assert "strict red" in DEV_SYSTEM_BASE.lower() or "FAIL" in DEV_SYSTEM_BASE
        assert "green" in DEV_SYSTEM_BASE.lower()

    def test_all_roles_loadable(self):
        from backend.agents.prompts import get_cached_role_base

        roles = [
            "project_manager", "researcher", "lead", "coordinator",
            "backend_dev", "frontend_dev", "devops", "qa",
        ]
        for role in roles:
            base = get_cached_role_base(role)
            assert isinstance(base, str)
            assert len(base) > 0
