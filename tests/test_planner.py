import os
import json
import tempfile
from research_agent.llm_engine import ResearchLLM
from research_agent.planner import ResearchPlanner


def test_fallback_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        planner = ResearchPlanner(None, tmpdir)
        plan = planner._fallback_plan("Test query")
        assert "sub_tasks" in plan
        assert len(plan["sub_tasks"]) == 1
        assert plan["sub_tasks"][0]["question"] == "Test query"
        assert plan["sub_tasks"][0]["status"] == "pending"


def test_save_load_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        planner = ResearchPlanner(None, tmpdir)
        plan = {"query": "test", "sub_tasks": [{"id": 1, "status": "pending"}]}
        planner.save_plan(plan)
        loaded = planner.load_plan()
        assert loaded is not None
        assert loaded["query"] == "test"


def test_get_pending_tasks():
    plan = {
        "sub_tasks": [
            {"id": 1, "status": "pending"},
            {"id": 2, "status": "completed"},
            {"id": 3, "status": "pending"},
        ]
    }
    planner = ResearchPlanner(None, "/tmp")
    pending = planner.get_pending_tasks(plan)
    assert len(pending) == 2
    assert pending[0]["id"] == 1
    assert pending[1]["id"] == 3


def test_mark_done():
    with tempfile.TemporaryDirectory() as tmpdir:
        planner = ResearchPlanner(None, tmpdir)
        plan = {"sub_tasks": [{"id": 1, "status": "pending"}]}
        planner.save_plan(plan)
        planner.mark_done(plan, 1)
        assert plan["sub_tasks"][0]["status"] == "completed"
