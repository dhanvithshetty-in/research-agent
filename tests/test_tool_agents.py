import os
import tempfile
from research_agent.tool_agents import CodeExecTool, FileTool


def test_code_exec_simple():
    code = CodeExecTool()
    result = code.execute("print('hello world')")
    assert "hello world" in result


def test_code_exec_with_error():
    code = CodeExecTool()
    result = code.execute("1/0")
    assert "Error" in result or "ZeroDivisionError" in result


def test_file_tool_save_finding():
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FileTool(tmpdir)
        path = ft.save_finding(1, "test content")
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "test content"


def test_file_tool_read_findings():
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FileTool(tmpdir)
        ft.save_finding(1, "content 1")
        ft.save_finding(2, "content 2")
        findings = ft.read_findings()
        assert len(findings) == 2
        assert "sub_task_1.md" in findings
        assert "sub_task_2.md" in findings


def test_file_tool_save_read_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FileTool(tmpdir)
        ft.save_report("# Report", "draft_report.md")
        assert os.path.exists(os.path.join(tmpdir, "draft_report.md"))
        content = ft.read_report("draft_report.md")
        assert content == "# Report"
        assert ft.read_report("nonexistent.md") is None


def test_build_thinking_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FileTool(tmpdir)
        plan = {
            "original_query": "Test query",
            "iteration": 0,
            "max_iterations": 3,
            "sub_tasks": [
                {"id": 1, "question": "Q1", "tool": "web_search",
                 "status": "completed", "findings_file": "sub_task_1.md"},
                {"id": 2, "question": "Q2", "tool": "code_exec",
                 "status": "pending", "findings_file": "sub_task_2.md"},
            ]
        }
        ft.save_finding(1, "## Finding 1\nContent here")
        context = ft.build_thinking_context(plan)
        assert "Test query" in context
        assert "Q1" in context
        assert "Q2" in context
        assert "Finding 1" in context
