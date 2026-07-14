import os
import sys
import io
import contextlib
import traceback


class WebSearchTool:
    def __init__(self):
        self._ddgs = None

    def _get_ddgs(self):
        if self._ddgs is None:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                print("       [WARN] duckduckgo_search not installed. Install with: pip install duckduckgo_search")
                return None
        return self._ddgs

    def search(self, query, max_results=5):
        ddgs = self._get_ddgs()
        if ddgs is None:
            return f"[WebSearch unavailable -- duckduckgo_search not installed]"

        try:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "[No results found]"

            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("body", "")
                link = r.get("href", "")
                lines.append(f"### Result {i}: {title}")
                lines.append(f"Source: {link}")
                lines.append(body)
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            return f"[WebSearch error: {e}]"


class CodeExecTool:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def execute(self, code):
        output = io.StringIO()
        error = None
        try:
            namespace = {
                "__builtins__": __builtins__,
                "print": print,
            }
            with contextlib.redirect_stdout(output):
                exec(code, namespace)
        except Exception:
            error = traceback.format_exc()

        result = output.getvalue()
        if error:
            return f"[Execution Output]\n{result}\n[Error]\n{error}"
        return f"[Execution Output]\n{result}" if result else "[Code executed -- no output]"


class FileTool:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir

    def save_finding(self, task_id, content):
        findings_dir = os.path.join(self.workspace_dir, "findings")
        os.makedirs(findings_dir, exist_ok=True)
        path = os.path.join(findings_dir, f"sub_task_{task_id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def read_findings(self):
        findings_dir = os.path.join(self.workspace_dir, "findings")
        if not os.path.isdir(findings_dir):
            return {}
        files = sorted(os.listdir(findings_dir))
        result = {}
        for fname in files:
            if fname.endswith(".md"):
                path = os.path.join(findings_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    result[fname] = f.read()
        return result

    def save_report(self, content, filename):
        path = os.path.join(self.workspace_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def read_report(self, filename):
        path = os.path.join(self.workspace_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def build_thinking_context(self, plan, last_n=3):
        context_parts = []
        context_parts.append("## Current Plan")
        context_parts.append(f"Iteration: {plan.get('iteration', 0)}/{plan.get('max_iterations', 3)}")
        context_parts.append(f"Query: {plan.get('original_query', plan.get('query', ''))}")
        context_parts.append("")
        context_parts.append("### Sub-Tasks")
        for t in plan.get("sub_tasks", []):
            status_icon = "OK" if t["status"] == "completed" else "O"
            context_parts.append(f"- {status_icon} #{t['id']} [{t['tool']}] {t['question'][:100]}")

        context_parts.append("")
        context_parts.append("### Findings from Recent Sub-Tasks")
        findings = self.read_findings()
        completed = [t for t in plan.get("sub_tasks", []) if t["status"] == "completed"]
        recent = completed[-last_n:] if len(completed) > last_n else completed
        for t in recent:
            fname = t.get("findings_file", f"sub_task_{t['id']}.md")
            content = findings.get(fname, "")
            if content:
                context_parts.append(f"\n--- #{t['id']}: {t['question'][:60]} ---")
                context_parts.append(content[:1500])

        return "\n".join(context_parts)
