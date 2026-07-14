import json
import os


class ResearchPlanner:
    def __init__(self, llm, workspace_dir):
        self.llm = llm
        self.workspace_dir = workspace_dir
        self.plan_path = os.path.join(workspace_dir, "plan.json")

    def create_plan(self, query):
        result = self.llm.generate_plan(query)
        if "error" in result:
            print(f"[Planner] LLM error: {result['error']}")
            result = self._fallback_plan(query)

        result["iteration"] = 0
        result["max_iterations"] = 3
        for task in result.get("sub_tasks", []):
            task["status"] = "pending"
            task["findings_file"] = f"sub_task_{task['id']}.md"

        self._save_plan(result)
        print(f"[Planner] Created {len(result.get('sub_tasks', []))} sub-tasks")
        for t in result["sub_tasks"]:
            print(f"       #{t['id']} [{t['tool']}] {t['question'][:80]}")
        return result

    def load_plan(self):
        if os.path.exists(self.plan_path):
            with open(self.plan_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_plan(self, plan):
        self._save_plan(plan)

    def _save_plan(self, plan):
        os.makedirs(self.workspace_dir, exist_ok=True)
        with open(self.plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

    def get_pending_tasks(self, plan):
        return [t for t in plan.get("sub_tasks", []) if t["status"] == "pending"]

    def mark_done(self, plan, task_id):
        for t in plan.get("sub_tasks", []):
            if t["id"] == task_id:
                t["status"] = "completed"
                break
        self.save_plan(plan)

    def _fallback_plan(self, query):
        return {
            "original_query": query,
            "query": query,
            "iteration": 0,
            "max_iterations": 3,
            "sub_tasks": [
                {"id": 1, "question": query, "tool": "web_search",
                 "rationale": "Main research question", "status": "pending",
                 "findings_file": "sub_task_1.md"},
            ],
        }
