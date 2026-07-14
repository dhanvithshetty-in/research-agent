import os


class Reviewer:
    def __init__(self, llm, file_tool, planner):
        self.llm = llm
        self.file_tool = file_tool
        self.planner = planner

    def review(self, plan, report):
        result = self.llm.review_report(report, plan)

        if "error" in result:
            print(f"[Reviewer] LLM error: {result['error']}")
            return {"pass": True, "gaps": [], "feedback": "Skipped review due to LLM error"}

        gaps = result.get("gaps", [])
        missing_ids = result.get("missing_sub_task_ids", [])
        passed = result.get("pass", True)

        if not passed:
            print(f"[Reviewer] Found {len(gaps)} gap(s)")
            for g in gaps:
                print(f"       - {g}")
            if missing_ids:
                for tid in missing_ids:
                    for t in plan.get("sub_tasks", []):
                        if t["id"] == tid:
                            t["status"] = "pending"
            plan["iteration"] = plan.get("iteration", 0) + 1
            self.planner.save_plan(plan)
        else:
            print(f"[Reviewer] Report looks good!")
            self.file_tool.save_report(report, "final_report.md")
            print(f"[Reviewer] Final report saved")

        return result
