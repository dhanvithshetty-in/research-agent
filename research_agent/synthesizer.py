class Synthesizer:
    def __init__(self, llm, file_tool):
        self.llm = llm
        self.file_tool = file_tool

    def synthesize(self, plan):
        findings = self.file_tool.read_findings()
        if not findings:
            return "# No findings collected\n\nThe research completed without producing findings."

        sub_tasks = {t["id"]: t for t in plan.get("sub_tasks", [])}
        context_parts = []
        context_parts.append(f"# Research Query\n{plan.get('original_query', '')}\n")

        for fname in sorted(findings.keys()):
            task_id = None
            for t_id, task in sub_tasks.items():
                if task.get("findings_file") == fname:
                    task_id = t_id
                    break
            question = sub_tasks[task_id]["question"] if task_id and task_id in sub_tasks else fname
            content = findings[fname]
            if len(content) > 1000:
                content = content[:1000] + "\n...[truncated]"
            context_parts.append(f"\n## {question}")
            context_parts.append(content)

        full_context = "\n".join(context_parts)
        report = self.llm.synthesize_report(full_context)

        self.file_tool.save_report(report, "draft_report.md")
        print(f"[Synthesizer] Draft report written")

        return report
