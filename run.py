#!/usr/bin/env python3
import argparse
import os
import sys
import shutil


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent -- file-centric state architecture"
    )
    parser.add_argument(
        "query", nargs="*",
        help="Research query (e.g. \"Compare Mamba vs Transformer for long context\")"
    )
    parser.add_argument(
        "--query-file", "-f",
        help="Read query from file"
    )
    parser.add_argument(
        "--workspace", "-w", default="workspace",
        help="Workspace directory for file-centric state (default: workspace/)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clean workspace before starting"
    )
    parser.add_argument(
        "--model", default="llama-3.1-8b-instant",
        help="Groq model name (default: llama-3.1-8b-instant)"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Get query
    query = " ".join(args.query).strip()
    if not query and args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            query = f.read().strip()
    if not query:
        print("[ERROR] Provide a query as arguments or via --query-file")
        return 1

    # Setup workspace
    workspace_dir = os.path.abspath(args.workspace)
    if args.clean and os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    # Print header
    print("=" * 60)
    print("  Autonomous Research Agent -- File-Centric State")
    print("=" * 60)
    print(f"  Workspace : {workspace_dir}")
    print(f"  Model     : {args.model}")
    print(f"  Query     : {query[:80]}{'...' if len(query) > 80 else ''}")
    print("-" * 60)

    # Build agents
    from research_agent.llm_engine import ResearchLLM
    from research_agent.planner import ResearchPlanner
    from research_agent.tool_agents import WebSearchTool, CodeExecTool, FileTool
    from research_agent.synthesizer import Synthesizer
    from research_agent.reviewer import Reviewer

    llm = ResearchLLM(model_name=args.model)
    file_tool = FileTool(workspace_dir)
    planner = ResearchPlanner(llm, workspace_dir)
    web = WebSearchTool()
    code = CodeExecTool()
    synthesizer = Synthesizer(llm, file_tool)
    reviewer = Reviewer(llm, file_tool, planner)

    # Phase 1: Plan
    print("\n[1/5] Planning research decomposition...")
    plan = planner.create_plan(query)
    max_iterations = plan.get("max_iterations", 3)

    # Phase 2-4: Research loop with review
    for iteration in range(max_iterations):
        current_iter = plan.get("iteration", 0)
        print(f"\n[2/5] Research iteration {current_iter + 1}/{max_iterations}...")

        # Execute pending sub-tasks
        pending = planner.get_pending_tasks(plan)
        if not pending:
            print("       All sub-tasks completed")
        else:
            for task in pending:
                print(f"\n       Working on #{task['id']}: {task['question'][:80]}")
                context = file_tool.build_thinking_context(plan)

                if task["tool"] in ("web_search", "both"):
                    print(f"       >> Searching web...")
                    search_result = web.search(task["question"])
                    file_tool.save_finding(task["id"],
                        f"## Web Search Results\n\n{search_result}")

                if task["tool"] in ("code_exec", "both"):
                    print(f"       >> Executing code analysis...")
                    code_prompt = f"""
Generate Python code to research this question:
{task['question']}

Context from previous findings:
{context[:2000]}

Output ONLY executable Python code -- no explanation, no markdown formatting.
Use print() to show results."""
                    code_text = llm.generate_text(
                        "You are a Python code generator. Output ONLY executable code.",
                        code_prompt, temperature=0.2, max_tokens=2048
                    )
                    code_clean = code_text.replace("```python", "").replace("```", "").strip()
                    code_result = code.execute(code_clean)

                    existing = file_tool.read_findings().get(f"sub_task_{task['id']}.md", "")
                    file_tool.save_finding(task["id"],
                        f"{existing}\n\n## Code Execution\n\n```\n{code_result}\n```")

                if task["tool"] == "reasoning":
                    print(f"       >> Reasoning directly...")
                    reasoning_prompt = f"""Answer this research question directly based on your knowledge:
{task['question']}

Be specific with numbers, benchmarks, and technical details where possible.
Provide a thorough answer with clear structure."""
                    reasoning = llm.generate_text(
                        "You are a senior research scientist. Provide a thorough, direct answer.",
                        reasoning_prompt, temperature=0.3, max_tokens=2048
                    )
                    file_tool.save_finding(task["id"],
                        f"## Direct Reasoning\n\n{reasoning}")

                planner.mark_done(plan, task["id"])

        # Phase 3: Synthesize
        print(f"\n[3/5] Synthesizing findings into report...")
        report = synthesizer.synthesize(plan)

        # Phase 4: Review
        print(f"[4/5] Reviewing report for gaps...")
        review_result = reviewer.review(plan, report)

        if review_result.get("pass", False):
            print(f"\n[5/5] Research complete!")
            break
        else:
            remaining = planner.get_pending_tasks(plan)
            if remaining and current_iter < max_iterations - 1:
                print(f"       Re-planning with {len(remaining)} gap(s) to fix...")
            else:
                print(f"       Max iterations reached. Saving best effort.")
                file_tool.save_report(report, "final_report.md")
                break

    # Show result
    final_path = os.path.join(workspace_dir, "final_report.md")
    draft_path = os.path.join(workspace_dir, "draft_report.md")
    result_path = final_path if os.path.exists(final_path) else draft_path

    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.strip().split("\n")
        print("\n" + "=" * 60)
        print(f"  Final Report: {result_path}")
        print("=" * 60)
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")
    else:
        print("\n[WARN] No report file found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
