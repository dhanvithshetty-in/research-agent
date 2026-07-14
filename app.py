import streamlit as st
import os
import sys
import json
import shutil
from io import StringIO

st.set_page_config(
    page_title="Research Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")


def init_session():
    if "workspace_dir" not in st.session_state:
        st.session_state.workspace_dir = WORKSPACE_DIR
    if "running" not in st.session_state:
        st.session_state.running = False
    if "report" not in st.session_state:
        st.session_state.report = None
    if "plan" not in st.session_state:
        st.session_state.plan = None
    if "findings" not in st.session_state:
        st.session_state.findings = {}
    if "logs" not in st.session_state:
        st.session_state.logs = []


def log(msg):
    st.session_state.logs.append(msg)


def run_pipeline(query, workspace_dir, model):
    from research_agent.llm_engine import ResearchLLM
    from research_agent.planner import ResearchPlanner
    from research_agent.tool_agents import WebSearchTool, CodeExecTool, FileTool
    from research_agent.synthesizer import Synthesizer
    from research_agent.reviewer import Reviewer

    llm = ResearchLLM(model_name=model)
    file_tool = FileTool(workspace_dir)
    planner = ResearchPlanner(llm, workspace_dir)
    web = WebSearchTool()
    code = CodeExecTool()
    synthesizer = Synthesizer(llm, file_tool)
    reviewer = Reviewer(llm, file_tool, planner)

    log("Planning research decomposition...")
    plan = planner.create_plan(query)
    max_iterations = plan.get("max_iterations", 3)

    for iteration in range(max_iterations):
        current_iter = plan.get("iteration", 0)
        log(f"--- Iteration {current_iter + 1}/{max_iterations} ---")

        pending = planner.get_pending_tasks(plan)
        if not pending:
            log("All sub-tasks completed")
        else:
            for task in pending:
                log(f"Working on #{task['id']}: {task['question'][:80]}")
                context = file_tool.build_thinking_context(plan)

                if task["tool"] in ("web_search", "both"):
                    log(f"  >> Searching web...")
                    search_result = web.search(task["question"])
                    file_tool.save_finding(task["id"],
                        f"## Web Search Results\n\n{search_result}")

                if task["tool"] in ("code_exec", "both"):
                    log(f"  >> Executing code...")
                    code_prompt = f"""Generate Python code to research: {task['question']}
Context: {context[:1500]}
Output ONLY executable Python code."""
                    code_text = llm.generate_text(
                        "You are a Python code generator. Output ONLY executable code.",
                        code_prompt, temperature=0.2, max_tokens=1024
                    )
                    code_clean = code_text.replace("```python", "").replace("```", "").strip()
                    code_result = code.execute(code_clean)
                    existing = file_tool.read_findings().get(f"sub_task_{task['id']}.md", "")
                    file_tool.save_finding(task["id"],
                        f"{existing}\n\n## Code Execution\n\n```\n{code_result}\n```")

                if task["tool"] == "reasoning":
                    log(f"  >> Reasoning directly...")
                    reasoning = llm.generate_text(
                        "You are a senior research scientist. Provide a thorough, direct answer.",
                        task["question"], temperature=0.3, max_tokens=1024
                    )
                    file_tool.save_finding(task["id"],
                        f"## Direct Reasoning\n\n{reasoning}")

                planner.mark_done(plan, task["id"])

        log("Synthesizing findings into report...")
        report = synthesizer.synthesize(plan)

        log("Reviewing report for gaps...")
        review_result = reviewer.review(plan, report)

        if review_result.get("pass", False):
            log("Report passed review!")
            break
        else:
            remaining = planner.get_pending_tasks(plan)
            if remaining and current_iter < max_iterations - 1:
                log(f"Re-planning with {len(remaining)} gap(s)...")
            else:
                log("Max iterations reached. Saving best effort.")
                file_tool.save_report(report, "final_report.md")
                break

    final_path = os.path.join(workspace_dir, "final_report.md")
    draft_path = os.path.join(workspace_dir, "draft_report.md")
    result_path = final_path if os.path.exists(final_path) else draft_path

    st.session_state.plan = planner.load_plan()
    st.session_state.findings = file_tool.read_findings()
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            st.session_state.report = f.read()
    st.session_state.running = False


def main():
    init_session()

    with st.sidebar:
        st.title(" Research Agent")
        st.markdown("File-centric autonomous research (InfiAgent pattern)")

        query = st.text_area(
            "Research Query",
            value=st.session_state.get("query", ""),
            placeholder="e.g. Compare Mamba and Transformer for long-context tasks",
            height=100,
        )
        st.session_state.query = query

        model = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
            index=0,
        )

        col1, col2 = st.columns(2)
        with col1:
            clean = st.button(" Clean Workspace", use_container_width=True)
        with col2:
            run = st.button(" Run Research", type="primary", use_container_width=True)

        if clean:
            if os.path.exists(WORKSPACE_DIR):
                shutil.rmtree(WORKSPACE_DIR)
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            st.session_state.report = None
            st.session_state.plan = None
            st.session_state.findings = {}
            st.session_state.logs = []
            st.rerun()

        if run and query and not st.session_state.running:
            if not os.environ.get("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not set!")
                st.stop()
            st.session_state.running = True
            st.session_state.logs = []
            st.session_state.report = None
            st.session_state.plan = None
            st.session_state.findings = {}
            run_pipeline(query, WORKSPACE_DIR, model)
            st.rerun()

        st.divider()
        st.markdown("**API Key**")
        key_status = " Set" if os.environ.get("GROQ_API_KEY") else " Not Set"
        st.markdown(f"GROQ_API_KEY: {key_status}")

    tabs = st.tabs([" Report", " Findings", " Plan", " Logs"])

    with tabs[0]:
        if st.session_state.report:
            st.markdown(st.session_state.report)

            col1, col2 = st.columns([1, 4])
            with col1:
                workspace_files = os.listdir(WORKSPACE_DIR) if os.path.isdir(WORKSPACE_DIR) else []
                report_path = os.path.join(WORKSPACE_DIR, "final_report.md")
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        st.download_button(
                            " Download Report",
                            data=f.read(),
                            file_name="research_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
        else:
            if st.session_state.running:
                st.info("Research in progress...")
            else:
                st.info("Enter a query and click Run Research to start")

    with tabs[1]:
        if st.session_state.findings:
            for fname, content in sorted(st.session_state.findings.items()):
                with st.expander(f" {fname}"):
                    st.text(content[:5000])
        else:
            st.info("No findings yet")

    with tabs[2]:
        if st.session_state.plan:
            st.json(st.session_state.plan)
        else:
            st.info("No plan yet")

    with tabs[3]:
        if st.session_state.logs:
            for line in st.session_state.logs:
                st.text(line)
            if st.session_state.running:
                st.info("Research in progress...")
        else:
            st.info("No logs yet")


if __name__ == "__main__":
    main()
