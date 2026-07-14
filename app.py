import streamlit as st
import os
import shutil
import time

st.set_page_config(
    page_title="Research Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")

st.markdown("""
<style>
    .main > div { padding: 1.5rem 2rem; }
    .stApp { background: #0f1117; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stTextArea textarea {
        background: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 8px;
        font-size: 15px;
    }
    .stTextArea textarea:focus { border-color: #58a6ff; box-shadow: 0 0 0 2px rgba(88,166,255,0.3); }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #238636, #2ea043);
        border: none;
        color: white;
    }
    .stButton button[kind="primary"]:hover { background: linear-gradient(135deg, #2ea043, #3fb950); transform: translateY(-1px); }
    div[data-testid="stExpander"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    div[data-testid="stExpander"] summary { font-weight: 600; color: #58a6ff; }
    .stCodeBlock { background: #0d1117 !important; border-radius: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #161b22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] { background: #1f2937; color: #f0f6fc; }
    div.stAlert { border-radius: 8px; border: none; }
    .stAlert.info { background: #0d1d3a; color: #58a6ff; }
    .stAlert.success { background: #0d2818; color: #3fb950; }
    .sidebar-header { margin-bottom: 1.5rem; }
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .stat-card .num { font-size: 1.5rem; font-weight: 700; color: #f0f6fc; }
    .stat-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .log-line {
        font-family: monospace;
        font-size: 0.8rem;
        padding: 2px 0;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
    }
    .log-line:last-child { border-bottom: none; }
    .log-line.active { color: #58a6ff; }
    .log-line.done { color: #3fb950; }
    .stDownloadButton button {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #e6edf3;
    }
    .stDownloadButton button:hover { background: #30363d; border-color: #8b949e; }
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #8b949e; }
    .findings-count { font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


def init_session():
    for key, default in [
        ("workspace_dir", WORKSPACE_DIR),
        ("running", False),
        ("report", None),
        ("plan", None),
        ("findings", {}),
        ("logs", []),
        ("query", ""),
        ("progress", 0),
        ("phase", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


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

    st.session_state.phase = "Planning..."
    st.session_state.progress = 5
    log("Planning research decomposition...")
    plan = planner.create_plan(query)
    n_tasks = len(plan.get("sub_tasks", []))
    log(f"Created {n_tasks} sub-tasks")
    max_iterations = plan.get("max_iterations", 3)
    base_progress = 10

    for iteration in range(max_iterations):
        current_iter = plan.get("iteration", 0)
        st.session_state.phase = f"Iteration {current_iter + 1}/{max_iterations}"
        log(f"--- Iteration {current_iter + 1}/{max_iterations} ---")

        pending = planner.get_pending_tasks(plan)
        if pending:
            for i, task in enumerate(pending):
                st.session_state.progress = base_progress + int(i / len(pending) * 60)
                log(f"#{task['id']}: {task['question'][:70]}...")
                context = file_tool.build_thinking_context(plan)

                if task["tool"] in ("web_search", "both"):
                    log(f"  >> Searching web...")
                    st.session_state.phase = f"Searching: #{task['id']}"
                    search_result = web.search(task["question"])
                    file_tool.save_finding(task["id"],
                        f"## Web Search Results\n\n{search_result}")

                if task["tool"] in ("code_exec", "both"):
                    log(f"  >> Executing code...")
                    st.session_state.phase = f"Running code: #{task['id']}"
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
                    log(f"  >> Reasoning...")
                    st.session_state.phase = f"Reasoning: #{task['id']}"
                    reasoning = llm.generate_text(
                        "You are a senior research scientist. Provide a thorough, direct answer.",
                        task["question"], temperature=0.3, max_tokens=1024
                    )
                    file_tool.save_finding(task["id"],
                        f"## Direct Reasoning\n\n{reasoning}")

                planner.mark_done(plan, task["id"])

        st.session_state.phase = "Synthesizing..."
        st.session_state.progress = base_progress + 65
        log("Synthesizing findings into report...")
        report = synthesizer.synthesize(plan)

        st.session_state.phase = "Reviewing..."
        st.session_state.progress = base_progress + 80
        log("Reviewing report for gaps...")
        review_result = reviewer.review(plan, report)

        if review_result.get("pass", False):
            log("Report passed review!")
            st.session_state.progress = 100
            st.session_state.phase = "Complete"
            break
        else:
            remaining = planner.get_pending_tasks(plan)
            if remaining and current_iter < max_iterations - 1:
                log(f"Re-planning with {len(remaining)} gap(s)...")
                base_progress += 20
            else:
                log("Max iterations reached. Saving best effort.")
                file_tool.save_report(report, "final_report.md")
                st.session_state.progress = 100
                st.session_state.phase = "Complete"
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
    st.session_state.phase = "Complete"


def main():
    init_session()

    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.markdown("##   Research Agent")
        st.markdown("*Autonomous file-centric research*")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Query")
        query = st.text_area(
            "",  # no label
            value=st.session_state.get("query", ""),
            placeholder="e.g. Compare Mamba and Transformer for long-context tasks",
            height=90,
            label_visibility="collapsed",
        )
        st.session_state.query = query

        model = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
            index=0,
        )

        col1, col2 = st.columns(2)
        with col1:
            clean = st.button(" Clear", use_container_width=True)
        with col2:
            disabled = st.session_state.running or not query
            run = st.button(
                " Run",
                type="primary",
                use_container_width=True,
                disabled=disabled,
            )

        if clean:
            if os.path.exists(WORKSPACE_DIR):
                shutil.rmtree(WORKSPACE_DIR)
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            st.session_state.report = None
            st.session_state.plan = None
            st.session_state.findings = {}
            st.session_state.logs = []
            st.session_state.progress = 0
            st.session_state.phase = ""
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
            st.session_state.progress = 0
            with st.spinner(""):
                run_pipeline(query, WORKSPACE_DIR, model)
            st.rerun()

        st.divider()

        st.markdown("### Status")
        cols = st.columns(3)
        with cols[0]:
            n_findings = len(st.session_state.findings)
            st.markdown(
                f'<div class="stat-card"><div class="num">{n_findings}</div>'
                f'<div class="label">Findings</div></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            log_count = len(st.session_state.logs)
            st.markdown(
                f'<div class="stat-card"><div class="num">{log_count}</div>'
                f'<div class="label">Steps</div></div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            has_report = "Yes" if st.session_state.report else "No"
            st.markdown(
                f'<div class="stat-card"><div class="num">{has_report}</div>'
                f'<div class="label">Report</div></div>',
                unsafe_allow_html=True,
            )

        if st.session_state.running:
            st.markdown(f"**Phase:** {st.session_state.phase}")
            st.progress(st.session_state.progress / 100)

        st.divider()
        key_status = " Set" if os.environ.get("GROQ_API_KEY") else " Not Set"
        st.markdown(f"**API Key:** {key_status}")

    tabs = st.tabs([" Report", " Findings", " Plan", " Logs"])

    with tabs[0]:
        if st.session_state.running:
            st.info(f"Research in progress — {st.session_state.phase}")
            st.progress(st.session_state.progress / 100)
        elif st.session_state.report:
            st.markdown(st.session_state.report)
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                report_path = os.path.join(WORKSPACE_DIR, "final_report.md")
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        st.download_button(
                            "Download",
                            data=f.read(),
                            file_name="research_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
            with col2:
                if st.button("Copy", use_container_width=True):
                    st.toast("Copied to clipboard!")
        else:
            st.info("Enter a query in the sidebar and click **Run** to start researching.")

    with tabs[1]:
        findings = st.session_state.findings
        if findings:
            st.markdown(f'<div class="findings-count">{len(findings)} file(s) in workspace</div>', unsafe_allow_html=True)
            for fname, content in sorted(findings.items()):
                tid = fname.replace("sub_task_", "").replace(".md", "")
                with st.expander(f" #{tid} — {fname}"):
                    st.code(content[:8000], language="markdown")
        else:
            st.info("No findings collected yet. Run a research query first.")

    with tabs[2]:
        if st.session_state.plan:
            import json
            plan = st.session_state.plan
            total = len(plan.get("sub_tasks", []))
            done = sum(1 for t in plan.get("sub_tasks", []) if t["status"] == "completed")
            st.markdown(f"**Progress:** {done}/{total} sub-tasks completed")
            st.progress(done / total if total else 0)
            st.divider()
            st.json(plan)
        else:
            st.info("No plan generated yet.")

    with tabs[3]:
        logs = st.session_state.logs
        if logs:
            n = len(logs)
            container = st.container()
            with container:
                for i, line in enumerate(logs):
                    cls = "active" if i == n - 1 and st.session_state.running else ""
                    st.markdown(
                        f'<div class="log-line {cls}">{line}</div>',
                        unsafe_allow_html=True,
                    )
            if st.session_state.running:
                st.markdown(f"*Running... step {n}*")
        else:
            st.info("No logs yet.")


if __name__ == "__main__":
    main()
