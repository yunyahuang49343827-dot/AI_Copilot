"""Streamlit UI for the WITS Governance, Compliance & Financial Control Copilot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from frontend.api_client import (
    BACKEND_UNREACHABLE_MESSAGE,
    approve_agent_run,
    ask_qa,
    check_health,
    create_case,
    get_audit_log,
    get_agent_run,
    get_case,
    get_workflow_advice,
    list_audit_logs,
    list_agent_runs,
    list_cases,
    reject_agent_run,
    start_agent_run,
)
from frontend.evaluation_examples import EVALUATION_EXAMPLES, evaluate_response, summarize_results


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEMO_QUERIES = [example["query"] for example in EVALUATION_EXAMPLES]
LOCAL_DEMO_DISCLAIMER = "Local portfolio demo only. Not legal advice. This is not an official WITS system."

RISK_BADGE_COLORS = {
    "High": "red",
    "Medium": "orange",
    "Low": "green",
    "Insufficient Evidence": "blue",
}

CITATION_COLUMN_ORDER = ["citation_id", "policy_name", "article", "article_title", "pages", "source_file"]
CITATION_COLUMN_CONFIG = {
    "citation_id": st.column_config.TextColumn("Citation", width="small"),
    "policy_name": st.column_config.TextColumn("Policy", width="medium"),
    "article": st.column_config.TextColumn("Article", width="small"),
    "article_title": st.column_config.TextColumn("Article Title", width="medium"),
    "pages": st.column_config.TextColumn("Pages", width="small"),
    "source_file": st.column_config.TextColumn("Source File", width="medium"),
}

APP_STYLE = """
<style>
    #MainMenu, footer, .stAppDeployButton {visibility: hidden;}
    .block-container {padding-top: 2.4rem;}
</style>
"""

HEADER_HTML = """
<div style="background: linear-gradient(90deg, #0F2E5C 0%, #1B4F8C 100%);
            padding: 1.1rem 1.5rem; border-radius: 0.4rem; margin-bottom: 0.3rem;">
    <div style="color: #FFFFFF; font-size: 1.35rem; font-weight: 700; letter-spacing: 0.02em;">
        WITS
        <span style="font-weight: 400; opacity: 0.55; margin: 0 0.4rem;">|</span>
        <span style="font-weight: 500;">Governance, Compliance &amp; Financial Control Copilot</span>
    </div>
    <div style="color: #C9D8EC; font-size: 0.82rem; margin-top: 0.35rem;">
        Policy intelligence · Grounded citations · Risk triage · Compliance workflow
        <span style="float: right; background: rgba(255,255,255,0.14); padding: 0.1rem 0.55rem;
                     border-radius: 0.25rem; font-size: 0.75rem; letter-spacing: 0.04em;">
            INTERNAL DEMO
        </span>
    </div>
</div>
"""


def init_state() -> None:
    for key in [
        "qa_result",
        "workflow_result",
        "created_case",
        "case_list",
        "evaluation_results",
        "agent_run_result",
        "agent_run_list",
        "selected_agent_run_id",
        "agent_reject_reason",
        "audit_log_result",
        "selected_audit_id",
        "audit_event_detail",
        "audit_filter_event_type",
        "audit_filter_run_id",
        "audit_filter_case_id",
    ]:
        st.session_state.setdefault(key, None)


def show_error(result: dict) -> None:
    st.error(result.get("error") or BACKEND_UNREACHABLE_MESSAGE)


def display_source_file(source_file: str) -> str:
    if not source_file:
        return ""
    return Path(str(source_file)).name


def display_article_title(article_title: str) -> str:
    if not article_title or article_title == "TBD":
        return "No explicit title"
    return article_title


def prepare_citations_for_display(citations: list[dict] | None) -> list[dict]:
    rows = []
    for citation in citations or []:
        row = dict(citation)
        row["source_file"] = display_source_file(row.get("source_file", ""))
        row["article_title"] = display_article_title(row.get("article_title", ""))
        rows.append(row)
    return rows


def risk_badge(risk_level: str) -> str:
    level = risk_level or "Unknown"
    color = RISK_BADGE_COLORS.get(level, "gray")
    return f":{color}-badge[{level}]"


def evidence_quality_badge(evidence_quality: str) -> str:
    text = (evidence_quality or "").lower()
    if "insufficient" in text:
        return ":orange-badge[Insufficient]"
    if "sufficient" in text:
        return ":green-badge[Sufficient]"
    return ":gray-badge[Unknown]"


def _is_numeric_timing_value(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _timing_label_from_key(key: str) -> str:
    label = str(key)
    if label.endswith("_ms"):
        label = label[:-3]
    return label.replace("_", " ").strip()


def format_timing_metadata(timing_metadata: dict | None) -> str:
    if not timing_metadata:
        return ""

    known_keys = [
        ("retrieval_time_ms", "retrieval"),
        ("answer_time_ms", "answer"),
        ("workflow_time_ms", "workflow"),
        ("agent_run_time_ms", "agent run"),
        ("total_time_ms", "total"),
    ]
    rendered: list[str] = []
    used_keys = set()

    for key, label in known_keys:
        value = timing_metadata.get(key)
        used_keys.add(key)
        if _is_numeric_timing_value(value):
            rendered.append(f"{label}: {value:.2f} ms")

    for key, value in timing_metadata.items():
        if key in used_keys or not _is_numeric_timing_value(value):
            continue
        rendered.append(f"{_timing_label_from_key(key)}: {value:.2f} ms")

    return " · ".join(rendered)


def render_timing_metadata(timing_metadata: dict | None, label: str = "Performance metadata") -> None:
    formatted = format_timing_metadata(timing_metadata)
    if not formatted:
        return
    with st.expander(label):
        st.caption(formatted)


def format_generation_mode(mode: str | None) -> str:
    return (mode or "deterministic").strip() or "deterministic"


SAFE_LLM_METADATA_FIELDS = ["provider", "used_llm", "fallback_reason", "evidence_count", "configured"]


def safe_llm_metadata(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    return {key: metadata[key] for key in SAFE_LLM_METADATA_FIELDS if key in metadata}


def format_llm_status_message(mode: str | None, metadata: dict | None = None) -> str:
    generation_mode = format_generation_mode(mode)
    safe_metadata = safe_llm_metadata(metadata)
    fallback_reason = safe_metadata.get("fallback_reason")
    if generation_mode == "llm_grounded":
        return "LLM-assisted answer generated from retrieved evidence. Please review citations and evidence quality."
    if generation_mode == "llm_fallback":
        if fallback_reason:
            return f"LLM generation was not used or fell back to the deterministic answer. Reason: {fallback_reason}."
        return "LLM generation was not used or fell back to the deterministic answer."
    return "Generation mode: deterministic."


def render_generation_status(mode: str | None, metadata: dict | None = None) -> None:
    generation_mode = format_generation_mode(mode)
    st.caption(f"Generation mode: `{generation_mode}`")
    message = format_llm_status_message(generation_mode, metadata)
    if generation_mode == "llm_grounded":
        st.info(message)
    elif generation_mode == "llm_fallback":
        st.warning(message)

    safe_metadata = safe_llm_metadata(metadata)
    if safe_metadata:
        with st.expander("LLM metadata"):
            st.json(safe_metadata)


def render_citations(citations: list[dict] | None) -> None:
    rows = prepare_citations_for_display(citations)
    st.markdown(f"##### Policy Citations ({len(rows)})")
    if not rows:
        st.info("No citations returned.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_order=CITATION_COLUMN_ORDER,
        column_config=CITATION_COLUMN_CONFIG,
    )


def render_answer_card(title: str, body: str) -> None:
    with st.container(border=True):
        st.markdown(f"##### {title}")
        st.markdown(body or "_No answer returned._")


def render_summary_card(label: str, value_markdown: str, detail: str = "") -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(value_markdown)
        if detail:
            st.caption(detail)


def render_disclaimer(disclaimer: str) -> None:
    if disclaimer:
        st.caption(disclaimer)


def sidebar_controls() -> tuple[str, str, int]:
    with st.sidebar:
        st.markdown("**System Connection**")
        base_url = st.text_input("API base URL", value=DEFAULT_BASE_URL)
        if st.button("Check Connection", use_container_width=True):
            result = check_health(base_url)
            if result["ok"]:
                st.success(f"Connected: {result['data'].get('service', 'API')}")
            else:
                st.error(result["error"])
        st.divider()
        st.markdown("**Query Settings**")
        query = st.selectbox("Sample question", DEMO_QUERIES)
        top_k = st.slider("Evidence passages (top_k)", min_value=1, max_value=10, value=5)
        st.divider()
        st.caption(LOCAL_DEMO_DISCLAIMER)
    return base_url, query, top_k


def query_controls(demo_query: str, default_top_k: int, key_prefix: str, button_label: str) -> tuple[str, int, bool]:
    query = st.text_area("Question", value=demo_query, key=f"{key_prefix}_query")
    col_btn, col_topk = st.columns([3, 1], vertical_alignment="bottom")
    with col_topk:
        top_k = st.slider("top_k", min_value=1, max_value=10, value=default_top_k, key=f"{key_prefix}_top_k")
    with col_btn:
        submitted = st.button(button_label, type="primary")
    return query, top_k, submitted


def policy_qa_tab(base_url: str, demo_query: str, default_top_k: int) -> None:
    st.markdown("#### Policy Q&A")
    st.caption("Answers are generated from retrieved internal policy evidence and always include citations.")
    query, top_k, submitted = query_controls(demo_query, default_top_k, "qa", "Submit Question")
    use_llm = st.checkbox(
        "Use LLM-assisted grounded answer",
        value=False,
        help="Uses an optional DeepSeek-powered grounded answer layer. The answer is constrained to retrieved policy evidence; citations and evidence quality still apply.",
    )
    st.caption("LLM is used only for answer wording. It does not create cases, approve actions, change risk levels, or perform external actions.")
    if submitted:
        with st.spinner("Retrieving policy evidence..."):
            st.session_state.qa_result = ask_qa(base_url, query, top_k, use_llm=use_llm)

    result = st.session_state.qa_result
    if not result:
        return
    if not result["ok"]:
        show_error(result)
        return
    data = result["data"]
    st.divider()
    render_answer_card("Grounded Answer", data.get("answer", ""))
    render_generation_status(data.get("generation_mode"), data.get("llm_metadata"))
    render_summary_card("Evidence Quality", evidence_quality_badge(data.get("evidence_quality", "")), detail=data.get("evidence_quality", ""))
    render_citations(data.get("citations"))
    render_disclaimer(data.get("disclaimer", ""))
    render_timing_metadata(data.get("timing_metadata"))


def workflow_advice_tab(base_url: str, demo_query: str, default_top_k: int) -> None:
    st.markdown("#### Workflow Advice")
    st.caption("Risk assessment and recommended compliance steps based on internal policy documents.")
    query, top_k, submitted = query_controls(demo_query, default_top_k, "workflow", "Get Workflow Advice")
    if submitted:
        with st.spinner("Running risk assessment..."):
            st.session_state.workflow_result = get_workflow_advice(base_url, query, top_k)

    result = st.session_state.workflow_result
    if not result:
        return
    if not result["ok"]:
        show_error(result)
        return
    data = result["data"]
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_summary_card("Risk Level", risk_badge(data.get("risk_level", "")))
    with col2:
        render_summary_card("Risk Category", f"**{data.get('risk_category', '') or 'Unknown'}**")
    with col3:
        render_summary_card("Evidence Quality", evidence_quality_badge(data.get("evidence_quality", "")), detail=data.get("evidence_quality", ""))

    render_answer_card("Grounded Answer", data.get("grounded_answer", ""))

    with st.container(border=True):
        st.markdown("##### Risk Reasoning")
        st.markdown(data.get("risk_reasoning", "") or "_No risk reasoning returned._")

    with st.container(border=True):
        st.markdown("##### Workflow Checklist")
        checklist = data.get("workflow_checklist") or []
        if checklist:
            st.markdown("\n".join(f"{index}. {step}" for index, step in enumerate(checklist, start=1)))
        else:
            st.info("No workflow checklist returned.")

    render_citations(data.get("citations"))
    render_disclaimer(data.get("disclaimer", ""))
    render_timing_metadata(data.get("timing_metadata"))


def render_case_details(case: dict) -> None:
    with st.container(border=True):
        header_col, badge_col = st.columns([3, 1])
        with header_col:
            st.markdown(f"##### Case `{case.get('case_id', '')}`")
        with badge_col:
            st.markdown(f"{risk_badge(case.get('risk_level', ''))} :gray-badge[{case.get('status', '') or 'Unknown'}]")

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Requester**\n\n{case.get('requester', '')}")
        col2.markdown(f"**Department**\n\n{case.get('department', '')}")
        col3.markdown(f"**Created At**\n\n{case.get('created_at', '')}")

        st.markdown(f"**Risk Category:** {case.get('risk_category', '') or 'Unknown'}")
        st.markdown(f"**Evidence Quality:** {evidence_quality_badge(case.get('evidence_quality', ''))}")
        st.markdown(f"**Query:** {case.get('query', '')}")

        steps = case.get("recommended_next_steps") or []
        if steps:
            st.markdown("**Recommended Next Steps**")
            st.markdown("\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1)))

        citations = case.get("citation_summary") or []
        if citations:
            with st.expander(f"Policy Citations ({len(citations)})"):
                st.dataframe(
                    pd.DataFrame(prepare_citations_for_display(citations)),
                    use_container_width=True,
                    hide_index=True,
                    column_order=CITATION_COLUMN_ORDER,
                    column_config=CITATION_COLUMN_CONFIG,
                )
        render_disclaimer(case.get("disclaimer", ""))


def compliance_cases_tab(base_url: str, demo_query: str, default_top_k: int) -> None:
    st.markdown("#### Compliance Cases")
    st.caption(
        "Converts a policy question into a demo compliance case record. "
        "Cases are stored in memory only and are cleared when the backend restarts."
    )

    with st.expander("Create New Case", expanded=True):
        with st.form("create_case_form"):
            query = st.text_area("Case query", value=demo_query)
            col1, col2, col3 = st.columns(3)
            with col1:
                requester = st.text_input("Requester", value="demo_user")
            with col2:
                department = st.text_input("Department", value="Finance")
            with col3:
                top_k = st.slider("top_k", min_value=1, max_value=10, value=default_top_k)
            submitted = st.form_submit_button("Create Case", type="primary")
        if submitted:
            with st.spinner("Creating case record..."):
                st.session_state.created_case = create_case(base_url, query, requester, department, top_k)

    created = st.session_state.created_case
    if created:
        if created["ok"]:
            st.success(f"Case {created['data'].get('case_id')} created.")
            render_case_details(created["data"])
        else:
            show_error(created)

    st.divider()
    st.markdown("##### Case Queue")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Refresh Case List", use_container_width=True):
            st.session_state.case_list = list_cases(base_url)
    with col2:
        fetch_col, btn_col = st.columns([2, 1])
        with fetch_col:
            case_id = st.text_input("Fetch case by case_id", label_visibility="collapsed", placeholder="Fetch case by case_id")
        with btn_col:
            if st.button("Fetch Case", use_container_width=True):
                st.session_state.created_case = get_case(base_url, case_id)

    case_list = st.session_state.case_list
    if case_list:
        if case_list["ok"]:
            cases = case_list["data"].get("cases", [])
            if cases:
                table_rows = [
                    {
                        "case_id": case.get("case_id", ""),
                        "requester": case.get("requester", ""),
                        "department": case.get("department", ""),
                        "risk_level": case.get("risk_level", ""),
                        "risk_category": case.get("risk_category", ""),
                        "status": case.get("status", ""),
                        "created_at": case.get("created_at", ""),
                    }
                    for case in cases
                ]
                st.dataframe(
                    pd.DataFrame(table_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "case_id": st.column_config.TextColumn("Case ID", width="small"),
                        "requester": st.column_config.TextColumn("Requester", width="small"),
                        "department": st.column_config.TextColumn("Department", width="small"),
                        "risk_level": st.column_config.TextColumn("Risk Level", width="small"),
                        "risk_category": st.column_config.TextColumn("Risk Category", width="medium"),
                        "status": st.column_config.TextColumn("Status", width="small"),
                        "created_at": st.column_config.TextColumn("Created At", width="medium"),
                    },
                )
            else:
                st.info("No cases created in this server runtime.")
        else:
            show_error(case_list)


def agent_state_from_result(data: dict | None) -> dict:
    if not isinstance(data, dict):
        return {}
    return data.get("agent_state") or {}


def workflow_response_from_agent_result(data: dict | None) -> dict:
    return agent_state_from_result(data).get("workflow_response") or {}


def refresh_agent_run_list(base_url: str) -> None:
    st.session_state.agent_run_list = list_agent_runs(base_url)


def render_agent_status(data: dict) -> None:
    agent_state = agent_state_from_result(data)
    risk_level = agent_state.get("risk_level", "")
    risk_category = agent_state.get("risk_category", "")
    case_id = data.get("case_id") or "Not created"

    st.markdown("##### Agent Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_summary_card("Run ID", f"`{data.get('run_id', '')}`")
    with col2:
        render_summary_card("Status", f":gray-badge[{data.get('status', '') or 'Unknown'}]")
    with col3:
        render_summary_card("Risk Level", risk_badge(risk_level))
    with col4:
        render_summary_card("Case", f"**{case_id}**", detail=f"case_created: {data.get('case_created', False)}")

    col5, col6, col7 = st.columns(3)
    with col5:
        render_summary_card("Risk Category", f"**{risk_category or 'Unknown'}**")
    with col6:
        render_summary_card("Approval Required", f"**{data.get('approval_required', False)}**")
    with col7:
        render_summary_card("Blocked", f"**{data.get('blocked', False)}**")


def render_agent_workflow_advice(agent_state: dict) -> None:
    workflow = agent_state.get("workflow_response") or {}
    if not workflow:
        st.info("No workflow advice returned for this agent run.")
        return

    st.markdown("##### Grounded Workflow Advice")
    col1, col2 = st.columns([1, 2])
    with col1:
        render_summary_card(
            "Evidence Quality",
            evidence_quality_badge(workflow.get("evidence_quality", "")),
            detail=workflow.get("evidence_quality", ""),
        )
    with col2:
        render_summary_card("Risk Reasoning", workflow.get("risk_reasoning", "") or "_No risk reasoning returned._")

    render_answer_card("Grounded Answer", workflow.get("grounded_answer", ""))

    with st.container(border=True):
        st.markdown("##### Workflow Checklist")
        checklist = workflow.get("workflow_checklist") or []
        if checklist:
            st.markdown("\n".join(f"{index}. {step}" for index, step in enumerate(checklist, start=1)))
        else:
            st.info("No workflow checklist returned.")

    render_citations(workflow.get("citations"))
    render_disclaimer(workflow.get("disclaimer", ""))
    render_timing_metadata(workflow.get("timing_metadata"), "Workflow performance metadata")


def render_pending_action(pending_action: dict | None) -> None:
    st.markdown("##### Pending Action")
    if not pending_action:
        st.info("No pending action.")
        return

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Action Type**\n\n`{pending_action.get('action_type', '')}`")
        col2.markdown(f"**Requires Approval**\n\n{pending_action.get('requires_approval', False)}")
        col3.markdown(f"**Blocked**\n\n{pending_action.get('blocked', False)}")
        st.markdown(f"**Description:** {pending_action.get('description', '')}")
        reason = pending_action.get("reason")
        if reason:
            st.caption(reason)


def render_decision_trace(trace: list[dict] | None) -> None:
    st.markdown("##### Decision Trace")
    rows = trace or []
    if not rows:
        st.info("No decision trace returned.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_order=["step", "status", "message"],
        column_config={
            "step": st.column_config.TextColumn("Step", width="small"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "message": st.column_config.TextColumn("Message", width="large"),
        },
    )


def render_agent_review_controls(base_url: str, data: dict) -> None:
    st.markdown("##### Human Review Controls")
    run_id = data.get("run_id", "")
    status = data.get("status", "")
    blocked = bool(data.get("blocked", False))
    case_created = bool(data.get("case_created", False))
    approval_required = bool(data.get("approval_required", False))
    agent_state = agent_state_from_result(data)

    if case_created:
        st.success(f"Mock compliance case created: {data.get('case_id')}")
        return

    if status == "Rejected" or agent_state.get("human_decision") == "rejected":
        st.info("This agent run was rejected. No mock case was created.")
        return

    reject_reason = st.text_input(
        "Optional rejection reason",
        value=st.session_state.agent_reject_reason or "",
        key="agent_reject_reason_input",
        placeholder="Reason for rejecting or closing the review",
    )
    st.session_state.agent_reject_reason = reject_reason

    if blocked:
        st.warning("This run is blocked for manual review. Mock case creation is disabled.")
        if st.button("Record Rejection / Close Review", key="agent_reject_blocked", use_container_width=True):
            result = reject_agent_run(base_url, run_id, reason=reject_reason)
            if result["ok"]:
                st.session_state.agent_run_result = result
                refresh_agent_run_list(base_url)
                st.info("Review rejection recorded.")
            else:
                show_error(result)
        return

    if approval_required:
        approve_label = "Approve"
        reject_label = "Reject"
    elif status == "ApprovalRecommended":
        approve_label = "Approve / Create Mock Case"
        reject_label = "Reject / Do Not Create"
    elif status == "ReadyForConfirmation":
        approve_label = "Create Mock Case"
        reject_label = "Reject / Do Not Create"
    else:
        st.info("No human review action is currently available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button(approve_label, type="primary", key="agent_approve_action", use_container_width=True):
            result = approve_agent_run(base_url, run_id)
            if result["ok"]:
                st.session_state.agent_run_result = result
                refresh_agent_run_list(base_url)
                case_id = result["data"].get("case_id")
                st.success(f"Mock compliance case created: {case_id}" if case_id else "Agent run approved.")
            else:
                show_error(result)
    with col2:
        if st.button(reject_label, key="agent_reject_action", use_container_width=True):
            result = reject_agent_run(base_url, run_id, reason=reject_reason)
            if result["ok"]:
                st.session_state.agent_run_result = result
                refresh_agent_run_list(base_url)
                st.info("Agent run rejected.")
            else:
                show_error(result)


def render_agent_run_list(base_url: str) -> None:
    st.markdown("##### Agent Run List")
    col_refresh, col_fetch = st.columns([1, 3], vertical_alignment="bottom")
    with col_refresh:
        if st.button("Refresh Agent Runs", use_container_width=True):
            refresh_agent_run_list(base_url)

    agent_list = st.session_state.agent_run_list
    run_ids: list[str] = []
    if agent_list:
        if agent_list["ok"]:
            runs = agent_list["data"].get("agent_runs", [])
            run_ids = [run.get("run_id", "") for run in runs if run.get("run_id")]
            if runs:
                table_rows = [
                    {
                        "run_id": run.get("run_id", ""),
                        "status": run.get("status", ""),
                        "query": run.get("query", ""),
                        "risk_level": run.get("risk_level", ""),
                        "risk_category": run.get("risk_category", ""),
                        "approval_required": run.get("approval_required", False),
                        "blocked": run.get("blocked", False),
                        "case_created": run.get("case_created", False),
                        "case_id": run.get("case_id", ""),
                    }
                    for run in runs
                ]
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No agent runs created in this server runtime.")
        else:
            show_error(agent_list)

    with col_fetch:
        if run_ids:
            selected = st.selectbox("Select agent run", run_ids, key="agent_run_select")
        else:
            selected = st.text_input("Fetch agent run by run_id", value=st.session_state.selected_agent_run_id or "")
        if st.button("Fetch Agent Run", use_container_width=True):
            result = get_agent_run(base_url, selected)
            if result["ok"]:
                st.session_state.agent_run_result = result
                st.session_state.selected_agent_run_id = result["data"].get("run_id")
            else:
                show_error(result)


AUDIT_EVENT_TYPES = [
    "All",
    "AGENT_RUN_CREATED",
    "APPROVAL_REQUIRED",
    "APPROVAL_RECOMMENDED",
    "READY_FOR_CONFIRMATION",
    "CASE_BLOCKED_INSUFFICIENT_EVIDENCE",
    "APPROVAL_GRANTED",
    "APPROVAL_REJECTED",
    "APPROVAL_BLOCKED",
    "CASE_CREATION_BLOCKED",
    "CASE_CREATED",
]

SENSITIVE_AUDIT_METADATA_KEYS = {
    "workflow_response",
    "grounded_answer",
    "evidence_text",
    "full_text",
    "content",
    "citations",
}


def sanitize_audit_metadata_for_display(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    return {key: value for key, value in metadata.items() if key not in SENSITIVE_AUDIT_METADATA_KEYS}


def render_audit_event_table(audit_events: list[dict] | None) -> None:
    events = audit_events or []
    if not events:
        st.info("No audit events returned.")
        return
    table_rows = [
        {
            "audit_id": event.get("audit_id", ""),
            "timestamp": event.get("timestamp", ""),
            "event_type": event.get("event_type", ""),
            "actor": event.get("actor", ""),
            "run_id": event.get("run_id", ""),
            "case_id": event.get("case_id", ""),
            "risk_level": event.get("risk_level", ""),
            "risk_category": event.get("risk_category", ""),
            "message": event.get("message", ""),
        }
        for event in events
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        column_order=["audit_id", "timestamp", "event_type", "actor", "run_id", "case_id", "risk_level", "risk_category", "message"],
        column_config={
            "audit_id": st.column_config.TextColumn("Audit ID", width="small"),
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "event_type": st.column_config.TextColumn("Event Type", width="medium"),
            "actor": st.column_config.TextColumn("Actor", width="small"),
            "run_id": st.column_config.TextColumn("Run ID", width="small"),
            "case_id": st.column_config.TextColumn("Case ID", width="small"),
            "risk_level": st.column_config.TextColumn("Risk", width="small"),
            "risk_category": st.column_config.TextColumn("Category", width="medium"),
            "message": st.column_config.TextColumn("Message", width="large"),
        },
    )


def render_audit_event_detail(event: dict | None) -> None:
    if not event:
        return
    with st.container(border=True):
        st.markdown(f"##### Audit Event `{event.get('audit_id', '')}`")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Event Type**\n\n`{event.get('event_type', '')}`")
        col2.markdown(f"**Actor**\n\n{event.get('actor') or 'Unknown'}")
        col3.markdown(f"**Timestamp**\n\n{event.get('timestamp', '')}")
        st.markdown(f"**Run ID:** `{event.get('run_id') or ''}`")
        st.markdown(f"**Case ID:** `{event.get('case_id') or ''}`")
        st.markdown(f"**Risk:** {risk_badge(event.get('risk_level', ''))} **Category:** {event.get('risk_category') or 'Unknown'}")
        st.markdown(f"**Message:** {event.get('message', '')}")
        st.markdown("**Metadata**")
        st.json(sanitize_audit_metadata_for_display(event.get("metadata")))


def render_audit_trail_section(base_url: str) -> None:
    st.markdown("##### Audit Trail")
    st.caption(
        "Audit logs record operational governance events for agent runs, approvals, rejections, blocked cases, "
        "and mock case creation. They do not store hidden reasoning or full evidence text."
    )

    col_event, col_run, col_case, col_limit = st.columns([2, 2, 2, 1])
    with col_event:
        event_type = st.selectbox("Event type", AUDIT_EVENT_TYPES, key="audit_filter_event_type_select")
    with col_run:
        run_id = st.text_input("Run ID", value=st.session_state.audit_filter_run_id or "", key="audit_filter_run_id_input")
    with col_case:
        case_id = st.text_input("Case ID", value=st.session_state.audit_filter_case_id or "", key="audit_filter_case_id_input")
    with col_limit:
        limit = st.number_input("Limit", min_value=1, max_value=100, value=20, step=1, key="audit_filter_limit")

    st.session_state.audit_filter_event_type = event_type
    st.session_state.audit_filter_run_id = run_id
    st.session_state.audit_filter_case_id = case_id

    filter_event_type = None if event_type == "All" else event_type
    col_refresh, col_current = st.columns(2)
    with col_refresh:
        if st.button("Refresh Audit Logs", key="refresh_audit_logs", use_container_width=True):
            result = list_audit_logs(base_url, event_type=filter_event_type, run_id=run_id, case_id=case_id, limit=int(limit))
            if result["ok"]:
                st.session_state.audit_log_result = result
            else:
                show_error(result)
    with col_current:
        current_run_id = st.session_state.selected_agent_run_id
        if current_run_id and st.button("Show audit logs for current run", key="audit_current_run", use_container_width=True):
            result = list_audit_logs(base_url, run_id=current_run_id, limit=20)
            if result["ok"]:
                st.session_state.audit_log_result = result
                st.session_state.audit_filter_run_id = current_run_id
            else:
                show_error(result)

    audit_result = st.session_state.audit_log_result
    audit_events: list[dict] = []
    if audit_result:
        if audit_result["ok"]:
            audit_events = audit_result["data"].get("audit_events", [])
            render_audit_event_table(audit_events)
        else:
            show_error(audit_result)
    else:
        st.info("Refresh audit logs to view governance events.")

    st.markdown("##### Audit Event Detail")
    audit_ids = [event.get("audit_id", "") for event in audit_events if event.get("audit_id")]
    if audit_ids:
        selected_audit_id = st.selectbox("Select audit event", audit_ids, key="audit_event_select")
    else:
        selected_audit_id = st.text_input("Fetch audit event by audit_id", value=st.session_state.selected_audit_id or "")
    if st.button("Fetch Audit Event", key="fetch_audit_event", use_container_width=True):
        result = get_audit_log(base_url, selected_audit_id)
        if result["ok"]:
            st.session_state.audit_event_detail = result
            st.session_state.selected_audit_id = result["data"].get("audit_id")
        else:
            show_error(result)

    detail_result = st.session_state.audit_event_detail
    if detail_result:
        if detail_result["ok"]:
            render_audit_event_detail(detail_result["data"])
        else:
            show_error(detail_result)


def agent_workflow_tab(base_url: str, demo_query: str, default_top_k: int) -> None:
    st.markdown("#### Agent Workflow")
    st.caption("Human-in-the-loop agent runs use the FastAPI /agent-runs flow. Cases are created only after explicit review action.")

    with st.expander("Start Agent Run", expanded=True):
        with st.form("agent_run_form"):
            query = st.text_area("Agent query", value=demo_query)
            col1, col2, col3 = st.columns(3)
            with col1:
                requester = st.text_input("Requester", value="Demo User")
            with col2:
                department = st.text_input("Department", value="Compliance")
            with col3:
                top_k = st.number_input("top_k", min_value=1, max_value=20, value=min(max(default_top_k, 1), 20), step=1)
            submitted = st.form_submit_button("Start Agent Run", type="primary")
        if submitted:
            with st.spinner("Starting agent run..."):
                result = start_agent_run(base_url, query, requester=requester, department=department, top_k=int(top_k))
            if result["ok"]:
                st.session_state.agent_run_result = result
                st.session_state.selected_agent_run_id = result["data"].get("run_id")
                refresh_agent_run_list(base_url)
                st.success(f"Agent run started: {result['data'].get('run_id')}")
            else:
                show_error(result)

    st.info(
        "LangGraph is used in this project as a conditional human-review branch for high-risk or insufficient-evidence cases. "
        "It is not the default execution path for /agent-runs. The stable FastAPI human-in-the-loop flow remains the main "
        "operational path. The LangGraph layer demonstrates graph-based orchestration, conditional routing, and approval "
        "gating without automatically creating cases or performing external actions."
    )

    result = st.session_state.agent_run_result
    if result:
        if result["ok"]:
            data = result["data"]
            st.divider()
            render_agent_status(data)
            render_timing_metadata(data.get("timing_metadata"), "Agent performance metadata")
            st.divider()
            render_agent_workflow_advice(agent_state_from_result(data))
            st.divider()
            render_pending_action(data.get("pending_action"))
            render_agent_review_controls(base_url, data)
            st.divider()
            render_decision_trace(data.get("trace") or agent_state_from_result(data).get("trace"))
        else:
            show_error(result)

    st.divider()
    render_agent_run_list(base_url)
    st.divider()
    render_audit_trail_section(base_url)


def evaluation_tab(base_url: str, default_top_k: int) -> None:
    st.markdown("#### Evaluation")
    st.caption("Deterministic smoke test for demonstration purposes. This is not a scientific benchmark.")
    col_btn, col_topk = st.columns([3, 1], vertical_alignment="bottom")
    with col_topk:
        top_k = st.slider("top_k", min_value=1, max_value=10, value=default_top_k, key="eval_top_k")
    with col_btn:
        run_clicked = st.button("Run Evaluation", type="primary")
    if run_clicked:
        rows = []
        progress = st.progress(0.0, text="Running evaluation examples...")
        for position, example in enumerate(EVALUATION_EXAMPLES, start=1):
            response = get_workflow_advice(base_url, example["query"], top_k)
            if response["ok"]:
                rows.append(evaluate_response(example, response["data"]))
            else:
                rows.append(
                    {
                        "query": example["query"],
                        "expected_risk": " or ".join(example["expected_risks"]),
                        "actual_risk": "ERROR",
                        "expected_category": " or ".join(example["expected_categories"]),
                        "actual_category": response["error"],
                        "pass": False,
                        "evidence_quality": "",
                    }
                )
            progress.progress(position / len(EVALUATION_EXAMPLES), text=f"Evaluated {position}/{len(EVALUATION_EXAMPLES)} examples")
        progress.empty()
        st.session_state.evaluation_results = rows

    rows = st.session_state.evaluation_results
    if not rows:
        return
    summary = summarize_results(rows)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric("Total Examples", summary["total_examples"])
    with col2:
        with st.container(border=True):
            st.metric("Passed", summary["passed_examples"])
    with col3:
        with st.container(border=True):
            st.metric("Pass Rate", f"{summary['pass_rate']:.0%}")

    display_rows = [dict(row, result="Pass" if row.get("pass") else "Fail") for row in rows]
    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        hide_index=True,
        column_order=["result", "query", "expected_risk", "actual_risk", "expected_category", "actual_category", "evidence_quality"],
        column_config={
            "result": st.column_config.TextColumn("Result", width="small"),
            "query": st.column_config.TextColumn("Query", width="medium"),
            "expected_risk": st.column_config.TextColumn("Expected Risk", width="small"),
            "actual_risk": st.column_config.TextColumn("Actual Risk", width="small"),
            "expected_category": st.column_config.TextColumn("Expected Category", width="medium"),
            "actual_category": st.column_config.TextColumn("Actual Category", width="medium"),
            "evidence_quality": st.column_config.TextColumn("Evidence Quality", width="small"),
        },
    )
    st.caption("This dashboard checks whether demo scenarios return expected workflow categories and risk levels. It is not an ML benchmark or legal validation.")


def main() -> None:
    st.set_page_config(page_title="WITS Compliance Copilot", layout="wide")
    st.markdown(APP_STYLE, unsafe_allow_html=True)
    init_state()
    base_url, demo_query, top_k = sidebar_controls()
    st.markdown(HEADER_HTML, unsafe_allow_html=True)
    tabs = st.tabs(["Policy Q&A", "Workflow Advice", "Compliance Cases", "Agent Workflow", "Evaluation"])
    with tabs[0]:
        policy_qa_tab(base_url, demo_query, top_k)
    with tabs[1]:
        workflow_advice_tab(base_url, demo_query, top_k)
    with tabs[2]:
        compliance_cases_tab(base_url, demo_query, top_k)
    with tabs[3]:
        agent_workflow_tab(base_url, demo_query, top_k)
    with tabs[4]:
        evaluation_tab(base_url, top_k)


if __name__ == "__main__":
    main()
