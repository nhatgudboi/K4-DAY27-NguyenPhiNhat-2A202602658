"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", page_icon="🤖", layout="wide")
st.title("🤖 HITL PR Review Agent")
st.caption("Human-in-the-Loop PR Review powered by LangGraph, SQLite Audit Trail & Streamlit")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def get_recent_sessions():
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 20
            """
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


with st.sidebar:
    st.header("📜 Recent sessions")
    try:
        sessions = asyncio.run(get_recent_sessions())
        if sessions:
            for s in sessions:
                tid = s["thread_id"]
                short_tid = tid[:8]
                risk = s["worst_risk"]
                risk_badge = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(risk, "⚪")
                pr_label = s["pr_url"].split("/")[-1] if s["pr_url"] else "?"
                if st.button(f"{risk_badge} PR #{pr_label} ({short_tid})", key=f"session_{tid}", use_container_width=True):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = s["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
                st.caption(f"Risk: **{risk}** · Events: **{s['events']}**\nLast: {s['last_event'][:19]}")
                st.divider()
        else:
            st.caption("No sessions recorded yet.")
    except Exception as e:
        st.caption(f"Audit table not initialized yet ({e}).")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        pr_url = st.text_input(
            "PR URL", value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
    with col_btn:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Run review", type="primary", use_container_width=True)


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.info(f"### 📋 Approval Requested — Confidence {conf:.0%}")
    st.markdown(f"**Reasoning:** {payload['confidence_reasoning']}")
    st.markdown(f"**Summary:** {payload['summary']}")

    comments = payload.get("comments", [])
    if comments:
        st.markdown("#### Proposed Comments:")
        for c in comments:
            severity = c.get("severity", "suggestion")
            file_path = c.get("file", "")
            line = c.get("line") or "?"
            st.markdown(f"- `[{severity.upper()}]` **{file_path}:{line}** — {c.get('body')}")

    with st.expander("🔍 View Diff Preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    st.markdown("---")
    feedback = st.text_input("Feedback / Reviewer note (optional):", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.warning(f"### ⚠️ Strong Escalation — Confidence {conf:.0%}")
    st.markdown(f"**Reasoning:** {payload['confidence_reasoning']}")
    if payload.get("risk_factors"):
        st.error("🚨 **Identified Risks:**\n- " + "\n- ".join(payload["risk_factors"]))
    st.markdown(f"**Summary:** {payload['summary']}")

    st.markdown("---")
    st.markdown("#### 💬 The agent has specific questions before proceeding:")
    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(f"Q{i+1}: {q}", key=f"q_{i}")
        submitted = st.form_submit_button("Submit answers & Refine Review", type="primary", use_container_width=True)
        if submitted:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM to analyze..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result
    st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind", "")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent and committing review..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    st.markdown("---")
    
    if action.startswith("auto"):
        analysis = final.get("analysis")
        conf = f"{analysis.confidence:.0%}" if analysis else ">72%"
        summary = analysis.summary if analysis else "Automated review completed."
        st.success(f"### 🎉 Auto-Approved (Confidence {conf})")
        st.markdown(summary)
        st.markdown(f"👉 [**View comment on GitHub**]({st.session_state.pr_url})")
    elif action.startswith("committed"):
        st.success(f"### ✅ Review Committed!")
        st.markdown(f"Your review comment has been posted successfully to the PR.")
        st.markdown(f"👉 [**View comment on GitHub**]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning("### 🚫 Review Rejected")
        st.write("The reviewer rejected the review. No comment was posted to GitHub.")
    else:
        st.info(f"Final Action: **{action}**")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}` · "
        f"Replay command: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )

    # Show replay timeline in the UI
    try:
        events = asyncio.run(replay_events(st.session_state.thread_id))
        if events:
            with st.expander("📊 View Audit Trail Timeline", expanded=True):
                for ev in events:
                    act = ev["action"]
                    dec = ev["decision"]
                    conf = f"{ev['confidence']:.0%}"
                    reviewer = ev["reviewer_id"] or "Agent"
                    dur = f"{ev['execution_time_ms']}ms"
                    reason = f" — _{ev['reason']}_" if ev.get("reason") else ""
                    st.markdown(f"- **`{act}`** | Decision: `{dec}` | Conf: `{conf}` | Reviewer: `{reviewer}` ({dur}){reason}")
    except Exception:
        pass
