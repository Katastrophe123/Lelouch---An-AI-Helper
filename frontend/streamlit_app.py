"""Streamlit frontend for the AI Homework Helper.

This is a **pure API client**. Compared with v1 it contains:
  * no LangChain / Hugging Face client
  * no prompt text
  * no conversation state in ``session_state`` beyond the current selection

Conversation history is fetched from ``GET /conversations/{id}`` on every run,
so a browser refresh - or a different browser entirely - shows the same
history. In v1 a refresh wiped the conversation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import ApiError, HomeworkHelperClient, Session  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

STATUS_BADGE = {
    "verified": ("✅", "Self-checked"),
    "unverified": ("⚠️", "Unverified - please check"),
    "failed": ("❌", "Failed"),
    "pending": ("⏳", "Pending"),
}

NODE_LABEL = {
    "understand": "Understanding the question",
    "solve": "Solving",
    "self_check": "Checking the answer",
    "explain": "Explaining step by step",
    "fallback": "Preparing a partial answer",
}


@st.cache_resource
def get_client() -> HomeworkHelperClient:
    return HomeworkHelperClient(API_BASE_URL)


def session() -> Session | None:
    return st.session_state.get("session")


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_status(message: dict[str, Any]) -> None:
    """Show the answer status, confidence and retry count for an answer."""
    status = message.get("answer_status")
    if not status:
        return
    icon, label = STATUS_BADGE.get(status, ("•", status))
    bits = [f"{icon} **{label}**"]
    if message.get("confidence") is not None:
        bits.append(f"confidence {message['confidence']:.0%}")
    if message.get("retry_count"):
        bits.append(f"{message['retry_count']} retry/retries")
    if message.get("fallback_reason"):
        bits.append(f"reason: `{message['fallback_reason']}`")
    if message.get("latency_ms") is not None:
        bits.append(f"{message['latency_ms']} ms")
    st.caption(" · ".join(bits))


def render_trace(message: dict[str, Any]) -> None:
    """Expandable view of the graph steps that produced this answer."""
    trace = message.get("trace") or []
    if not trace:
        return
    with st.expander(f"How I got here ({len(trace)} steps)"):
        for step in trace:
            label = NODE_LABEL.get(step["node"], step["node"])
            marker = "✔" if step["status"] == "ok" else "✖"
            st.markdown(
                f"{marker} **{step['sequence']}. {label}** "
                f"→ `{step.get('routing_decision') or 'end'}` "
                f"({step['latency_ms']} ms)"
            )
            if step.get("error"):
                st.error(step["error"], icon="⚠️")
            output = step.get("output") or {}
            if output:
                st.json(output, expanded=False)


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_status(message)
        render_trace(message)


# --------------------------------------------------------------------------- #
# Panes
# --------------------------------------------------------------------------- #
def render_sign_in() -> None:
    st.title("📘 AI Homework Helper")
    st.caption(
        "A LangGraph tutor that solves your question, checks its own answer, "
        "then explains it step by step."
    )

    client = get_client()
    try:
        health = client.health()
        st.success(
            f"Backend is up · model `{health['llm_model']}` (provider `{health['llm_provider']}`)",
            icon="🔌",
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(f"Cannot reach the API at {API_BASE_URL}: {exc}", icon="🚫")
        st.stop()

    with st.form("sign-in"):
        username = st.text_input("Your name", value="", max_chars=80)
        password = st.text_input("Shared password", type="password")
        submitted = st.form_submit_button("Start studying")

    if submitted:
        if not username.strip():
            st.warning("Please enter a name.")
            return
        try:
            st.session_state["session"] = client.sign_in(username.strip(), password)
        except ApiError as exc:
            st.error(exc.message, icon="🔒")
            return
        st.rerun()


def render_sidebar(current: Session) -> str | None:
    """Conversation picker. Returns the selected conversation id."""
    client = get_client()
    with st.sidebar:
        st.markdown(f"**Signed in as** `{current.username}`")
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.divider()
        if st.button("➕ New conversation", use_container_width=True, type="primary"):
            created = client.create_conversation(current.access_token)
            st.session_state["conversation_id"] = created["id"]
            st.rerun()

        st.markdown("### Your conversations")
        try:
            page = client.list_conversations(current.access_token, page_size=50)
        except ApiError as exc:
            st.error(exc.message)
            return st.session_state.get("conversation_id")

        if not page["items"]:
            st.caption("No conversations yet - create one above.")
            return None

        options = {item["id"]: item for item in page["items"]}
        selected = st.session_state.get("conversation_id") or page["items"][0]["id"]
        if selected not in options:
            selected = page["items"][0]["id"]

        choice = st.radio(
            "Conversations",
            options=list(options),
            index=list(options).index(selected),
            format_func=lambda key: f"{options[key]['title']} ({options[key]['message_count']})",
            label_visibility="collapsed",
        )
        st.session_state["conversation_id"] = choice

        st.divider()
        st.caption(f"{page['total']} conversation(s) stored in PostgreSQL")
        st.toggle("Stream the explanation (SSE)", value=True, key="use_streaming")
        return str(choice)


def render_streamed_answer(current: Session, conversation_id: str, question: str) -> None:
    """Consume the SSE stream, showing node progress then the explanation."""
    client = get_client()
    with st.chat_message("assistant"):
        progress = st.empty()
        body = st.empty()
        chunks: list[str] = []
        final: dict[str, Any] = {}
        try:
            for event in client.ask_streaming(current.access_token, conversation_id, question):
                if event.name == "step":
                    label = NODE_LABEL.get(event.data["node"], event.data["node"])
                    marker = "✔" if event.data["status"] == "ok" else "✖"
                    progress.caption(f"{marker} {label}…")
                elif event.name == "token":
                    chunks.append(event.data.get("text", ""))
                    body.markdown("".join(chunks))
                elif event.name == "done":
                    final = event.data
                elif event.name == "error":
                    st.error(event.data.get("message", "The run failed."), icon="⚠️")
                    return
        except ApiError as exc:
            st.error(exc.message, icon="⚠️")
            return

        progress.empty()
        # The `done` event carries the persisted content, which is authoritative
        # (a fallback answer is not streamed token by token).
        body.markdown(final.get("content") or "".join(chunks))
        render_status(final)


def render_conversation(current: Session, conversation_id: str) -> None:
    client = get_client()
    try:
        conversation = client.get_conversation(current.access_token, conversation_id)
    except ApiError as exc:
        st.error(exc.message, icon="⚠️")
        return

    st.subheader(conversation["title"])
    meta = [f"{conversation['message_count']} message(s)"]
    if conversation.get("subject"):
        meta.append(f"subject: `{conversation['subject']}`")
    st.caption(" · ".join(meta))

    for message in conversation["messages"]:
        render_message(message)

    question = st.chat_input("Ask a homework question…")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.get("use_streaming", True):
        render_streamed_answer(current, conversation_id, question)
    else:
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                turn = client.ask(current.access_token, conversation_id, question)
            except ApiError as exc:
                st.error(exc.message, icon="⚠️")
                return
            assistant = turn["assistant_message"]
            st.markdown(assistant["content"])
            render_status(assistant)
            render_trace(assistant)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI Homework Helper", page_icon="📘", layout="centered")

    current = session()
    if current is None:
        render_sign_in()
        return

    conversation_id = render_sidebar(current)
    if conversation_id is None:
        st.info("Create a conversation from the sidebar to get started.", icon="👈")
        return
    render_conversation(current, conversation_id)


main()
