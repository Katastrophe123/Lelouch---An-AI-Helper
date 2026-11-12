"""Streamlit frontend for Lelouch (agent mode) and the AI Homework Helper (tutor mode).

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
    "completed": ("🤖", "Agent answer"),
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
    "agent_start": "Agent started",
    "thinking": "Thinking",
    "tool": "Using a tool",
    "answer": "Answer",
}

MODE_LABEL = {"agent": "🤖 Agent (Claude)", "tutor": "📘 Tutor (self-checking)"}


def step_label(step: dict[str, Any]) -> str:
    """Agent steps carry their own label (e.g. the tool name); tutor ones do not."""
    output = step.get("output") or {}
    label = step.get("label") or output.get("label")
    return str(label or NODE_LABEL.get(step["node"], step["node"]))


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
        bits.append(f"{message['latency_ms'] / 1000:.1f} s")
    trace = message.get("trace") or []
    cost = (trace[-1].get("output") or {}).get("cost_usd") if trace else None
    if cost:
        bits.append(f"${cost:.3f} of agent credit")
    st.caption(" · ".join(bits))


def render_trace(message: dict[str, Any]) -> None:
    """Expandable view of the graph steps that produced this answer."""
    trace = message.get("trace") or []
    if not trace:
        return
    with st.expander(f"How I got here ({len(trace)} steps)"):
        for step in trace:
            label = step_label(step)
            marker = "✔" if step["status"] == "ok" else "✖"
            detail = (step.get("output") or {}).get("input")
            st.markdown(
                f"{marker} **{step['sequence']}. {label}** "
                + (f"— {detail} " if detail else "")
                + (
                    f"→ `{step['routing_decision']}` "
                    if step.get("routing_decision") and step["node"] != "answer"
                    else ""
                )
                + f"({step['latency_ms']} ms)"
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
    st.title("🤖 Lelouch - An AI Helper")
    st.caption(
        "A Claude agent that researches, browses and reads your own documents - "
        "plus the original self-checking homework tutor."
    )

    client = get_client()
    try:
        health = client.health()
        agent = health.get("agent_model")
        st.success(
            f"Backend is up · agent `{agent}` ({health.get('agent_auth')}) · "
            f"tutor `{health['llm_model']}` ({health['llm_provider']})",
            icon="🔌",
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(f"Cannot reach the API at {API_BASE_URL}: {exc}", icon="🚫")
        st.stop()

    with st.form("sign-in"):
        username = st.text_input("Your name", value="", max_chars=80)
        password = st.text_input("Shared password", type="password")
        submitted = st.form_submit_button("Sign in")

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
        mode = st.radio(
            "New conversation type",
            options=list(MODE_LABEL),
            format_func=MODE_LABEL.__getitem__,
            key="new_mode",
        )
        model = None
        if mode == "agent":
            config = agent_config(current)
            if config:
                model = st.selectbox(
                    "Model",
                    options=config["models"],
                    index=config["models"].index(config["default_model"])
                    if config["default_model"] in config["models"]
                    else 0,
                )
        if st.button("➕ New conversation", use_container_width=True, type="primary"):
            try:
                created = client.create_conversation(current.access_token, mode=mode, model=model)
            except ApiError as exc:
                st.error(exc.message)
            else:
                st.session_state["conversation_id"] = created["id"]
                st.rerun()

        render_documents(current)

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
            format_func=lambda key: (
                f"{'🤖' if options[key].get('mode') == 'agent' else '📘'} "
                f"{options[key]['title']} ({options[key]['message_count']})"
            ),
            label_visibility="collapsed",
        )
        st.session_state["conversation_id"] = choice

        st.divider()
        st.caption(f"{page['total']} conversation(s) stored in PostgreSQL")
        st.toggle("Stream answers live (SSE)", value=True, key="use_streaming")
        return str(choice)


def agent_config(current: Session) -> dict[str, Any] | None:
    cached: dict[str, Any] | None = st.session_state.get("agent_config")
    if cached is None:
        try:
            cached = get_client().agent_config(current.access_token)
        except ApiError as exc:
            st.caption(f"Agent unavailable: {exc.message}")
            return None
        st.session_state["agent_config"] = cached
    return cached


def render_documents(current: Session) -> None:
    """Upload and manage the files the agent can search."""
    client = get_client()
    with st.expander("📚 My documents", expanded=False):
        st.caption("Agent conversations can search these (PDF, Word, text, Markdown, CSV, code).")
        uploads = st.file_uploader(
            "Add documents",
            accept_multiple_files=True,
            key=f"uploader-{st.session_state.get('uploader_round', 0)}",
            label_visibility="collapsed",
        )
        if uploads:
            for upload in uploads:
                try:
                    with st.spinner(f"Reading {upload.name}…"):
                        client.upload_document(
                            current.access_token, upload.name, upload.getvalue(), upload.type
                        )
                    st.toast(f"Added {upload.name}", icon="📄")
                except ApiError as exc:
                    st.error(f"{upload.name}: {exc.message}")
            # A fresh uploader key clears the widget so files are not re-sent.
            st.session_state["uploader_round"] = st.session_state.get("uploader_round", 0) + 1
            st.rerun()
        try:
            documents = client.list_documents(current.access_token)
        except ApiError as exc:
            st.error(exc.message)
            return
        if not documents:
            st.caption("No documents yet.")
        for document in documents:
            left, right = st.columns([5, 1])
            left.markdown(f"📄 {document['filename']}  \n`{document['chunk_count']} chunks`")
            if right.button("🗑", key=f"del-{document['id']}", help="Delete"):
                client.delete_document(current.access_token, document["id"])
                st.rerun()


def render_streamed_answer(current: Session, conversation_id: str, question: str) -> None:
    """Consume the SSE stream, showing live steps then the answer."""
    client = get_client()
    with st.chat_message("assistant"):
        progress = st.empty()
        body = st.empty()
        chunks: list[str] = []
        lines: list[str] = []
        final: dict[str, Any] = {}
        try:
            for event in client.ask_streaming(current.access_token, conversation_id, question):
                if event.name == "step":
                    label = step_label(event.data)
                    marker = "✔" if event.data["status"] == "ok" else "✖"
                    lines.append(f"{marker} {label}")
                    progress.caption("  \n".join(lines[-6:]) + " …")
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
    agent_mode = conversation.get("mode") == "agent"
    meta = [MODE_LABEL.get(conversation.get("mode", "tutor"), "")]
    if conversation.get("agent_model"):
        meta.append(f"`{conversation['agent_model']}`")
    meta.append(f"{conversation['message_count']} message(s)")
    if conversation.get("subject"):
        meta.append(f"subject: `{conversation['subject']}`")
    st.caption(" · ".join(meta))

    for message in conversation["messages"]:
        render_message(message)

    question = st.chat_input(
        "Ask anything - research, browse a site, or ask about your documents…"
        if agent_mode
        else "Ask a homework question…"
    )
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
    st.set_page_config(page_title="Lelouch - An AI Helper", page_icon="🤖", layout="centered")

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
