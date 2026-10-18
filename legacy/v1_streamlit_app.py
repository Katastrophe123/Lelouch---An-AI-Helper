"""v1 of the AI Homework Helper, kept verbatim for reference.

DO NOT RUN THIS AS THE APPLICATION - it is here so the README's v1 -> v2
comparison points at real code rather than a description. It is excluded from
ruff, mypy, the test suite and both Docker images.

What v1 was
-----------
A single Streamlit script that:
  * built a LangChain ``HuggingFaceEndpoint`` wrapper around
    ``meta-llama/Meta-Llama-3-8B-Instruct``
  * kept ``SystemMessage`` / ``HumanMessage`` / ``AIMessage`` history in
    ``st.session_state``
  * sent the whole history to the model on every turn and printed the reply

Everything it could not do is what v2 exists to fix:
  * **No persistence.** A browser refresh cleared ``session_state``, so the
    conversation was gone. Nothing was queryable afterwards.
  * **No API.** The UI *was* the application; no other client could use it, and
    the model credentials lived in the same process as the UI.
  * **No verification.** Whatever the model said went straight to the student.
    A confidently wrong answer was indistinguishable from a correct one.
  * **No retry or fallback.** One transient Hugging Face 503 produced a stack
    trace in the browser.
  * **No tests.** There was nothing to test against; prompt, state, transport
    and rendering were one function.
"""

from __future__ import annotations

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

SYSTEM_PROMPT = (
    "You are a patient, encouraging homework tutor for school and early "
    "university students. You explain reasoning rather than only giving "
    "answers, and you never invent facts, formulas or citations."
)


def build_llm() -> ChatHuggingFace:
    """One LangChain wrapper, constructed on every script rerun."""
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        task="text-generation",
        max_new_tokens=900,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=endpoint)


def main() -> None:
    st.set_page_config(page_title="AI Homework Helper", page_icon="📘")
    st.title("📘 AI Homework Helper")

    # The entire application state. Lost on refresh, invisible to everything
    # outside this browser tab.
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            continue
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    question = st.chat_input("Ask a homework question...")
    if not question:
        return

    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        # No retry, no validation, no persistence: whatever comes back is shown.
        response = build_llm().invoke(st.session_state.messages)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))


if __name__ == "__main__":
    main()
