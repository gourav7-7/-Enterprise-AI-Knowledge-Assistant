"""Streamlit UI — a pure HTTP client of the FastAPI backend.

This UI holds NO business logic and NO secrets. Every action (register, login,
upload, query, history, feedback) is an HTTP call to the FastAPI service. The
JWT returned by login lives in st.session_state and is sent as a Bearer token
on every authenticated request.

Run the backend first:
    uvicorn app.main:app --reload --port 8000
Then run this UI (separate terminal):
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120  # seconds — LLM-backed /query can be slow


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
def _auth_headers() -> dict:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _extract_error(response: requests.Response) -> str:
    """Pull a readable message from either error shape the API returns."""
    try:
        data = response.json()
    except Exception:
        return f"HTTP {response.status_code}"
    if isinstance(data, dict):
        # custom AppError handler -> {"error": {"message": ...}}
        if isinstance(data.get("error"), dict):
            return data["error"].get("message", "Unknown error")
        # FastAPI defaults -> {"detail": ...} (401) or [{"msg": ...}] (422)
        detail = data.get("detail")
        if isinstance(detail, list) and detail:
            return detail[0].get("msg", "Validation error")
        if detail is not None:
            return str(detail)
    return f"HTTP {response.status_code}"


def api_request(method: str, path: str, **kwargs):
    """Single choke point: injects auth, handles connection errors and 401."""
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    try:
        response = requests.request(
            method, f"{API_BASE_URL}{path}", headers=headers, timeout=REQUEST_TIMEOUT, **kwargs
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE_URL}. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error("The request timed out. Try again.")
        return None

    # token expired/invalid mid-session -> force re-login
    if response.status_code == 401 and st.session_state.get("token"):
        st.session_state.pop("token", None)
        st.warning("Your session expired. Please log in again.")
        st.rerun()
    return response


# --------------------------------------------------------------------------- #
# API actions
# --------------------------------------------------------------------------- #
def do_register(username: str, password: str) -> bool:
    r = api_request("POST", "/auth/register",
                    json={"username": username, "password": password})
    if r is None:
        return False
    if r.status_code == 201:
        st.success("Account created — log in on the other tab.")
        return True
    st.error(_extract_error(r))
    return False


def do_login(username: str, password: str) -> bool:
    # /auth/login uses OAuth2PasswordRequestForm -> form-encoded, NOT json
    r = api_request("POST", "/auth/login",
                    data={"username": username, "password": password})
    if r is None:
        return False
    if r.status_code == 200:
        st.session_state.token = r.json()["access_token"]
        st.session_state.username = username
        return True
    st.error(_extract_error(r))
    return False


def do_upload(file) -> dict | None:
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    r = api_request("POST", "/upload", files=files)
    if r is None:
        return None
    if r.status_code == 200:
        return r.json()
    st.error(_extract_error(r))
    return None


def do_query(question: str) -> dict | None:
    r = api_request("POST", "/query", json={"question": question})
    if r is None:
        return None
    if r.status_code == 200:
        return r.json()
    st.error(_extract_error(r))
    return None


def do_history() -> list:
    r = api_request("GET", "/history")
    if r is None or r.status_code != 200:
        return []
    return r.json()


def do_feedback(question: str, answer: str, rating: int, comment: str) -> bool:
    r = api_request("POST", "/feedback", json={
        "question": question, "answer": answer,
        "rating": rating, "comment": comment or None,
    })
    if r is None:
        return False
    if r.status_code == 201:
        return True
    st.error(_extract_error(r))
    return False


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        seen = set()
        for s in sources:
            page = s.get("page")
            tag = (s.get("source"), page)
            if tag in seen:
                continue
            seen.add(tag)
            page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
            st.markdown(f"- **{s.get('source', 'unknown')}** {page_str} — {s.get('snippet', '')}")


def feedback_widget(index: int, message: dict) -> None:
    if message.get("feedback_sent"):
        st.caption("✓ Feedback submitted")
        return
    with st.expander("Rate this answer"):
        with st.form(f"fb_{index}"):
            rating = st.slider("Rating", 1, 5, 4)
            comment = st.text_input("Comment (optional)")
            if st.form_submit_button("Submit feedback"):
                if do_feedback(message.get("question", ""), message.get("content", ""),
                               rating, comment):
                    message["feedback_sent"] = True
                    st.rerun()


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
def login_screen() -> None:
    st.title("📄 AI Knowledge Assistant")
    st.caption("Log in or create an account to continue.")
    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", use_container_width=True) and do_login(u, p):
                st.rerun()

    with tab_register:
        with st.form("register_form"):
            u = st.text_input("Username (3–50 chars)")
            p = st.text_input("Password (min 6 chars)", type="password")
            if st.form_submit_button("Register", use_container_width=True):
                do_register(u, p)


def main_app() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_doc", None)

    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.get('username', '')}**")
        if st.button("Log out", use_container_width=True):
            api_request("POST", "/auth/logout")  # stateless: server just acknowledges
            for k in ("token", "username", "messages", "active_doc"):
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()
        st.header("Document")
        uploaded = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded is not None and st.button("Upload & ingest", use_container_width=True):
            with st.spinner("Uploading and ingesting..."):
                summary = do_upload(uploaded)
            if summary:
                st.session_state.active_doc = summary["filename"]
                st.success(
                    f"Ingested {summary['filename']} "
                    f"({summary['pages']} pages, {summary['chunks']} chunks)"
                )
        if st.session_state.active_doc:
            st.caption(f"Last ingested: {st.session_state.active_doc}")

        st.divider()
        show_history = st.checkbox("Show my history")

    st.title("📄 AI Knowledge Assistant")
    st.caption("Answers are grounded in your uploaded PDFs. Each question is answered independently.")

    if show_history:
        with st.expander("My past questions (from the server)", expanded=True):
            history = do_history()
            if not history:
                st.caption("No history yet.")
            for item in history:
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                if item.get("created_at"):
                    st.caption(item["created_at"])
                st.divider()

    # render the conversation so far (with per-answer feedback)
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources", []))
                feedback_widget(i, m)

    # new question
    prompt = st.chat_input("Ask a question about your uploaded documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = do_query(prompt)
            if result:
                st.markdown(result["answer"])
                render_sources(result.get("sources", []))
                new_index = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "question": prompt,
                })
                feedback_widget(new_index, st.session_state.messages[-1])


def main() -> None:
    st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📄")
    if st.session_state.get("token"):
        main_app()
    else:
        login_screen()


if __name__ == "__main__":
    main()