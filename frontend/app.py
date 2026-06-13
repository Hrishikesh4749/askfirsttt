"""
frontend/app.py
-----------------
Streamlit frontend for the AI Memory Chat application.

Features:
- Sidebar listing all chat threads, with a button to create new ones
- Clicking a thread switches the active conversation and loads its history
- Main panel chat interface with message input + send button
- Talks to the FastAPI backend over HTTP
"""

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Memory Chat", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------------------------------
# API helper functions
# --------------------------------------------------------------------------


def fetch_threads():
    try:
        resp = requests.get(f"{API_URL}/threads", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach backend: {e}")
        return []


def create_thread(title="New Chat"):
    try:
        resp = requests.post(f"{API_URL}/create-thread", json={"title": title}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't create thread: {e}")
        return None


def fetch_messages(thread_id):
    try:
        resp = requests.get(f"{API_URL}/messages/{thread_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't load messages: {e}")
        return []


def send_message(thread_id, message):
    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"thread_id": thread_id, "message": message},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't send message: {e}")
        return None


# --------------------------------------------------------------------------
# Sidebar -- thread list + new thread button
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 AI Memory Chat")
    st.caption("Persistent multi-thread chat with universal memory")

    if st.button("➕ New Chat", use_container_width=True):
        new_thread = create_thread()
        if new_thread:
            st.session_state.current_thread_id = new_thread["id"]
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("Your threads")

    threads = fetch_threads()

    if not threads:
        st.info("No chats yet. Click 'New Chat' to start one!")
    else:
        for thread in threads:
            is_active = thread["id"] == st.session_state.current_thread_id
            label = ("🟢 " if is_active else "") + thread["title"]
            if st.button(label, key=f"thread_{thread['id']}", use_container_width=True):
                st.session_state.current_thread_id = thread["id"]
                st.session_state.messages = fetch_messages(thread["id"])
                st.rerun()

    st.divider()
    st.caption("Memories are shared across ALL threads automatically.")


# --------------------------------------------------------------------------
# Main panel -- chat interface
# --------------------------------------------------------------------------

if st.session_state.current_thread_id is None:
    st.title("Welcome 👋")
    st.write(
        "Start a new chat from the sidebar to begin. The AI will remember "
        "important things about you across **all** your conversations."
    )
else:
    # Load messages on first render of this thread
    if not st.session_state.messages:
        st.session_state.messages = fetch_messages(st.session_state.current_thread_id)

    active_thread = next(
        (t for t in threads if t["id"] == st.session_state.current_thread_id), None
    )
    thread_title = active_thread["title"] if active_thread else "Chat"

    st.title(f"💬 {thread_title}")

    # Render chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Type a message...")

    if user_input:
        # Show user message immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("thinking..."):
                result = send_message(st.session_state.current_thread_id, user_input)

            if result:
                reply = result["reply"]
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

                if result.get("memory_stored"):
                    st.caption(f"🧠 Remembered: _{result['memory_stored']}_")
                if result.get("memories_used"):
                    with st.expander("📎 Memories used for this reply"):
                        for m in result["memories_used"]:
                            st.write(f"- {m}")
            else:
                st.error("Something went wrong getting a response.")

        st.rerun()
