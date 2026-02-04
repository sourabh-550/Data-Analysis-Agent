"""Streamlit UI for the IT Helpdesk Chatbot."""

from __future__ import annotations

import streamlit as st

from chatbot import ITHelpdeskChatbot

st.set_page_config(page_title="IT Helpdesk Chatbot", page_icon="💻", layout="centered")

st.title("💻 IT Helpdesk Chatbot")
st.caption("Describe your issue and get guided troubleshooting steps.")


@st.cache_resource
def load_bot() -> ITHelpdeskChatbot:
    return ITHelpdeskChatbot("data/ticket_history.csv")


bot = load_bot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.subheader("Session")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Share the exact error message if possible")
    st.markdown("- Mention the device or app name")
    st.markdown("- Example: *Wifi is not working*, *Keyboard not working*, *Cannot login to email*")


# Render messages (show category only for bot messages)
def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "category" in message:
            st.caption(f"**Category:** {message['category']}")


# Show chat history
for message in st.session_state.messages:
    render_message(message)

# Chat input
user_input = st.chat_input("Type your issue here...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get bot response (dict from chatbot.py)
    response = bot.get_response(user_input)

    # Add bot message (show message + category)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["message"],
            "category": response["category"],
        }
    )

    st.rerun()
