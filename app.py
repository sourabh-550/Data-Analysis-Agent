import streamlit as st
from chatbot import ITHelpdeskChatbot

st.set_page_config(page_title="IT Helpdesk Chatbot", page_icon="💻")

st.title("IT Helpdesk Chatbot")
st.write("Chat with the IT support bot and get help for your issues!")

# Load bot only once
@st.cache_resource
def load_bot():
    return ITHelpdeskChatbot("data/ticket_history.csv")

bot = load_bot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Input box at bottom
user_input = st.text_input("Type your message and press Enter:", key="user_input")

# Send button
if st.button("Send"):
    if user_input.strip() != "":
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Get bot reply
        reply = bot.get_response(user_input)

        # Add bot message
        st.session_state.messages.append({
            "role": "bot",
            "content": reply
        })

        # Clear input box
        st.session_state.user_input = ""

    else:
        st.warning("Please type something!")

# Clear chat button
if st.button("🗑️Clear Chat"):
    st.session_state.messages = []
