import streamlit as st
from chatbot import ITHelpdeskChatbot

st.set_page_config(page_title="IT Helpdesk Chatbot", page_icon="💻")

st.title("💻 IT Helpdesk Chatbot")
st.write("Chat with the IT support bot and get help for your issues!")

# Load bot only once
@st.cache_resource
def load_bot():
    return ITHelpdeskChatbot("data/ticket_history.csv")

bot = load_bot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")

# Input form (auto clears after submit)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip() != "":
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

    elif submitted:
        st.warning("Please type something!")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
