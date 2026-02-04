import streamlit as st
from chatbot import ITHelpdeskChatbot

st.set_page_config(page_title="IT Helpdesk Chatbot", page_icon="💻")

st.title("IT Helpdesk Chatbot")
st.write("Ask your IT issue and get instant help!")

@st.cache_resource
def load_bot():
    return ITHelpdeskChatbot("data/ticket_history.csv")

bot = load_bot()

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Enter your issue:")

if st.button("Get Solution"):
    if user_input.strip():
        response = bot.get_response(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))
    else:
        st.warning("Please enter a problem.")

st.write("###Conversation")
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
