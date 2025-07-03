import streamlit as st
import requests

# Rasa server URL (make sure your Rasa is running with REST channel)
RASA_URL = "http://localhost:5002/webhooks/rest/webhook"

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Chat with Rasa Bot")

# Display chat history
for msg in st.session_state.messages:
    if msg["sender"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant").write(msg["text"])

# Input box at the bottom
prompt = st.chat_input("Type your message here...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"sender": "user", "text": prompt})
    st.chat_message("user").write(prompt)

    # Send message to Rasa bot
    try:
        response = requests.post(
            RASA_URL,
            json={"sender": "streamlit-user", "message": prompt}
        )

        if response.status_code == 200:
            bot_messages = response.json()
            for bot_msg in bot_messages:
                bot_text = bot_msg.get("text")
                if bot_text:
                    st.session_state.messages.append({"sender": "bot", "text": bot_text})
                    st.chat_message("assistant").write(bot_text)
        else:
            error_msg = "Error: Unable to reach Rasa server"
            st.session_state.messages.append({"sender": "bot", "text": error_msg})
            st.chat_message("assistant").write(error_msg)
    except Exception as e:
        st.chat_message("assistant").write("Server error: " + str(e))
