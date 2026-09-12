import streamlit as st
from openai import OpenAI

st.title("Lab 3 - Streaming Chatbot with Memory")

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

model = "gpt-5-nano"

SYSTEM_PROMPT = (
    "You are a friendly tutor talking to a 10-year-old. "
    "Explain everything in simple words and short sentences. "
    "After you answer, always end your message by asking exactly: "
    "'Do you want more info?' "
    "If the user says yes, give more detail on the same topic, then ask "
    "'Do you want more info?' again. "
    "If the user says no, say something cheerful and ask what else you can help with."
)

# Conversation buffer: how many user turns to remember
BUFFER_TURNS = 2

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build what we send: system prompt + last N turns (user + assistant pairs)
    buffer = st.session_state.messages[-(BUFFER_TURNS * 2):]
    payload = [{"role": "system", "content": SYSTEM_PROMPT}] + buffer

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            messages=payload,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
