import os
import streamlit as st
from dotenv import load_dotenv
import openai

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-5-mini"

st.title("🎵🎶Auralis🎵🎶")
st.write("🎵🎶Bienvenido a Auralis, tu recomendador de música.🎵🎶")

if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': "Hello! I'm Auralis, your AI-powered music recommendation system. How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input(placeholder="Escribe algo para que te recomiende una canción"):
    st.session_state.messages.append({'role': 'user', 'content': prompt})   
    st.chat_message("user").write(prompt)

    conversation = [{'role': 'system', 'content': "You are a helpful assistant that recommends songs based on the user's prompt."}]
    conversation.extend({'role': m['role'], 'content': m["content"] for m in st.session_state.messages})
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(model=model,messages=conversation, stream=True)
        response = st.write_stream(stream)
 