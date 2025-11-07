import streamlit as st


st.title("Auralis")
st.write("Welcome to Auralis, the AI-powered music recommendation system.")

prompt = st.chat_input("Enter your prompt here")

if prompt:
    st.write(f"You: {prompt}")
    st.write("Auralis: Generating music...")
    st.write("Auralis: Here is the music for your prompt.")
    st.write("Auralis: Enjoy!")
