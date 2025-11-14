import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import openai

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-5-mini"

st.title("🎵🎶Auralis🎵🎶")
st.write("🎵🎶Bienvenido a Auralis, tu recomendador de música.🎵🎶")

with st.form("survey_form"):
    st.write("Por favor, responda las siguientes preguntas:")

    # Pregunta 1: Entrada de texto
    nombre = st.text_input("¿Cuál es tu nombre?")

    # Pregunta 2: Opción múltiple (radio)
    experiencia = st.radio(
        "¿Cuánta experiencia tienes con Streamlit?",
        ('Ninguna', 'Básica', 'Intermedia', 'Avanzada')
    )

    # Pregunta 3: Checkbox
    interes_ml = st.checkbox("¿Te interesa el Machine Learning?")

    # Botón de envío para el formulario
    submitted = st.form_submit_button("Enviar Respuestas")

if submitted:
    if nombre:
        st.success(f"¡Gracias, {nombre}! Respuestas enviadas.")
        # Aquí puedes procesar los datos (por ejemplo, guardar en una base de datos o archivo CSV)
        data = {"Nombre": nombre, "Experiencia": experiencia, "Interés ML": interes_ml}
        st.write("Datos recopilados:")
        st.write(pd.DataFrame([data]))
    else:
        st.warning("🎵🎶Por favor, introduce tu nombre antes de enviar.🎵🎶")