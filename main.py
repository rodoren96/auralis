# For Production
import os
import streamlit as st
import random
import requests
import base64
import numpy as np
import pandas as pd
import time
# Environment Variables
from dotenv import load_dotenv
import openai
# Visualization
import networkx as nx
from pyvis.network import Network

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Spotify_API_CLIENT_KEY = os.getenv("Spotify_API_CLIENT_KEY")
Spotify_API_CLIENT_SECRET_KEY = os.getenv("Spotify_API_CLIENT_SECRET_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-5-mini"

# Load Data
@st.cache_data
def load_data_artist_level():
    df = pd.read_parquet("data_model/Auralis_MusicRecommender.parquet")
    return df


df_perfil = load_data_artist_level()


st.title("🎵🎶 Auralis 🎵🎶")
st.write("🎵🎶Bienvenido a **Auralis**, tu recomendador de música.🎵🎶")

with st.form("survey_form"):
    # st.write("Por favor, responda las siguientes preguntas:")

    # Pregunta 1: Artista
    nombre_artista = st.text_input("¿De qué artista te gustaría recibir una recomendación?")

    # Botón de envío para el formulario
    submitted = st.form_submit_button("Enviar")

if submitted:
    if nombre_artista:
        st.success(f"Procesando información del artista: {nombre_artista} ...")
        st.write("Este proceso puede tardar unos segundos...")
        # Get Spotify Access Token
        auth_url = "https://accounts.spotify.com/api/token"
        auth_response = requests.post(auth_url, {
            "grant_type": "client_credentials",
            "client_id": Spotify_API_CLIENT_KEY,
            "client_secret": Spotify_API_CLIENT_SECRET_KEY,
        })
        access_token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        ticker = nombre_artista
        # Busqueda de Artista
        search_url = "https://api.spotify.com/v1/search"
        params = {"q": ticker, "type": "artist", "limit": 1}
        search_response = requests.get(search_url, headers=headers, params=params)
        search_response.raise_for_status()
        artist_data = search_response.json()["artists"]["items"][0]
        artist_name = artist_data.get("name")

        # Search Artist - Cluster Model
        artist_filter = df_perfil[df_perfil['Grupos K-Means'] == df_perfil[df_perfil['main_artist'] == artist_name]["Grupos K-Means"].iloc[0]]
        # DataFrame con ancho del contenedor
        st.subheader("Resultados en tabla:")
        st.dataframe(artist_filter, use_container_width=True)
        # Network Graph
        st.subheader("Resultados en red:")
        st.graph(artist_filter)
    else:
        st.warning("🎵🎶Por favor, introduce el nombre del artista antes de enviar.🎵🎶")