# For Production
import os
import streamlit as st
import random
import requests
import base64
import numpy as np
import pandas as pd
import time
# Standardization
from sklearn.preprocessing import RobustScaler
# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
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
    df = pd.read_parquet("data_model/Auralis_MusicRecommender2.parquet")
    return df


df_perfil = load_data_artist_level()


st.title("🎵🎶 Auralis 🎵🎶")
st.write("🎵🎶Bienvenido a **Auralis**, tu recomendador de música.🎵🎶")

with st.form("survey_form"):
    # st.write("Por favor, responda las siguientes preguntas:")

    # Pregunta: Artista
    nombre_artista = st.text_input("¿De qué artista te gustaría recibir una recomendación?")

    # Botón de envío para el formulario
    submitted = st.form_submit_button("Enviar")

if submitted:
    if nombre_artista:
        st.success(f"Procesando información del artista: {nombre_artista} ...")
        with st.spinner("Espere un momento...", show_time=True):
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

            ################################# Similar Artist #################################
            feature_cols = [
                "Tracks/Artista", "danceability", "energy", "loudness",
                "speechiness", "acousticness", "instrumentalness", "liveness",
                "valence", "tempo", "duration_seg", "#Colaboraciones",
                "Indice de sentimiento", "Colaboraciones_x_Track",
                "Albums_per_Artists", "organic_score", "tempo_duration_ratio"
            ]
            X_artist = artist_filter[feature_cols].values

            Rscaler = RobustScaler()
            X_artist_scaled = Rscaler.fit_transform(X_artist)

            # Extraer el vector del artista objetivo
            artist_target = artist_name

            vec_target = artist_filter.loc[
                artist_filter["main_artist"] == artist_target,
                feature_cols
            ].values

            # escalarlo también
            vec_target_scaled = Rscaler.transform(vec_target)
            # Calculamos la distancia coseno
            similarities = cosine_similarity(vec_target_scaled, X_artist_scaled)[0]

            artist_similarity = artist_filter.copy()
            artist_similarity["Similitud"] = similarities
            artist_similarity = artist_similarity.sort_values(by="Similitud", ascending=False)
            artist_similarity.reset_index(inplace=True, drop=True)

        # DataFrame con ancho del contenedor
        st.subheader("Artistas más cercanos:")
        st.dataframe(artist_similarity[["main_artist", "Similitud"]][artist_similarity["Similitud"]>=0.4], width="stretch", hide_index = True)
        
        artist_similarity = artist_similarity[artist_similarity["Similitud"]>=0.4]
        with st.spinner("Procesando datos...", show_time=True):
        ################################# API Spotify #################################
            unique_artists = artist_similarity['main_artist'].unique().tolist()
            auth_response = requests.post(auth_url, {
                "grant_type": "client_credentials",
                "client_id": Spotify_API_CLIENT_KEY,
                "client_secret": Spotify_API_CLIENT_SECRET_KEY,
            })
            access_token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}

            spotify_artist_data = []

            for unique_artist_name in unique_artists:
                search_url = "https://api.spotify.com/v1/search"
                params = {"q": unique_artist_name, "type": "artist", "limit": 1}
                try:
                    search_response = requests.get(search_url, headers=headers, params=params)
                    search_response.raise_for_status()
                    artist_data = search_response.json()["artists"]["items"][0]
                    genres = artist_data.get("genres", [])
                    image_url = artist_data["images"][0]["url"] if artist_data.get("images") and artist_data["images"] else None
                    spotify_artist_data.append({"main_artist": unique_artist_name, "genres": ", ".join(genres), "image_url": image_url})
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for artist {unique_artist_name}: {e}")
                time.sleep(0.333)

        df_spotify_artist_data = pd.DataFrame(spotify_artist_data)

        artist_filter = pd.merge(artist_similarity, df_spotify_artist_data, on='main_artist', how='left')

        # Network Graph
        with st.container(border=True):
            st.subheader("Artistas altamente relacionados:")

            # ============================
            #     PYVIS GRAPH SECTION
            # ============================

            pyvis_nodes = []
            pyvis_edges = []
            unique_songs_with_images = {}
            song_to_album_map = {}

            df_graph = artist_filter.copy()

            # Create mapping nodes and edges
            for _, row in df_graph.iterrows():
                artist_name_row = row['main_artist']
                cluster_group = row['Grupos K-Means']
                image_url = row['image_url']

                if artist_name_row not in unique_songs_with_images:
                    unique_songs_with_images[artist_name_row] = image_url

                if cluster_group not in song_to_album_map:
                    song_to_album_map[cluster_group] = []
                song_to_album_map[cluster_group].append(artist_name_row)

            # Create Pyvis nodes
            for artist_name_row, image_url in unique_songs_with_images.items():
                pyvis_nodes.append({
                    'id': artist_name_row,
                    'label': artist_name_row,
                    'shape': 'image',
                    'image': image_url
                })

            # Create Pyvis edges
            for cluster_group, artists_in_group in song_to_album_map.items():
                if len(artists_in_group) > 1:
                    for i in range(len(artists_in_group)):
                        for j in range(i + 1, len(artists_in_group)):
                            pyvis_edges.append({
                                'source': artists_in_group[i],
                                'target': artists_in_group[j]
                            })

            # Build Pyvis network
            net = Network(
                height="500px",
                width="100%",
                bgcolor="#23282b",
                notebook=False,
                cdn_resources="in_line"
            )

            for node in pyvis_nodes:
                net.add_node(
                    node['id'],
                    label=node['label'],
                    shape=node['shape'],
                    image=node['image']
                )

            for edge in pyvis_edges:
                net.add_edge(edge['source'], edge['target'],color="#1DB954")

            net.set_options("""
                var options = {
                "nodes": {
                    "shapeProperties": {
                        "borderDashes": false,
                        "useBorderWithImage": false
                    },
                    "borderWidth": 0,
                    "color": {
                    "background": "#23282b",
                    "border": "#23282b",
                    "highlight": {
                        "background": "#23282b",
                        "border": "#1DB954"
                    }
                    },
                    "shapeProperties": {
                    "useBorderWithImage": false
                    },
                    "font": {
                    "color": "white"
                    }
                },
                "edges": {
                    "color": {
                    "color": "#1DB954",
                    "highlight": "#1DB954",
                    "inherit": false
                    },
                    "smooth": {
                    "type": "dynamic"
                    }
                },
                "interaction": {
                    "hover": true,
                    "zoomView": true
                },
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.09
                    }
                }
                }
                """)

            graph_html = "pyvis_graph.html"
            net.write_html(graph_html)

            with open(graph_html, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.components.v1.html(html_content, height=500, scrolling=False)

    else:
        st.warning("🎵🎶Por favor, introduce el nombre del artista antes de enviar.🎵🎶")