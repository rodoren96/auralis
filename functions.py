# For Production
import os
import streamlit as st
import numpy as np
import pandas as pd
import time
from math import pi
# Standardization
from sklearn.preprocessing import RobustScaler
# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
# Environment Variables
from dotenv import load_dotenv
# Structured Outputs
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
# Visualization
from PIL import Image
from PIL import ImageDraw
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Si Pillow >= 10 eliminó textsize, lo reimplementamos
if not hasattr(ImageDraw.ImageDraw, "textsize"):
   def textsize(self, text, font=None, *args, **kwargs):
       bbox = self.textbbox((0, 0), text, font=font, *args, **kwargs)
       return (bbox[2] - bbox[0], bbox[3] - bbox[1])
   ImageDraw.ImageDraw.textsize = textsize
from stylecloud import gen_stylecloud

load_dotenv(override=True)

## AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# Clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_groq = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
client_google = OpenAI(api_key=GOOGLE_API_KEY, base_url=GEMINI_BASE_URL)
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
# AI Models
## OpenAI
model_openai = "gpt-5-mini"
## Llama
model_groq = "llama-3.3-70b-versatile"
## Llama
model_groq_so = "meta-llama/llama-4-maverick-17b-128e-instruct"
## Gemini
model_google = "gemini-2.5-flash"
## deepseek
model_deepseek = "deepseek-chat"
################################# Structured Outputs para Lyrics #################################
class EarningsCallInsights(BaseModel):
    """
    Output estructurado con insights clave en español extraídos de una earnings call.
    """
    sentiment: Optional[str] = Field(description="Sentimiento general dominante: miedo, tristeza, ira, alegría, amor, neutral")
    key_topics: List[str] = Field(default_factory=list, description="Temas principales de la canción, cada uno con máximo 4 palabras")
    summary: Optional[str] = Field(description="Resumen en máximo 2 oraciones sobre el contenido de la letra")
    explicit_themes: List[str] = Field(default_factory=list,description="Temas explícitos o sensibles presentes, como drogas, violencia, sexualidad, etc.")
    lyrical_complexity: Optional[str] = Field(description="Complejidad lírica: simple, moderada, compleja")
    emotional_palette: List[str] = Field(default_factory=list,description="Lista de emociones presentes: nostalgia, anhelo, vulnerabilidad, esperanza, enojo, euforia, etc.")
    relationship_context: Optional[str] = Field(description="Contexto emocional de una relación: ruptura, amor correspondido, amor no correspondido, duelo, deseo, nostalgia")

def get_earnings_call_insights(client, call_songs: str, model_name: str = "gpt-5-mini") -> EarningsCallInsights:
    """
    Obtiene insights clave de una transcripción de earnings call utilizando un modelo de OpenAI.

    Args:
        client: Cliente de OpenAI inicializado.
        transcript_text: El texto de la transcripción de la llamada.
        model_name: El nombre del modelo de OpenAI a utilizar.

    Returns:
        Un objeto diccionario derivado de EarningsCallInsights con los insights extraídos.
    """
    response = client.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": "Eres un experto analizando letras de canciones musicales. Devuelve SOLO un JSON válido que siga exactamente el esquema de EarningsCallInsights. Salidas en español."},
            {"role": "user", "content": call_songs},
        ],
        response_format=EarningsCallInsights,
    )
    insights = response.choices[0].message.parsed
    return insights.model_dump()
################################# Spinner con Fun Facts #################################
def get_fun_fact_from_gpt(artist: str) -> str:
    prompt = f"Dame un dato curioso, interesante y poco conocido sobre el artista de música {artist}. Solo uno, máximo 5 líneas. Que ningún dato curioso se repita."
    
    response = client_deepseek.chat.completions.create(
        model=model_deepseek,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.9
    )

    return response.choices[0].message.content.strip()

def show_gpt_fun_facts_spinner(artist: str, duration_minutes: int = 5):
    duration_seconds = duration_minutes*60
    start_time = time.time()

    with st.spinner(f"Obteniendo insights para las lyrics de {artist}…"):
        placeholder = st.empty()

        while time.time() - start_time < duration_seconds:
            fun_fact = get_fun_fact_from_gpt(artist)
            placeholder.write(fun_fact)
            
            time.sleep(30)

        placeholder.empty()
################################# Insights de Lyrics #################################
def get_embedding(text):
    resp = client_openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding
################################# Similitud de Artistas #################################
def compute_artist_similarity(df, target_artist):
    # Extraer embeddings en matriz
    vectors = np.vstack(df["embedding"].values)
    similarity_matrix = cosine_similarity(vectors)
    # Lista de artistas
    artists = df["main_artist"].tolist()
    # Índice del artista target
    idx = artists.index(target_artist)
    # Similaridades contra todos
    similarities = similarity_matrix[idx]
    # Crear nueva columna en el dataframe
    df = df.copy()
    df[f"Similitud por Lyrics"] = similarities
    return df
################################# Radar plot (Emotional Palette) #################################
def plot_emotional_radar(emotions_list):
    emotions = list(emotions_list)
    values = [1] * len(emotions)

    N = len(emotions)
    if N == 0:
        st.warning("Este artista no contiene suficientes emociones para generar un radar.")
        return

    angles = [n / float(N) * 2 * pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions)
    ax.set_yticklabels([])
    st.pyplot(fig)
################################# Wordcloud (Key Topics) #################################
def get_wordcloud(text, icon="fas fa-cloud", background_color=None, output_name="./wordcloud.png"):
    gen_stylecloud(text=text, icon_name=icon, background_color=background_color, output_name=output_name)
    return Image.open(output_name)
def plot_wordcloud(topics, background_color=None):
    if not topics:
        st.warning("No hay key topics para generar la nube.")
        return
    text = " ".join(topics)
    output_name = "wordcloud_streamlit.png"
    # Generacion de Nube
    img = get_wordcloud(text=text,background_color=background_color,output_name=output_name)
    st.image(img)
def centered_image_with_subheader(image_path, subheader_text):
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="{image_path}" style="max-width: 100%; border-radius: 8px; margin-bottom: 6px;">
            <h3 style="margin: 0; padding: 0;">{subheader_text}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
################################# Storytelling AI #################################
def generate_storytelling(row):
    prompt = f"""
    Eres un analista musical. A partir de este perfil lírico:

    Sentiment: {row.sentiment}
    Lyrical complexity: {row.lyrical_complexity}
    Explicit themes: {row.explicit_themes}
    Emotional palette: {row.emotional_palette}
    Relationship context: {row.relationship_context}
    Key topics: {row.key_topics}

    Genera un análisis narrativo de máximo 4 frases sobre el estilo emocional,
    lírico y temático del artista. Escríbelo en tono artistico.
    """

    resp = client_groq.chat.completions.create(
        model=model_groq,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=180
    )

    return resp.choices[0].message.content.strip()

def combine_similarities(df):
    # Penalización por diferencia
    df["penalty"] = 1 - abs(df["Similitud por Features"] - df["Similitud por Lyrics"])

    # Score final combinado con ponderadores
    w_num = 0.1
    w_emb = 0.9

    df["Similitud Combinada"] = (
        df["Similitud por Features"] * w_num +
        df["Similitud por Lyrics"] * w_emb
    ) * df["penalty"]

    return df