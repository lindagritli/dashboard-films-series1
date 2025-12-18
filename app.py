import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


df = pd.read_csv("C:/Users/Guide Info/Desktop/miniprojetpython/titles_cleaned.csv")
st.subheader("📋 Dataset nettoyé")
st.dataframe(df) 

col1, col2 = st.columns(2)

with col1:
    st.metric("📄 Nombre de lignes", df.shape[0])

with col2:
    st.metric("📊 Nombre de colonnes", df.shape[1])


st.set_page_config(page_title="Dashboard Films/Séries", layout="wide")
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
st.title("🎬 Dashboard Films et Séries")



st.subheader("📊 Statistiques descriptives")
num_cols = ['release_year', 'runtime', 'seasons', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
stats_df = pd.DataFrame({
    "Moyenne": df[num_cols].mean(),
    "Médiane": df[num_cols].median(),
    "Écart-type": df[num_cols].std(),
    "Variance": df[num_cols].var()
})
st.dataframe(stats_df)


if "active_chart" not in st.session_state:
    st.session_state.active_chart = None


def show_chart(name):
    st.session_state.active_chart = name


st.subheader("📈 Choisissez un graphique")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🎥 IMDb Score"): show_chart("imdb")
with col2:
    if st.button("⏱ Runtime"): show_chart("runtime")
with col3:
    if st.button("📅 Titres par année"): show_chart("year")
with col4:
    if st.button("🎭 Genres"): show_chart("genres")

col5, col6, col7, col8 = st.columns(4)
with col5:
    if st.button("🌍 Top pays IMDb"): show_chart("countries")
with col6:
    if st.button("🎬 Films vs Séries"): show_chart("type")
with col7:
    if st.button("📈 IMDb par année"): show_chart("score_year")
with col8:
    if st.button("🔢 Corrélation"): show_chart("corr")

col9, col10 = st.columns(2)
with col9:
    if st.button("📊 Scatterplots"): show_chart("scatter")
with col10:
    if st.button("🌟 WordCloud"): show_chart("wordcloud")


if st.session_state.active_chart == "imdb":
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df['imdb_score'], bins=20, kde=True, color="skyblue", ax=ax)
    ax.set_title("Distribution IMDb Score")
    st.pyplot(fig)

elif st.session_state.active_chart == "runtime":
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df['runtime'], bins=20, kde=True, color="lightgreen", ax=ax)
    ax.set_title("Distribution Runtime")
    st.pyplot(fig)

elif st.session_state.active_chart == "year":
    year_counts = df['release_year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", color="purple", ax=ax)
    ax.set_title("Nombre de titres par année")
    st.pyplot(fig)

elif st.session_state.active_chart == "genres":
    # Séparer les genres en lignes individuelles
    df_genres = df.explode('genres')
    
    # Top 10 genres les plus fréquents
    top_genres_pie = df_genres['genres'].value_counts().head(10)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Générer le camembert
    ax.pie(
        top_genres_pie.values, 
        labels=top_genres_pie.index, 
        autopct='%1.1f%%', 
        colors=sns.color_palette('tab10')
    )
    
    # Titre
    ax.set_title("Top 10 genres - répartition en pourcentage")
    
    # Afficher dans Streamlit
    st.pyplot(fig)


elif st.session_state.active_chart == "countries":
    df_countries = df.explode('production_countries')
    mean_score_country = df_countries.groupby('production_countries')['imdb_score'].mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12,5))
    sns.barplot(x=mean_score_country.values, y=mean_score_country.index, palette="magma", ax=ax)
    ax.set_title("Top 15 pays par score IMDb")
    st.pyplot(fig)

elif st.session_state.active_chart == "type":
    type_score = df.groupby('type')['imdb_score'].mean()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=type_score.index, y=type_score.values, palette="Set2", ax=ax)
    ax.set_title("Score IMDb moyen : Films vs Séries")
    st.pyplot(fig)

elif st.session_state.active_chart == "score_year":
    mean_score_year = df.groupby('release_year')['imdb_score'].mean()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(x=mean_score_year.index, y=mean_score_year.values, marker="o", color="red", ax=ax)
    ax.set_title("Évolution du score IMDb moyen par année")
    st.pyplot(fig)

elif st.session_state.active_chart == "corr":
    corr = df[['imdb_score','tmdb_score','tmdb_popularity','runtime','imdb_votes']].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)

elif st.session_state.active_chart == "scatter":
    for x_col, y_col, palette in [("imdb_score","tmdb_score","Set1"), ("tmdb_popularity","imdb_score","Set2"), ("runtime","imdb_score","Set3")]:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue="type", palette=palette, alpha=0.6, ax=ax)
        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig)

elif st.session_state.active_chart == "wordcloud":
    text = " ".join(df['description'].dropna().tolist())
    if text.strip() == "":
        st.warning("Aucune description disponible pour générer le WordCloud.")
    else:
        wordcloud = WordCloud(background_color='black', width=1920, height=1080).generate(text)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
