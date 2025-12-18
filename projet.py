import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
# Lecture du dataset
df = pd.read_csv("C:/Users/Guide Info/Downloads/titles.csv")
nb_lignes, nb_colonnes = df.shape

print("Nombre de lignes :", nb_lignes)
print("Nombre de colonnes :", nb_colonnes)

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df_numeriques = df.select_dtypes(include='number')
print(df_numeriques)


#  Nettoyage des valeurs manquantes , remplacer les valurs 
df['title'] = df['title'].fillna("Missing Title")
df['description'] = df['description'].fillna("No Description Available")#permet de générer le WordCloud sans erreur
df['age_certification'] = df['age_certification'].fillna("Unrated")
df['seasons'] = df['seasons'].fillna(0).astype(int) #Mettre 0 signifie ,ce n’est pas une série




# filtrer Colonnes numériques à analyser
num_cols = ['release_year', 'runtime', 'seasons', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']


# Remplacer les NaN par la médiane pour que les calculs soient corrects,car la mediane n’est pas influencée par les valeurs extrêmes
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Médiane
median_values = df[num_cols].median()
print("Médiane des colonnes :\n", median_values, "\n")

# Variance
variance_values = df[num_cols].var()
print("Variance des colonnes :\n", variance_values, "\n")

# Écart-type
std_values = df[num_cols].std()
print("Écart-type des colonnes :\n", std_values)



# --- Conversion des strings en listes ---
def to_list_safe(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df['genres'] = df['genres'].apply(to_list_safe) #ça transforme le texte inutile en une liste que Python peut manipuler facilement
df['production_countries'] = df['production_countries'].apply(to_list_safe)

# --- Détection & correction des outliers ---
def fix_outliers_median(df, col):
    median = df[col].median()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df.loc[(df[col] < lower) | (df[col] > upper), col] = median

for col in ['runtime', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']:
    fix_outliers_median(df, col)

# --- Supprimer les doublons ---
df = df.drop_duplicates(subset=['title', 'release_year'], keep='first')

# --- Vérification finale ---
print(df.isna().sum())
print("\nCleaning finished.")
print("Shape final :", df.shape)
print(df.head())

# --- Style graphique ---
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

#  Distribution IMDb,note des films 
plt.figure(figsize=(8,5))
sns.histplot(df['imdb_score'], bins=20, kde=True, color="skyblue")
plt.title("Distribution IMDb Score")
plt.xlabel("IMDb Score") #C’est la note des films
plt.ylabel("Nombre de titres") #C’est combien de films ont cette note
plt.show()

#  Distribution Runtime,Savoir si la majorité des films sont courts, moyens ou longs.
plt.figure(figsize=(8,5))
sns.histplot(df['runtime'], bins=20, kde=True, color="lightgreen")
plt.title("Distribution Runtime")
plt.xlabel("Minutes")
plt.ylabel("Nombre de titres")
plt.show()

#  Titres par année,combien de films sortent chaque année et comment ça change dans le temps 
plt.figure(figsize=(10,5))
sns.lineplot(x=df['release_year'].value_counts().sort_index().index,
             y=df['release_year'].value_counts().sort_index().values,
             marker="o", color="purple")
plt.title("Nombre de titres par année")
plt.xlabel("Année")
plt.ylabel("Nombre de titres")
plt.show()


#Voir quels genres sont les plus populaires
df_genres = df.explode('genres')

plt.figure(figsize=(8,8))
top_genres_pie = df_genres['genres'].value_counts().head(10)
plt.pie(top_genres_pie.values, labels=top_genres_pie.index, autopct='%1.1f%%', colors=sns.color_palette('tab10'))
plt.title("Top 10 genres - répartition en pourcentage")
plt.show()


#  ce diag en barre horizs montre les 15 pays dont les films ont le meilleur score IMDb moyen.
df_countries = df.explode('production_countries')#transforme chaque pays d’un film en ligne séparée pour qu’on puisse les compter correctement.
mean_score_country = df_countries.groupby('production_countries')['imdb_score'].mean().sort_values(ascending=False).head(15)#Pour chaque pays, on calcule le score IMDb moyen des films produits. etOn trie les pays par score moyen décroissant et on garde les 15 premiers.
plt.figure(figsize=(12,5))
sns.barplot(x=mean_score_country.values, y=mean_score_country.index, palette="magma") #barplot diagramme en barres horizontales
plt.title("Top 15 pays par score IMDb")
plt.xlabel("Score IMDb")
plt.ylabel("Pays")
plt.show()

#  Comparer les scores moyens entre Films et Séries ,On peut voir quel type de contenu a généralement les meilleures notes.
type_score = df.groupby('type')['imdb_score'].mean()
plt.figure(figsize=(6,4))
sns.barplot(x=type_score.index, y=type_score.values, palette="Set2")
plt.title("Score IMDb moyen : Films vs Séries")
plt.ylabel("Score IMDb")
plt.show()



#Mesure la qualité moyenne des films chaque année (score IMDb moyen)
mean_score_year = df.groupby('release_year')['imdb_score'].mean()
plt.figure(figsize=(10,5))
sns.lineplot(x=mean_score_year.index, y=mean_score_year.values, marker="o", color="red")
plt.title("Évolution du score IMDb moyen par année")
plt.xlabel("Année")
plt.ylabel("Score IMDb")
plt.show()

#  Matrice de corrélation,Voir quelles variables sont liées entre elles.
plt.figure(figsize=(8,6))
corr = df[['imdb_score','tmdb_score','tmdb_popularity','runtime','imdb_votes']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.show()

#  Scatterplots colorés par type
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='imdb_score', y='tmdb_score', hue='type', palette="Set1", alpha=0.6)
plt.title("IMDb vs TMDB Score par type")
plt.xlabel("IMDb Score")
plt.ylabel("TMDB Score")
plt.show()

#Scatterplots voir si les films populaires ont tendance à avoir de meilleures notes.

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='tmdb_popularity', y='imdb_score', hue='type', palette="Set2", alpha=0.6)
plt.title("Popularité TMDB vs IMDb Score par type")
plt.xlabel("TMDB Popularity")
plt.ylabel("IMDb Score")
plt.show()


#analyse pour voir si les films trop longs ou trop courts ont des scores plus faibles
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='runtime', y='imdb_score', hue='type', palette="Set3", alpha=0.6)
plt.title("Runtime vs IMDb Score par type")
plt.xlabel("Runtime (min)")
plt.ylabel("IMDb Score")
plt.show()



#  Boxplot pour visualiser les outliers
plt.figure(figsize=(8,5))
sns.boxplot(x='type', y='runtime', data=df, palette="pastel")
plt.title("Distribution du runtime par type (Films vs Séries)")
plt.show()




st.subheader("🌟 WordCloud des descriptions des films/séries")

# Générer le texte à partir des descriptions
text = " ".join(df['description'].dropna().tolist()) #Prend toutes les descriptions de films/séries dans le dataset,transforme la liste de phrases en un seul gros texte


# Vérifier que le texte n'est pas vide
if text.strip() == "":
    st.warning("Aucune description disponible pour générer le WordCloud.")
else:
    # Créer le WordCloud
    wordcloud = WordCloud(
        background_color='black',
        width=800,   # réduire la largeur
        height=400,  # réduire la hauteur
        colormap='viridis'
    ).generate(text)

    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")  # supprimer les axes
    st.pyplot(fig)






plt.title("📊 Répartition Films + Séries par Année", y=1.08)
plt.show()



# Sauvegarder le dataset nettoyé
df.to_csv(r"C:\Users\Guide Info\Desktop\miniprojetpython\titles_cleaned.csv", index=False)
print("Fichier nettoyé sauvegardé !")



