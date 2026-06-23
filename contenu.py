import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les films
movies = pd.read_csv("data/movies.csv")

# Remplacer les '|' entre genres par des espaces (pour TF-IDF)
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
movies['genres'] = movies['genres'].fillna('')

# Créer la matrice TF-IDF des genres
tfidf = TfidfVectorizer()
matrice_tfidf = tfidf.fit_transform(movies['genres'])

# Calculer la similarité entre tous les films
similarite = cosine_similarity(matrice_tfidf, matrice_tfidf)

# Index des titres
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommander_par_contenu(titre_film, n=5):
    """Recommande des films similaires en termes de genres."""
    if titre_film not in indices:
        print(f"Film '{titre_film}' non trouvé.")
        return []
    idx = indices[titre_film]
    scores = list(enumerate(similarite[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]  # on exclut le film lui-même
    resultats = []
    for i, score in scores:
        resultats.append((movies.iloc[i]['title'], round(score, 3)))
    return resultats

# Test
print("--- Films similaires à 'Toy Story (1995)' ---")
similaires = recommander_par_contenu("Toy Story (1995)", n=5)
for titre, score in similaires:
    print(f"  🎯 Similarité {score}  →  {titre}")