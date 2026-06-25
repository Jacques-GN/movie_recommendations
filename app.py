from flask import Flask, render_template, request, jsonify
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# --- Charger les données ---
try:
    ratings = pd.read_csv("data/ratings.csv")
    movies_df = pd.read_csv("data/movies.csv")
    print("✅ Données chargées avec succès")
except FileNotFoundError:
    print("❌ Les fichiers data/ratings.csv ou data/movies.csv sont manquants")
    ratings = pd.DataFrame()
    movies_df = pd.DataFrame()

# --- Entraîner le modèle SVD ---
if not ratings.empty and not movies_df.empty:
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    modele_svd = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02)
    modele_svd.fit(trainset)
    print("✅ Modèle SVD entraîné")
else:
    modele_svd = None
    print("⚠️ Modèle SVD non entraîné (données manquantes)")

# --- Filtrage par contenu (TF-IDF) ---
if not movies_df.empty:
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf = TfidfVectorizer()
    try:
        matrice_tfidf = tfidf.fit_transform(movies_df['genres'])
        similarite = cosine_similarity(matrice_tfidf, matrice_tfidf)
        print("✅ Modèle TF-IDF entraîné")
    except:
        print("⚠️ Erreur dans le modèle TF-IDF")
        similarite = None
        tfidf = None
else:
    similarite = None
    tfidf = None


def obtenir_recommandations_collaboratif(user_id, n=10):
    """Recommandations basées sur le filtrage collaboratif (SVD)"""
    if modele_svd is None or ratings.empty or movies_df.empty:
        return []
    
    try:
        films_vus = ratings[ratings['userId'] == user_id]['movieId'].tolist()
        tous_films = movies_df['movieId'].tolist()
        films_non_vus = [f for f in tous_films if f not in films_vus]
        
        predictions = [modele_svd.predict(user_id, fid) for fid in films_non_vus[:5000]]
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        resultats = []
        for pred in predictions[:n]:
            titre = movies_df[movies_df['movieId'] == pred.iid]['title'].values
            if len(titre) > 0:
                resultats.append({
                    'title': titre[0],
                    'rating': round(pred.est, 1),
                    'movieId': pred.iid
                })
        return resultats
    except Exception as e:
        print(f"Erreur recommandations collaboratif: {e}")
        return []


def obtenir_recommandations_contenu(titre_film, n=10):
    """Recommandations basées sur le filtrage par contenu"""
    if similarite is None or movies_df.empty:
        return []
    
    try:
        indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
        if titre_film not in indices.index:
            return []
        
        idx = indices[titre_film]
        scores = list(enumerate(similarite[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]
        
        resultats = []
        for i, score in scores:
            resultats.append({
                'title': movies_df.iloc[i]['title'],
                'similarity': round(float(score), 3),
                'movieId': int(movies_df.iloc[i]['movieId'])
            })
        return resultats
    except Exception as e:
        print(f"Erreur recommandations contenu: {e}")
        return []


def obtenir_top_films(n=20):
    """Récupère les films les plus populaires"""
    if ratings.empty or movies_df.empty:
        return []
    
    try:
        top_films = ratings.groupby('movieId')['rating'].agg(['count', 'mean']).sort_values('count', ascending=False).head(n)
        resultats = []
        for movie_id in top_films.index:
            titre = movies_df[movies_df['movieId'] == movie_id]['title'].values
            if len(titre) > 0:
                resultats.append({
                    'title': titre[0],
                    'rating': round(float(top_films.loc[movie_id, 'mean']), 1),
                    'count': int(top_films.loc[movie_id, 'count']),
                    'movieId': int(movie_id)
                })
        return resultats
    except Exception as e:
        print(f"Erreur top films: {e}")
        return []


def obtenir_films_par_genre(genre, n=20):
    """Récupère les films d'un genre spécifique"""
    if movies_df.empty:
        return []
    
    try:
        films_genre = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)].head(n)
        resultats = []
        for _, row in films_genre.iterrows():
            resultats.append({
                'title': row['title'],
                'genres': row['genres'],
                'movieId': int(row['movieId'])
            })
        return resultats
    except Exception as e:
        print(f"Erreur films par genre: {e}")
        return []


def rechercher_films(query, n=20):
    """Recherche des films par titre"""
    if movies_df.empty:
        return []
    
    try:
        resultats_search = movies_df[movies_df['title'].str.contains(query, case=False, na=False)].head(n)
        resultats = []
        for _, row in resultats_search.iterrows():
            resultats.append({
                'title': row['title'],
                'genres': row['genres'],
                'movieId': int(row['movieId'])
            })
        return resultats
    except Exception as e:
        print(f"Erreur recherche: {e}")
        return []


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """API: Obtenir les recommandations"""
    data = request.json
    user_id = data.get('user_id')
    method = data.get('method', 'collaborative')  # 'collaborative' ou 'content'
    
    if method == 'collaborative' and user_id:
        recommendations = obtenir_recommandations_collaboratif(int(user_id), n=12)
    else:
        recommendations = []
    
    return jsonify({'recommendations': recommendations})


@app.route('/api/similar-movies', methods=['POST'])
def get_similar_movies():
    """API: Obtenir les films similaires"""
    data = request.json
    title = data.get('title')
    
    if title:
        similar = obtenir_recommandations_contenu(title, n=12)
    else:
        similar = []
    
    return jsonify({'similar': similar})


@app.route('/api/top-movies')
def get_top_movies():
    """API: Obtenir les films les plus populaires"""
    top = obtenir_top_films(n=20)
    return jsonify({'top': top})


@app.route('/api/movies-by-genre/<genre>')
def get_movies_by_genre(genre):
    """API: Obtenir les films par genre"""
    movies = obtenir_films_par_genre(genre, n=20)
    return jsonify({'movies': movies})


@app.route('/api/search')
def search():
    """API: Rechercher des films"""
    query = request.args.get('q', '')
    results = rechercher_films(query, n=20)
    return jsonify({'results': results})


@app.route('/api/genres')
def get_genres():
    """API: Obtenir la liste des genres"""
    if movies_df.empty:
        return jsonify({'genres': []})
    
    genres_list = set()
    for genres_str in movies_df['genres']:
        if pd.notna(genres_str):
            genres_list.update(genres_str.split())
    
    return jsonify({'genres': sorted(list(genres_list))})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
