from flask import Flask, render_template, request, jsonify
import pandas as pd
from surprise import SVD, Dataset, Reader
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# --- Charger les données ---
# Sur Vercel, on essaie de charger les données si elles existent
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
ratings = None
movies_df = None
modele_svd = None
similarite = None
tfidf = None

try:
    ratings_file = os.path.join(DATA_PATH, 'ratings.csv')
    movies_file = os.path.join(DATA_PATH, 'movies.csv')
    
    if os.path.exists(ratings_file) and os.path.exists(movies_file):
        ratings = pd.read_csv(ratings_file)
        movies_df = pd.read_csv(movies_file)
        print("✅ Données chargées avec succès")
        
        # --- Entraîner le modèle SVD ---
        try:
            reader = Reader(rating_scale=(0.5, 5.0))
            data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
            trainset = data.build_full_trainset()
            modele_svd = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02)
            modele_svd.fit(trainset)
            print("✅ Modèle SVD entraîné")
        except Exception as e:
            print(f"⚠️ Erreur lors de l'entraînement SVD: {e}")
        
        # --- Filtrage par contenu (TF-IDF) ---
        try:
            movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)
            movies_df['genres'] = movies_df['genres'].fillna('')
            tfidf = TfidfVectorizer()
            matrice_tfidf = tfidf.fit_transform(movies_df['genres'])
            similarite = cosine_similarity(matrice_tfidf, matrice_tfidf)
            print("✅ Modèle TF-IDF entraîné")
        except Exception as e:
            print(f"⚠️ Erreur lors de l'entraînement TF-IDF: {e}")
    else:
        print("⚠️ Fichiers de données non trouvés")
        # Données de démonstration vides
        ratings = pd.DataFrame()
        movies_df = pd.DataFrame()
except Exception as e:
    print(f"⚠️ Erreur lors du chargement des données: {e}")
    ratings = pd.DataFrame()
    movies_df = pd.DataFrame()


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
    method = data.get('method', 'collaborative')
    
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


@app.route('/api/status')
def get_status():
    """API: Vérifier le statut de l'application"""
    return jsonify({
        'status': 'online',
        'data_loaded': not ratings.empty and not movies_df.empty,
        'movies_count': len(movies_df) if not movies_df.empty else 0,
        'ratings_count': len(ratings) if not ratings.empty else 0
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000)
