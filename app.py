"""
=============================================================
  SYSTÈME DE RECOMMANDATION DE FILMS — Flask Backend
  Dataset : MovieLens | L1 Informatique - UJKZ
=============================================================
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES AU DÉMARRAGE
# ─────────────────────────────────────────────
print("⏳ Chargement des données...")
ratings   = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")

# Entraînement SVD
reader   = Reader(rating_scale=(0.5, 5.0))
data     = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
modele   = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02)
modele.fit(trainset)
print("✅ Modèle SVD entraîné !")

# Filtrage par contenu (TF-IDF sur genres)
movies_df['genres_clean'] = movies_df['genres'].str.replace('|', ' ', regex=False).fillna('')
tfidf        = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df['genres_clean'])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices      = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
print("✅ Modèle TF-IDF prêt !")

# Couleurs par genre
GENRE_COLORS = {
    'Action':      '#C0392B',
    'Comedy':      '#D35400',
    'Drama':       '#1A5276',
    'Horror':      '#6C3483',
    'Romance':     '#B03A2E',
    'Sci-Fi':      '#117A65',
    'Thriller':    '#784212',  
    'Adventure':   '#B7950B',
    'Fantasy':     '#4A235A',
    'Crime':       '#1B2631',
    'Mystery':     '#0E6655',
    'Music':       '#6E2F8F',
}

def get_genre_color(genres_str):
    for genre, color in GENRE_COLORS.items():
        if genre in str(genres_str):
            return color
    return '#1a1a2e'

def get_primary_genre(genres_str):
    genres = str(genres_str).split('|')
    return genres[0] if genres else 'Unknown'

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    # Top films les mieux notés (min 50 votes)
    popular = ratings.groupby('movieId')['rating'].agg(['mean','count'])
    popular = popular[popular['count'] >= 50].sort_values('mean', ascending=False).head(24)
    popular = popular.reset_index().merge(movies_df, on='movieId')
    popular_list = []
    for _, row in popular.iterrows():
        popular_list.append({
            'title':  row['title'],
            'genres': row['genres'],
            'rating': round(row['mean'], 1),
            'count':  int(row['count']),
            'color':  get_genre_color(row['genres']),
            'genre_label': get_primary_genre(row['genres'])
        })
    stats = {
        'nb_films':  int(movies_df['movieId'].nunique()),
        'nb_users':  int(ratings['userId'].nunique()),
        'nb_notes':  len(ratings),
        'moy_note':  round(float(ratings['rating'].mean()), 2)
    }
    return render_template('index.html', popular_movies=popular_list, stats=stats)


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.json.get('user_id', 1))
        n       = int(request.json.get('n', 12))

        if user_id not in ratings['userId'].values:
            return jsonify({'error': f"Utilisateur {user_id} introuvable.", 'recommendations': []})

        films_vus    = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
        films_non_vus = [f for f in movies_df['movieId'].tolist() if f not in films_vus]

        preds = [modele.predict(user_id, fid) for fid in films_non_vus[:4000]]
        preds.sort(key=lambda x: x.est, reverse=True)

        results = []
        for pred in preds[:n]:
            row = movies_df[movies_df['movieId'] == pred.iid]
            if not row.empty:
                g = row.iloc[0]['genres']
                results.append({
                    'title':       row.iloc[0]['title'],
                    'genres':      g,
                    'score':       round(pred.est, 2),
                    'color':       get_genre_color(g),
                    'genre_label': get_primary_genre(g)
                })

        return jsonify({'recommendations': results, 'user_id': user_id,
                        'nb_vus': len(films_vus)})
    except Exception as e:
        return jsonify({'error': str(e), 'recommendations': []})


@app.route('/similar', methods=['POST'])
def similar():
    try:
        title = request.json.get('title', '').strip()
        n     = int(request.json.get('n', 12))

        if title not in indices:
            # Cherche le plus proche
            match = movies_df[movies_df['title'].str.lower().str.contains(
                title.lower(), na=False)].head(1)
            if match.empty:
                return jsonify({'error': f"Film '{title}' non trouvé.", 'results': []})
            title = match.iloc[0]['title']

        idx    = indices[title]
        scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]

        results = []
        for i, score in scores:
            row = movies_df.iloc[i]
            g   = row['genres']
            results.append({
                'title':       row['title'],
                'genres':      g,
                'similarity':  round(float(score) * 100, 1),
                'color':       get_genre_color(g),
                'genre_label': get_primary_genre(g)
            })

        return jsonify({'results': results, 'base_film': title})
    except Exception as e:
        return jsonify({'error': str(e), 'results': []})


@app.route('/search')
def search():
    query   = request.args.get('q', '').lower()
    results = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)].head(8)
    return jsonify({'results': results[['movieId','title','genres']].to_dict('records')})


if __name__ == '__main__':
    print("\n🎬 Interface disponible sur : http://localhost:5000\n")
    app.run(debug=True, port=5000)