import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Charger les données
ratings = pd.read_csv("data/ratings.csv")

# Format attendu par Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Diviser : 80% entraînement / 20% test
trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

# Entraîner le modèle SVD
print("Entraînement du modèle SVD...")
modele = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02)
modele.fit(trainset)

# Évaluer sur le jeu de test
predictions = modele.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE du modèle SVD : {rmse:.4f}")

# Recommander des films pour un utilisateur donné
def recommander_films(user_id, n=10):
    movies = pd.read_csv("data/movies.csv")
    # Films déjà vus par l'utilisateur
    films_vus = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    # Tous les films
    tous_films = movies['movieId'].tolist()
    # Films non encore vus
    films_non_vus = [f for f in tous_films if f not in films_vus]
    # Prédire la note pour chaque film non vu
    predictions_user = [modele.predict(user_id, film_id) for film_id in films_non_vus]
    # Trier par note prédite décroissante
    predictions_user.sort(key=lambda x: x.est, reverse=True)
    # Récupérer les N meilleurs
    top_n = predictions_user[:n]
    resultats = []
    for pred in top_n:
        titre = movies[movies['movieId'] == pred.iid]['title'].values[0]
        resultats.append((titre, round(pred.est, 2)))
    return resultats

# Test avec l'utilisateur n°1
print("\n--- Recommandations pour l'utilisateur 1 ---")
reco = recommander_films(user_id=1, n=5)
for titre, note in reco:
    print(f"  ⭐ {note}/5  →  {titre}")