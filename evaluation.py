import pandas as pd
from surprise import SVD, KNNBasic, Dataset, Reader
from surprise.model_selection import cross_validate

ratings = pd.read_csv("data/ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

print("=== ÉVALUATION COMPARATIVE (5-fold cross-validation) ===\n")

# --- Modèle 1 : SVD (filtrage collaboratif)
print("▶ SVD (Décomposition en valeurs singulières)")
resultats_svd = cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
print(f"  RMSE moyen : {resultats_svd['test_rmse'].mean():.4f}")
print(f"  MAE moyen  : {resultats_svd['test_mae'].mean():.4f}\n")

# --- Modèle 2 : KNN (plus proches voisins)
print("▶ KNN (K plus proches voisins)")
resultats_knn = cross_validate(KNNBasic(), data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
print(f"  RMSE moyen : {resultats_knn['test_rmse'].mean():.4f}")
print(f"  MAE moyen  : {resultats_knn['test_mae'].mean():.4f}\n")

print("=== INTERPRÉTATION ===")
print("RMSE faible = meilleures prédictions")
print("Un RMSE < 1.0 est généralement considéré comme bon sur MovieLens")