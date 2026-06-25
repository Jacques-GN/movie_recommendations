"""
=============================================================
  ÉVALUATION DU SYSTÈME DE RECOMMANDATION DE FILMS
  Dataset : MovieLens | L1 Informatique - UJKZ
  Métriques : RMSE, MAE, Précision@N, Rappel@N
=============================================================
"""

import pandas as pd
from collections import defaultdict
from surprise import SVD, KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("=" * 55)
print("  CHARGEMENT DES DONNÉES")
print("=" * 55)

ratings = pd.read_csv("data/ratings.csv")

print(f"  Nombre de notes       : {len(ratings)}")
print(f"  Nombre d'utilisateurs : {ratings['userId'].nunique()}")
print(f"  Nombre de films       : {ratings['movieId'].nunique()}")
print(f"  Note moyenne          : {ratings['rating'].mean():.2f} / 5")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Division : 75% entraînement / 25% test
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)


# ─────────────────────────────────────────────
# 2. FONCTION PRÉCISION & RAPPEL
# ─────────────────────────────────────────────
def calculer_precision_rappel(predictions, seuil_note=3.5, n=10):
    """
    Calcule Précision@N et Rappel@N pour un ensemble de prédictions.

    - seuil_note : note minimale pour qu'un film soit 'pertinent' (défaut 3.5/5)
    - n          : nombre de recommandations considérées (défaut 10)

    Précision@N = (films pertinents dans le top N) / N
    Rappel@N    = (films pertinents dans le top N) / (total films pertinents)
    """
    # Regrouper les prédictions par utilisateur
    user_estimations = defaultdict(list)
    for pred in predictions:
        user_estimations[pred.uid].append((pred.est, pred.r_ui))

    precisions = []
    rappels = []

    for uid, notes in user_estimations.items():
        # Trier par note prédite décroissante
        notes.sort(key=lambda x: x[0], reverse=True)
        top_n = notes[:n]

        # Films bien prédits ET réellement appréciés
        pertinents_recommandes = sum(
            1 for (est, reel) in top_n
            if est >= seuil_note and reel >= seuil_note
        )
        # Total des films réellement appréciés par cet utilisateur
        total_pertinents = sum(
            1 for (est, reel) in notes if reel >= seuil_note
        )

        precision = pertinents_recommandes / n if n > 0 else 0
        rappel = (pertinents_recommandes / total_pertinents
                  if total_pertinents > 0 else 0)

        precisions.append(precision)
        rappels.append(rappel)

    precision_moy = sum(precisions) / len(precisions) if precisions else 0
    rappel_moy    = sum(rappels)    / len(rappels)    if rappels    else 0

    return precision_moy, rappel_moy


# ─────────────────────────────────────────────
# 3. MÉTHODE 1 — SVD (Filtrage Collaboratif)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MÉTHODE 1 : SVD — Filtrage Collaboratif")
print("=" * 55)

modele_svd = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02)
modele_svd.fit(trainset)
predictions_svd = modele_svd.test(testset)

rmse_svd = accuracy.rmse(predictions_svd, verbose=False)
mae_svd  = accuracy.mae(predictions_svd,  verbose=False)
precision_svd, rappel_svd = calculer_precision_rappel(predictions_svd, n=10)

print(f"  RMSE         : {rmse_svd:.4f}")
print(f"  MAE          : {mae_svd:.4f}")
print(f"  Précision@10 : {precision_svd:.4f}  ({precision_svd*100:.1f}%)")
print(f"  Rappel@10    : {rappel_svd:.4f}  ({rappel_svd*100:.1f}%)")


# ─────────────────────────────────────────────
# 4. MÉTHODE 2 — KNN (Filtrage Collaboratif)
#    (comparé au filtrage par contenu dans le rapport)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MÉTHODE 2 : KNN — Plus Proches Voisins")
print("=" * 55)

modele_knn = KNNBasic(k=30, sim_options={'name': 'cosine', 'user_based': True})
modele_knn.fit(trainset)
predictions_knn = modele_knn.test(testset)

rmse_knn = accuracy.rmse(predictions_knn, verbose=False)
mae_knn  = accuracy.mae(predictions_knn,  verbose=False)
precision_knn, rappel_knn = calculer_precision_rappel(predictions_knn, n=10)

print(f"  RMSE         : {rmse_knn:.4f}")
print(f"  MAE          : {mae_knn:.4f}")
print(f"  Précision@10 : {precision_knn:.4f}  ({precision_knn*100:.1f}%)")
print(f"  Rappel@10    : {rappel_knn:.4f}  ({rappel_knn*100:.1f}%)")


# ─────────────────────────────────────────────
# 5. TABLEAU COMPARATIF FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  COMPARAISON FINALE DES DEUX MÉTHODES")
print("=" * 55)

print(f"""
┌─────────────────┬──────────────┬──────────────┐
│ Métrique        │     SVD      │     KNN      │
├─────────────────┼──────────────┼──────────────┤
│ RMSE            │   {rmse_svd:.4f}   │   {rmse_knn:.4f}   │
│ MAE             │   {mae_svd:.4f}   │   {mae_knn:.4f}   │
│ Précision@10    │   {precision_svd:.4f}   │   {precision_knn:.4f}   │
│ Rappel@10       │   {rappel_svd:.4f}   │   {rappel_knn:.4f}   │
└─────────────────┴──────────────┴──────────────┘
""")

# Conclusion automatique
if rmse_svd < rmse_knn:
    meilleur = "SVD"
    raison   = "RMSE plus faible = prédictions plus précises"
else:
    meilleur = "KNN"
    raison   = "RMSE plus faible = prédictions plus précises"

print(f"  ✅ Meilleure méthode : {meilleur}")
print(f"  📌 Raison            : {raison}")


# ─────────────────────────────────────────────
# 6. INTERPRÉTATION (pour le rapport)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  INTERPRÉTATION DES RÉSULTATS")
print("=" * 55)
print(f"""
  RMSE :
    → Un RMSE de {rmse_svd:.2f} (SVD) signifie qu'en moyenne,
      la note prédite s'écarte de {rmse_svd:.2f} point(s) de la note réelle.
    → En dessous de 1.0 = bon résultat sur MovieLens.

  PRÉCISION@10 :
    → Sur 10 films recommandés, {precision_svd*10:.1f} sont vraiment
      pertinents pour l'utilisateur (SVD).

  RAPPEL@10 :
    → Le système retrouve {rappel_svd*100:.1f}% des films
      que l'utilisateur aurait aimés (SVD).

  CONCLUSION :
    → SVD est plus adapté pour ce projet car il gère mieux
      les grandes matrices creuses comme MovieLens.
    → KNN est plus simple mais moins performant sur
      des datasets de grande taille.
""")

print("=" * 55)
print("  Évaluation terminée ✅")
print("=" * 55)