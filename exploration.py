import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
ratings = pd.read_csv("data/ratings.csv")
movies  = pd.read_csv("data/movies.csv")

print("=== APERÇU DES DONNÉES ===")
print(f"Nombre de notes     : {len(ratings)}")
print(f"Nombre d'utilisateurs : {ratings['userId'].nunique()}")
print(f"Nombre de films     : {ratings['movieId'].nunique()}")
print(f"\nNote moyenne        : {ratings['rating'].mean():.2f} / 5")

print("\n--- Premières lignes ratings ---")
print(ratings.head())

print("\n--- Premières lignes movies ---")
print(movies.head())

# Distribution des notes
ratings['rating'].value_counts().sort_index().plot(
    kind='bar', title='Distribution des notes', color='steelblue'
)
plt.xlabel("Note")
plt.ylabel("Nombre de fois")
plt.tight_layout()
plt.savefig("distribution_notes.png")
plt.show()

# Top 10 films les plus notés
top_films = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False).head(10)
top_films = top_films.reset_index()
top_films = top_films.merge(movies[['movieId','title']], on='movieId')
print("\n--- Top 10 films les plus notés ---")
print(top_films[['title','rating']])