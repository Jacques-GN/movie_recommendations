import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# --- Charger et entraîner le modèle au démarrage ---
ratings = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
modele = SVD()
modele.fit(trainset)

def obtenir_recommandations():
    try:
        user_id = int(entry_user.get())
    except ValueError:
        messagebox.showerror("Erreur", "Entrez un numéro d'utilisateur valide.")
        return

    films_vus = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    tous_films = movies_df['movieId'].tolist()
    films_non_vus = [f for f in tous_films if f not in films_vus]

    preds = [modele.predict(user_id, fid) for fid in films_non_vus[:5000]]
    preds.sort(key=lambda x: x.est, reverse=True)

    # Afficher les résultats
    listbox.delete(0, tk.END)
    for pred in preds[:10]:
        titre = movies_df[movies_df['movieId'] == pred.iid]['title'].values
        if len(titre) > 0:
            listbox.insert(tk.END, f"⭐ {pred.est:.1f}  {titre[0]}")

# --- Interface graphique ---
fenetre = tk.Tk()
fenetre.title("🎬 Système de Recommandation de Films")
fenetre.geometry("500x400")
fenetre.configure(bg="#1a1a2e")

tk.Label(fenetre, text="🎬 Film Recommender", font=("Arial", 16, "bold"),
         bg="#1a1a2e", fg="white").pack(pady=15)

tk.Label(fenetre, text="Numéro d'utilisateur :", bg="#1a1a2e", fg="white").pack()
entry_user = tk.Entry(fenetre, font=("Arial", 12), width=10, justify='center')
entry_user.pack(pady=5)

tk.Button(fenetre, text="Obtenir mes recommandations",
          command=obtenir_recommandations,
          bg="#e94560", fg="white", font=("Arial", 11, "bold"),
          padx=10, pady=5).pack(pady=10)

tk.Label(fenetre, text="Top 10 films recommandés :", bg="#1a1a2e", fg="#aaa").pack()
listbox = tk.Listbox(fenetre, width=60, height=12, bg="#16213e", fg="white",
                     font=("Arial", 10), selectbackground="#e94560")
listbox.pack(pady=5, padx=20)

fenetre.mainloop()