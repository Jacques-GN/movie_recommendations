# 🎬 NetflixFlix - Interface Web Style Netflix

## Description

Cette interface web style Netflix permet de découvrir des films recommandés en utilisant des algorithmes de machine learning :
- **Filtrage collaboratif** (SVD) pour des recommandations personnalisées
- **Filtrage par contenu** (TF-IDF) pour des films similaires
- **Interface moderne** inspirée de Netflix

## 🚀 Installation

### 1. Installez les dépendances

```bash
pip install -r requirements.txt
```

### 2. Préparation des données

Assurez-vous d'avoir les fichiers de données MovieLens :
```
data/
  ├── ratings.csv
  └── movies.csv
```

### 3. Lancez l'application

```bash
python app.py
```

L'application sera accessible à : **http://localhost:5000**

## 📋 Fonctionnalités

### 🎬 Accueil
- Affiche les 20 films les plus populaires
- Design moderne avec grille responsive

### 👤 Recommandations Personnalisées
- Entrez votre numéro d'utilisateur
- Recevez 12 films recommandés basés sur le filtrage collaboratif (SVD)
- Note prédite affichée pour chaque film

### 📽️ Parcourir par Catégories
- Sélectionnez un genre de film
- Parcourez tous les films de cette catégorie
- Navigation facile entre les genres

### 🔍 Recherche
- Barre de recherche dans la navigation
- Recherchez des films par titre
- Résultats instantanés

## 🎨 Design

### Couleurs
- **Fond** : Noir (`#0f0f0f`)
- **Primaire** : Gris foncé (`#1a1a2e`, `#16213e`)
- **Accent** : Rouge Netflix (`#e94560`)
- **Texte** : Blanc et gris

### Composants
- Navigation sticky avec logo et barre de recherche
- Hero section avec gradient
- Grille de films responsive
- Cards animées avec hover effects
- Modal pour les détails du film

## 📱 Responsive

L'interface s'adapte à tous les appareils :
- **Desktop** : Grille 6-8 colonnes
- **Tablet** : Grille 4-5 colonnes
- **Mobile** : Grille 2-3 colonnes

## 🔌 API Endpoints

```
GET  /                          # Page d'accueil
GET  /api/top-movies            # Films populaires
GET  /api/genres                # Liste des genres
GET  /api/movies-by-genre/<g>  # Films d'un genre
GET  /api/search?q=<query>     # Recherche de films
POST /api/recommendations       # Recommandations personnalisées
POST /api/similar-movies        # Films similaires
```

## 📊 Technologies

- **Backend** : Flask (Python)
- **ML** : Scikit-learn, Surprise (SVD)
- **Frontend** : HTML5, CSS3, JavaScript (Vanilla)
- **Données** : MovieLens Dataset

## 🎯 Prochaines améliorations possibles

- [ ] Authentification utilisateur
- [ ] Sauvegarde des films favoris
- [ ] Affichage des affiches réelles
- [ ] Intégration de vidéos de bande annonce
- [ ] System de notation utilisateur
- [ ] Recommandations en temps réel
- [ ] Déploiement (Heroku, AWS, etc.)

## 📝 Notes

- L'interface teste le frontend et backend ensemble
- Les données sont chargées une seule fois au démarrage
- Le modèle SVD est entraîné au lancement (peut prendre quelques secondes)
- Utilisez des numéros d'utilisateur entre 1 et le nombre d'utilisateurs du dataset

---

**Branche de test** : `testing` 🧪
