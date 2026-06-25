// Fonctions utilitaires
function showSection(sectionId) {
    // Masquer tous les sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Afficher la section sélectionnée
    document.getElementById(sectionId).classList.add('active');
    
    // Charger les données appropriées
    if (sectionId === 'home') {
        loadTopMovies();
    } else if (sectionId === 'genres') {
        loadGenres();
    }
}

function closeModal() {
    document.getElementById('movieModal').style.display = 'none';
}

function showMovieDetails(movie) {
    const modal = document.getElementById('movieModal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
        <div style="text-align: center;">
            <h2>${movie.title}</h2>
            <p style="color: #aaa; margin: 15px 0;">${movie.genres || 'Genres non spécifiés'}</p>
            ${movie.rating ? `<p style="font-size: 18px; color: #e94560;">⭐ ${movie.rating}/5</p>` : ''}
            ${movie.count ? `<p style="color: #aaa;">👥 ${movie.count} avis</p>` : ''}
            ${movie.similarity ? `<p style="font-size: 16px; color: #e94560;">🎯 Similarité: ${movie.similarity}</p>` : ''}
        </div>
    `;
    
    modal.style.display = 'block';
}

function createMovieCard(movie) {
    const card = document.createElement('div');
    card.className = 'movie-card';
    card.onclick = () => showMovieDetails(movie);
    
    card.innerHTML = `
        <div class="movie-poster">🎬</div>
        <div class="movie-info">
            <div class="movie-title">${movie.title}</div>
            <div class="movie-meta">
                ${movie.rating ? `<span class="movie-rating">⭐ ${movie.rating}</span>` : ''}
                ${movie.similarity ? `<span style="color: #e94560;">🎯 ${movie.similarity}</span>` : ''}
            </div>
        </div>
    `;
    
    return card;
}

// API Calls
function loadTopMovies() {
    const container = document.getElementById('topMoviesContainer');
    container.innerHTML = '<div class="loading"><div class="spinner"></div> Chargement...</div>';
    
    fetch('/api/top-movies')
        .then(response => response.json())
        .then(data => {
            container.innerHTML = '';
            if (data.top.length === 0) {
                container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #aaa;">Aucun film trouvé</p>';
            } else {
                data.top.forEach(movie => {
                    container.appendChild(createMovieCard(movie));
                });
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #e94560;">Erreur lors du chargement</p>';
        });
}

function loadGenres() {
    const container = document.getElementById('genresContainer');
    container.innerHTML = '<div class="loading"><div class="spinner"></div> Chargement...</div>';
    
    fetch('/api/genres')
        .then(response => response.json())
        .then(data => {
            container.innerHTML = '';
            if (data.genres.length === 0) {
                container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #aaa;">Aucun genre trouvé</p>';
            } else {
                data.genres.forEach(genre => {
                    const card = document.createElement('div');
                    card.className = 'genre-card';
                    card.textContent = genre;
                    card.onclick = () => loadMoviesByGenre(genre);
                    container.appendChild(card);
                });
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #e94560;">Erreur lors du chargement</p>';
        });
}

function loadMoviesByGenre(genre) {
    const genresContainer = document.getElementById('genresContainer');
    const moviesContainer = document.getElementById('genreMoviesContainer');
    
    genresContainer.style.display = 'none';
    moviesContainer.style.display = 'block';
    moviesContainer.innerHTML = `<h3 style="grid-column: 1/-1; margin-bottom: 20px;">Films: ${genre}</h3><div style="grid-column: 1/-1;"><button class="btn-secondary" onclick="showSection('genres')">← Retour aux catégories</button></div>`;
    
    fetch(`/api/movies-by-genre/${genre}`)
        .then(response => response.json())
        .then(data => {
            moviesContainer.innerHTML = `<h3 style="grid-column: 1/-1; margin-bottom: 20px;">Films: ${genre}</h3><button class="btn-secondary" onclick="showSection('genres')" style="grid-column: 1/-1; margin-bottom: 20px;">← Retour aux catégories</button>`;
            if (data.movies.length === 0) {
                moviesContainer.innerHTML += '<p style="grid-column: 1/-1; text-align: center; color: #aaa;">Aucun film trouvé</p>';
            } else {
                const grid = document.createElement('div');
                grid.className = 'movies-grid';
                grid.style.gridColumn = '1 / -1';
                data.movies.forEach(movie => {
                    grid.appendChild(createMovieCard(movie));
                });
                moviesContainer.appendChild(grid);
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            moviesContainer.innerHTML += '<p style="grid-column: 1/-1; text-align: center; color: #e94560;">Erreur lors du chargement</p>';
        });
}

function getRecommendations() {
    const userId = document.getElementById('userId').value;
    const container = document.getElementById('recommendationsContainer');
    
    if (!userId) {
        alert('Veuillez entrer un numéro d\'utilisateur');
        return;
    }
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div> Calcul des recommandations...</div>';
    
    fetch('/api/recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            user_id: userId,
            method: 'collaborative'
        })
    })
        .then(response => response.json())
        .then(data => {
            container.innerHTML = '';
            if (data.recommendations.length === 0) {
                container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #aaa;">Aucune recommandation trouvée pour cet utilisateur</p>';
            } else {
                data.recommendations.forEach(movie => {
                    container.appendChild(createMovieCard(movie));
                });
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #e94560;">Erreur lors du chargement</p>';
        });
}

// Recherche
document.getElementById('searchInput')?.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        const query = this.value.trim();
        if (query) {
            searchMovies(query);
        }
    }
});

document.querySelector('.search-btn')?.addEventListener('click', function() {
    const query = document.getElementById('searchInput').value.trim();
    if (query) {
        searchMovies(query);
    }
});

function searchMovies(query) {
    window.currentSearchQuery = query;
    const container = document.getElementById('recommendationsContainer');
    
    if (container.parentElement.id === 'recommendations') {
        showSection('recommendations');
    }
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div> Recherche...</div>';
    
    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            container.innerHTML = '';
            if (data.results.length === 0) {
                container.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: #aaa;">Aucun film trouvé pour "${query}"</p>`;
            } else {
                container.innerHTML = `<h3 style="grid-column: 1/-1; margin-bottom: 20px;">Résultats pour "${query}" (${data.results.length} trouvé${data.results.length > 1 ? 's' : ''})</h3>`;
                const grid = document.createElement('div');
                grid.className = 'movies-grid';
                grid.style.gridColumn = '1 / -1';
                data.results.forEach(movie => {
                    grid.appendChild(createMovieCard(movie));
                });
                container.appendChild(grid);
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #e94560;">Erreur lors du chargement</p>';
        });
}

// Fermer le modal en cliquant en dehors
window.onclick = function(event) {
    const modal = document.getElementById('movieModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
};

// Charger les films populaires au démarrage
window.addEventListener('load', () => {
    loadTopMovies();
});
