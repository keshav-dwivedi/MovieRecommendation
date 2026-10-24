"""
recommender.py - Movie Recommendation Logic
A simple collaborative filtering recommender system using MovieLens dataset.
"""

import os
import re
import requests
import pandas as pd
import numpy as np

# Path to data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    """Loads and prepares movies and ratings data."""
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # Calculate average rating and total number of ratings per movie
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()

    # Merge stats into movies dataframe
    movies = pd.merge(movies, stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0).round(2)
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)

    return movies, ratings


def build_recommender():
    """Builds pivot table and cosine similarity matrix between movies."""
    movies, ratings = load_data()

    # Merge ratings with movie titles
    df = pd.merge(ratings, movies[["movieId", "title", "genres"]], on="movieId")

    # Filter movies with at least 15 ratings for reliable recommendations
    popular_titles = movies[movies["num_ratings"] >= 15]["title"]
    filtered_df = df[df["title"].isin(popular_titles)]

    # Create user-movie pivot table (rows = movie titles, columns = users)
    pivot_table = filtered_df.pivot_table(
        index="title", columns="userId", values="rating"
    ).fillna(0)

    # Calculate cosine similarity between movies
    matrix = pivot_table.values
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity_matrix = np.dot(matrix, matrix.T) / (np.dot(norms, norms.T) + 1e-9)

    return movies, ratings, pivot_table, similarity_matrix


def get_similar_movies(movie_title, movies, pivot_table, similarity_matrix, top_n=5):
    """Finds top similar movies using cosine similarity."""
    if movie_title not in pivot_table.index:
        return pd.DataFrame()

    movie_idx = pivot_table.index.get_loc(movie_title)
    scores = similarity_matrix[movie_idx]

    # Get top movie indices (excluding the movie itself at index 0)
    top_indices = np.argsort(scores)[::-1][1:top_n + 1]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        title = pivot_table.index[idx]
        sim_score = scores[idx]
        movie_info = movies[movies["title"] == title].iloc[0]

        results.append({
            "Rank": rank,
            "Title": title,
            "Genres": movie_info["genres"],
            "Average Rating": movie_info["avg_rating"],
            "Similarity": round(float(sim_score), 2)
        })

    return pd.DataFrame(results)


def get_user_recommendations(user_id, movies, ratings, pivot_table, similarity_matrix, top_n=5):
    """Recommends movies to a user based on movies they rated highly."""
    user_ratings = ratings[ratings["userId"] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()

    # Find movies rated 4.0 or above by this user
    liked_movies = user_ratings[user_ratings["rating"] >= 4.0]
    if liked_movies.empty:
        liked_movies = user_ratings.sort_values(by="rating", ascending=False)

    watched_movie_ids = set(user_ratings["movieId"])
    top_movie_ids = liked_movies.sort_values(by="rating", ascending=False)["movieId"].head(3)
    favorite_titles = movies[movies["movieId"].isin(top_movie_ids)]["title"].tolist()

    recommendations = []
    for fav_title in favorite_titles:
        sim_df = get_similar_movies(fav_title, movies, pivot_table, similarity_matrix, top_n=top_n)
        for _, row in sim_df.iterrows():
            movie_row = movies[movies["title"] == row["Title"]].iloc[0]
            if movie_row["movieId"] not in watched_movie_ids:
                recommendations.append(row.to_dict())

    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df = rec_df.drop_duplicates(subset=["Title"]).head(top_n)
        rec_df["Rank"] = range(1, len(rec_df) + 1)
        rec_df = rec_df[["Rank", "Title", "Genres", "Average Rating", "Similarity"]]

    return rec_df


def get_popular_movies(movies, top_n=10, genre="All"):
    """Returns top rated popular movies with at least 50 ratings."""
    df = movies[movies["num_ratings"] >= 50].copy()

    if genre and genre != "All":
        df = df[df["genres"].str.contains(genre, na=False)]

    popular = df.sort_values(by="avg_rating", ascending=False).head(top_n)
    popular["Rank"] = range(1, len(popular) + 1)

    return popular[["Rank", "title", "genres", "avg_rating", "num_ratings"]].rename(
        columns={
            "title": "Title",
            "genres": "Genres",
            "avg_rating": "Average Rating",
            "num_ratings": "Total Ratings"
        }
    )


def get_movie_details(movie_title):
    """
    Fetches movie wallpaper/poster, cast, ratings, genres and plot
    using the free OMDb API.
    """
    # Extract release year from title (e.g. 'Toy Story (1995)' -> 1995)
    year_match = re.search(r"\((\d{4})\)", movie_title)
    year = year_match.group(1) if year_match else ""

    # Clean movie title for API search
    clean_name = re.sub(r"\s*\(\d{4}\)", "", movie_title).strip()
    clean_name = re.sub(r"\(a\.k\.a\..*?\)", "", clean_name).strip()

    # Handle titles ending with articles like ', The' or ', A'
    if clean_name.endswith(", The"):
        clean_name = "The " + clean_name[:-5]
    elif clean_name.endswith(", A"):
        clean_name = "A " + clean_name[:-3]
    elif clean_name.endswith(", An"):
        clean_name = "An " + clean_name[:-4]

    default_poster = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=400&q=80"
    api_url = f"http://www.omdbapi.com/?t={clean_name}&y={year}&apikey=b9a5e69d"

    try:
        response = requests.get(api_url, timeout=3).json()
        if response.get("Response") == "True":
            poster_url = response.get("Poster")
            if not poster_url or poster_url == "N/A":
                poster_url = default_poster

            return {
                "title": response.get("Title", movie_title),
                "year": response.get("Year", year),
                "poster": poster_url,
                "plot": response.get("Plot", "No plot description available."),
                "cast": response.get("Actors", "Not specified"),
                "genres": response.get("Genre", "Not specified"),
                "rating": response.get("imdbRating", "N/A"),
                "director": response.get("Director", "Not specified")
            }
    except Exception:
        pass

    # Fallback default values
    return {
        "title": movie_title,
        "year": year,
        "poster": default_poster,
        "plot": "Detailed plot summary currently unavailable.",
        "cast": "Not specified",
        "genres": "Not specified",
        "rating": "N/A",
        "director": "Not specified"
    }
