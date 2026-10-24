"""
test_recommender.py - Simple test script for movie recommendation system.
"""

from recommender import (
    build_recommender,
    get_similar_movies,
    get_user_recommendations,
    get_popular_movies,
    get_movie_details,
)


def main():
    print("=" * 50)
    print("Testing Movie Recommendation System")
    print("=" * 50)

    # 1. Load data and build model
    print("\n1. Building recommendation model...")
    movies, ratings, pivot_table, similarity_matrix = build_recommender()
    print(f"   [OK] Loaded {len(movies)} movies and {len(ratings)} ratings.")
    print(f"   [OK] Pivot table shape: {pivot_table.shape}")

    # 2. Test movie recommendations
    print("\n2. Testing recommendations for 'Toy Story (1995)':")
    recs = get_similar_movies("Toy Story (1995)", movies, pivot_table, similarity_matrix, top_n=5)
    print(recs.to_string(index=False))
    assert not recs.empty, "Failed to get similar movies!"

    # 3. Test user recommendations
    print("\n3. Testing recommendations for User 1:")
    user_recs = get_user_recommendations(1, movies, ratings, pivot_table, similarity_matrix, top_n=5)
    print(user_recs.to_string(index=False))
    assert not user_recs.empty, "Failed to get user recommendations!"

    # 4. Test popular movies
    print("\n4. Testing top popular movies:")
    pop_movies = get_popular_movies(movies, top_n=5)
    print(pop_movies.to_string(index=False))
    assert not pop_movies.empty, "Failed to get popular movies!"

    # 5. Test movie details (wallpaper, cast, ratings, genres, about)
    print("\n5. Testing movie details (poster, cast, rating):")
    details = get_movie_details("Toy Story (1995)")
    print(f"   Title: {details['title']}")
    print(f"   Poster: {details['poster'][:50]}...")
    print(f"   Cast: {details['cast']}")
    print(f"   Rating: {details['rating']}")
    print(f"   Genres: {details['genres']}")
    assert details['title'], "Failed to get movie title!"

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
