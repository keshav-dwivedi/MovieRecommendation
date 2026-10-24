import streamlit as st
import pandas as pd
from recommender import (
    build_recommender,
    get_similar_movies,
    get_user_recommendations,
    get_popular_movies,
    get_movie_details,
)

# Page configuration
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Search for a movie to see similar recommendations with posters, cast, ratings, and genres.")


# Load datasets and precomputed matrix
@st.cache_data
def load_system():
    return build_recommender()


with st.spinner("Loading movie database..."):
    movies, ratings, pivot_table, similarity_matrix = load_system()


# Cached helper to fetch details so we don't repeat API calls
@st.cache_data(show_spinner=False)
def fetch_details(movie_title):
    return get_movie_details(movie_title)


# Modal dialog popup to show movie details when clicked
@st.dialog("🎬 Movie Details")
def show_movie_dialog(movie_title):
    info = fetch_details(movie_title)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(info["poster"], use_container_width=True)
    with col2:
        st.subheader(f"{info['title']} ({info['year']})")
        st.write(f"⭐ **Rating:** {info['rating']} / 10")
        st.write(f"🎭 **Genres:** {info['genres']}")
        st.write(f"👥 **Cast:** {info['cast']}")
        st.write(f"🎬 **Director:** {info['director']}")
        st.write(f"📖 **About:** {info['plot']}")


# Function to display movies in a clean grid with wallpaper posters
def display_movie_cards(movie_titles, key_prefix="rec"):
    """Displays movies in rows with poster images and a details button."""
    cols_per_row = 5
    for row_start in range(0, len(movie_titles), cols_per_row):
        row_movies = movie_titles[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, title in enumerate(row_movies):
            info = fetch_details(title)
            with cols[col_idx]:
                st.image(info["poster"], use_container_width=True)
                st.write(f"**{info['title']}**")
                st.caption(f"⭐ {info['rating']} | {info['year']}")
                if st.button("View Details", key=f"{key_prefix}_{row_start}_{col_idx}", use_container_width=True):
                    show_movie_dialog(title)


# Sidebar Menu
st.sidebar.title("📌 Menu")
menu_choice = st.sidebar.radio(
    "Select an Option:",
    ["1. Movie Recommender", "2. User Recommendations", "3. Top Popular Movies"]
)

top_n = st.sidebar.slider("Number of recommendations:", min_value=3, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Statistics")
st.sidebar.write(f"• **Total Movies:** {len(movies):,}")
st.sidebar.write(f"• **Total Ratings:** {len(ratings):,}")
st.sidebar.write(f"• **Total Users:** {ratings['userId'].nunique():,}")


# -----------------------------------------------------------------------------
# 1. Movie Recommender (Item-Based with Wallpapers)
# -----------------------------------------------------------------------------
if menu_choice == "1. Movie Recommender":
    st.header("🔍 Search Movie & Get Recommendations")

    movie_titles = sorted(pivot_table.index.tolist())
    default_idx = movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in movie_titles else 0
    selected_movie = st.selectbox("Search or select a movie:", movie_titles, index=default_idx)

    # Show preview of the selected searched movie
    st.markdown("### 📽️ Selected Movie")
    sel_info = fetch_details(selected_movie)
    preview_col1, preview_col2 = st.columns([1, 4])
    with preview_col1:
        st.image(sel_info["poster"], width=180)
    with preview_col2:
        st.subheader(f"{sel_info['title']} ({sel_info['year']})")
        st.write(f"⭐ **Rating:** {sel_info['rating']} / 10")
        st.write(f"🎭 **Genres:** {sel_info['genres']}")
        st.write(f"👥 **Cast:** {sel_info['cast']}")
        st.write(f"📖 **About:** {sel_info['plot']}")

    st.markdown("---")
    if st.button("✨ Get Similar Movie Recommendations", type="primary"):
        with st.spinner("Finding similar movies and fetching wallpapers..."):
            recs_df = get_similar_movies(selected_movie, movies, pivot_table, similarity_matrix, top_n=top_n)

            if not recs_df.empty:
                st.session_state["current_recs"] = recs_df["Title"].tolist()
            else:
                st.session_state["current_recs"] = []

    # Display recommended movies if generated
    if "current_recs" in st.session_state and st.session_state["current_recs"]:
        st.subheader(f"🍿 Movies Similar to '{selected_movie}'")
        display_movie_cards(st.session_state["current_recs"], key_prefix="similar")


# -----------------------------------------------------------------------------
# 2. User Recommendations
# -----------------------------------------------------------------------------
elif menu_choice == "2. User Recommendations":
    st.header("👤 Recommendations for User")
    st.write("Select a User ID to see movies they enjoyed and discover new recommendations.")

    user_ids = sorted(ratings["userId"].unique().tolist())
    selected_user = st.selectbox("Select User ID:", user_ids, index=0)

    # Display user's top rated movies
    user_ratings = ratings[ratings["userId"] == selected_user].merge(
        movies[["movieId", "title", "genres"]], on="movieId"
    ).sort_values(by="rating", ascending=False).head(5)

    st.subheader(f"Favorites of User {selected_user}")
    display_movie_cards(user_ratings["title"].tolist(), key_prefix="user_fav")

    st.markdown("---")
    if st.button("Generate Recommendations for User", type="primary"):
        with st.spinner("Finding recommendations for user..."):
            user_recs = get_user_recommendations(
                selected_user, movies, ratings, pivot_table, similarity_matrix, top_n=top_n
            )
            if not user_recs.empty:
                st.session_state["user_recs_list"] = user_recs["Title"].tolist()
            else:
                st.session_state["user_recs_list"] = []

    if "user_recs_list" in st.session_state and st.session_state["user_recs_list"]:
        st.subheader(f"🍿 Recommendations for User {selected_user}")
        display_movie_cards(st.session_state["user_recs_list"], key_prefix="user_rec")


# -----------------------------------------------------------------------------
# 3. Top Popular Movies
# -----------------------------------------------------------------------------
else:
    st.header("🔥 Top Rated & Popular Movies")
    st.write("Browse popular movies with posters across different genres.")

    genres_set = set()
    for g in movies["genres"].dropna():
        for item in g.split("|"):
            if item != "(no genres listed)":
                genres_set.add(item)
    genre_list = ["All"] + sorted(list(genres_set))

    selected_genre = st.selectbox("Filter by Genre:", genre_list)

    pop_df = get_popular_movies(movies, top_n=top_n, genre=selected_genre)
    display_movie_cards(pop_df["Title"].tolist(), key_prefix="popular")


# Footer
st.markdown("---")
st.caption("🎬 Movie Recommendation System | Simple Collaborative Filtering Project with Wallpapers & Cast Details")
