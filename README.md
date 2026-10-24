# 🎬 Movie Recommendation System

A simple, beginner-friendly Movie Recommendation System built using **Python**, **Pandas**, and **Streamlit**. It uses **Collaborative Filtering** (Cosine Similarity) on the MovieLens dataset to suggest movies.

---

## 📌 Features
- **Movie Recommender**: Select a movie from the dropdown to find similar movies based on audience ratings.
- **User Recommendations**: Select a User ID to see what movies they enjoyed and discover new recommendations.
- **Top Popular Movies**: View the highest-rated popular movies with an optional genre filter.

---

## 📁 Project Structure
```
movie-recommendation-system/
├── app.py                # Streamlit user interface
├── recommender.py        # Data loading and recommendation logic
├── test_recommender.py   # Test script to test functions
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── data/
    ├── movies.csv        # Movie details (movieId, title, genres)
    └── ratings.csv       # User ratings (userId, movieId, rating, timestamp)
```

---

## 🚀 How to Run the Project

### 1. Install Required Packages
```bash
pip install -r requirements.txt
```

### 2. Test the Code (Optional)
```bash
python test_recommender.py
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The web app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Dataset Used
- **MovieLens Latest Small Dataset** (GroupLens Research):
  - 100,836 ratings across 9,742 movies by 610 users.
