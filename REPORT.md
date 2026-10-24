# 🎬 Movie Recommendation System: Project Report & Technical Workflow

## Executive Summary
This project delivers a complete, AI-powered **Movie Recommendation System** developed with Python, NumPy, Pandas, and Streamlit. Utilizing the benchmark **MovieLens Latest Small** dataset (100,836 ratings across 9,742 movies and 610 users), the platform implements both **Item-Based** and **User-Based Collaborative Filtering** algorithms. The system addresses information overload by offering personalized recommendations, interactive exploration, and comprehensive data visualization.

---

## 1. Project Overview & Workflow Architecture
The end-to-end architecture follows a clean, modular design structured across five major operational stages:

```
[1. Data Ingestion]       --> MovieLens (movies.csv, ratings.csv)
       │
[2. Data Preprocessing]   --> Year extraction, genre parsing, Bayesian weighting
       │
[3. EDA & Visualizations] --> Rating distributions, genre shares, user activity
       │
[4. Recommendation Engine]--> Item-Based CF (Cosine Similarity)
                          --> User-Based CF (Pearson Correlation / Centered Cosine)
                          --> Interactive Real-Time Custom Profiling
       │
[5. Interactive UI]       --> Streamlit Dashboard (Cards, Sliders, Filters, Altair Charts)
```

---

## 2. Dataset Collection (Task 1)
- **Source**: GroupLens Research, University of Minnesota ([MovieLens Latest Small](https://grouplens.org/datasets/movielens/)).
- **Volume**:
  - Total Ratings: **100,836**
  - Unique Movies: **9,742**
  - Unique Users: **610**
  - Rating Scale: **0.5 to 5.0 stars** (in 0.5 increments)
  - Sparsity: **98.3%** ($\text{Sparsity} = 1 - \frac{|R|}{|U| \times |I|}$)

---

## 3. Data Cleaning & Preprocessing (Task 2)
The raw datasets were cleaned and enriched using Python:
1. **Title & Year Extraction**:
   Extracted four-digit release years (1902–2018) from movie titles via regular expressions (`r'\((\d{4})\)'`), isolating clean titles for presentation.
2. **Genre Normalization**:
   Split pipe-delimited genre strings (`Adventure|Animation|Comedy`) into Python lists, establishing 19 unique genres.
3. **Temporal Processing**:
   Converted UNIX epoch timestamps into standard UTC datetime objects and calendar years.
4. **Bayesian Weighted Rating (IMDB Formula)**:
   To prevent movies with a single 5.0-star rating from dominating rankings over established classics, we computed a Bayesian weighted rating ($WR$):
   $$WR = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
   Where:
   - $v$: Number of votes/ratings for the movie.
   - $m$: Minimum vote threshold (70th percentile of ratings count).
   - $R$: Average rating of the movie.
   - $C$: Mean rating across the entire dataset ($\approx 3.50$).

---

## 4. Exploratory Data Analysis & Trends (Tasks 3 & 6)
Key statistical patterns identified during exploratory data analysis:

| Analysis Dimension | Key Finding | Statistical Highlight |
|---|---|---|
| **Rating Distribution** | Positive skewness; users favor 3.0, 4.0, and 5.0 stars. | Mode = 4.0 ⭐ (26.8% of ratings), Mean = 3.50 ⭐ |
| **Catalog Distribution** | Drama and Comedy represent the vast majority of titles. | Drama: 4,361 titles; Comedy: 3,756 titles |
| **Most Reviewed Movies** | Universal blockbusters dominate review counts. | 1. *Forrest Gump* (329 ratings)<br>2. *The Shawshank Redemption* (317)<br>3. *Pulp Fiction* (307) |
| **Highest Quality Movies** | High-volume critical classics top the Bayesian score. | 1. *The Shawshank Redemption* (4.38/5.0)<br>2. *The Godfather* (4.21/5.0) |
| **User Participation** | Long-tailed activity distribution. | Median user rated 68 movies; Top user rated 2,698 movies |

---

## 5. Collaborative Filtering Algorithms (Task 4)

### A. Item-Based Collaborative Filtering (Item-Item Cosine Similarity)
Item-Based CF measures similarity between movies based on the vectors of ratings assigned by users who watched both.

**Mathematical Formulation**:
$$\text{sim}(i, j) = \frac{\vec{i} \cdot \vec{j}}{\|\vec{i}\| \|\vec{j}\|} = \frac{\sum_{u \in U} r_{u,i} r_{u,j}}{\sqrt{\sum_{u \in U} r_{u,i}^2} \sqrt{\sum_{u \in U} r_{u,j}^2}}$$

**Predicted Rating for User $u$ on Unseen Movie $i$**:
$$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{j \in \text{Rated}_u} \text{sim}(i, j) \cdot (r_{u,j} - \bar{r}_u)}{\sum_{j \in \text{Rated}_u} |\text{sim}(i, j)|}$$

**Advantages**:
- **Stability**: Movie relationships remain stable even as new users enter the system.
- **Explainability**: Allows generating transparent rationales (e.g., *"Recommended because you rated Inception 5.0 stars"*).
- **Inference Speed**: Precomputed $M \times M$ matrix delivers $O(1)$ lookup and fast predictions.

---

### B. User-Based Collaborative Filtering (User-User Pearson Correlation)
User-Based CF finds peer users with historical tastes strongly aligned with the target user.

**Mathematical Formulation**:
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{v}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{v}_v)^2}}$$

**Predicted Rating**:
$$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in K\text{-Nearest}} \text{sim}(u, v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in K\text{-Nearest}} |\text{sim}(u, v)|}$$

**Advantages**:
- Captures unexpected serendipity across genres.
- Naturally mirrors word-of-mouth human recommendations.

---

## 6. Recommendations & Demonstrations (Task 5)

### Scenario 1: Existing User #1 (Preferences: Action, Adventure, Crime)
- **Top Past Ratings**: *Star Wars: A New Hope (5.0)*, *Pulp Fiction (5.0)*, *The Matrix (5.0)*.
- **Item-Based CF Recommendations**:
  1. *Perks of Being a Wallflower, The (2012)* - Predicted: 4.49 ⭐
  2. *Maleficent (2014)* - Predicted: 4.47 ⭐
  3. *Mission: Impossible - Rogue Nation (2015)* - Predicted: 4.46 ⭐
  4. *Double Indemnity (1944)* - Predicted: 4.46 ⭐
- **User-Based CF Recommendations**:
  1. *Twelve Monkeys (1995)* - Predicted: 5.00 ⭐
  2. *Blade Runner (1982)* - Predicted: 5.00 ⭐
  3. *2001: A Space Odyssey (1968)* - Predicted: 5.00 ⭐

### Scenario 2: Interactive Real-Time Custom User
When an active visitor selects *The Matrix (1999) = 5.0 ⭐* and *Terminator 2: Judgment Day (1991) = 4.5 ⭐*:
- The system computes vector similarities on the fly.
- Suggested titles: *Fight Club*, *Twelve Monkeys*, *Alien*, and *The Matrix Reloaded*.

---

## 7. Model Evaluation & Quantitative Benchmarks
Using an 80/20 train/test split on held-out user interactions:

| Model Architecture | RMSE (Root Mean Squared Error) | MAE (Mean Absolute Error) | Relative Improvement |
|---|---|---|---|
| **Global Mean Baseline** | 1.0420 | 0.8310 | Reference Baseline |
| **User & Movie Bias Baseline** | 0.8822 | 0.6725 | +15.3% Error Reduction |
| **Collaborative Filtering (Item-Based)** | **0.8399** | **0.6376** | **+19.4% Error Reduction** |

*Note: Lower RMSE and MAE indicate superior prediction accuracy.*

---

## 8. Conclusion
The Movie Recommendation System fulfills all project specifications:
1. Collected and structured MovieLens benchmark data.
2. Cleaned and preprocessed titles, years, genres, and ratings.
3. Delivered rich exploratory data analyses across categories and user activities.
4. Implemented both Item-Based and User-Based Collaborative Filtering.
5. Provided multi-mode user recommendation interfaces with explainability.
6. Created dynamic visual trend analytics using Altair and Streamlit.
7. Documented the complete mathematical formulation and empirical evaluation in this report.
