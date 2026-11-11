# 🎬 CineMatch AI: Enterprise Hybrid Movie Recommendation Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange.svg)](https://huggingface.co/Vaibhav28092004/movielens-artifacts)
[![TMDb API](https://img.shields.io/badge/TMDb-v3%20API-01b4e4.svg)](https://developer.themoviedb.org/)

**CineMatch AI** is a production-ready, full-stack hybrid recommendation system deployed on Streamlit Cloud. It combines **User-Based Collaborative Filtering ($k$-NN)** with **Deep Semantic Vector Search (FAISS)**, resolving the classic machine learning **cold-start problem** while delivering sub-second personalized recommendations.

The frontend features a high-end, dark-mode cinematic interface with interactive modal cards, TMDb metadata (director, top cast, plot overview, ratings), and embedded YouTube trailers.

---

## 📌 Table of Contents
- [Architecture & Routing Gateway](#-architecture--routing-gateway)
- [Key Features](#-key-features)
- [Tech Stack & APIs](#-tech-stack--apis)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Project Directory Structure](#-project-directory-structure)
- [Model Artifact Storage & Orchestration](#-model-artifact-storage--orchestration)
- [Local Installation & Setup](#-local-installation--setup)
- [Environment Variables & Secrets](#-environment-variables--secrets)

---

## 📐 Architecture & Routing Gateway

1. **Behavioral Personalization (User-Based Collaborative Filtering):**
   * Uses vectorized $k$-NN lookups ($k=21$) over a sparse user-item interaction matrix (`CSR` format).
   * Predicts movie scores using similarity-weighted ratings and masks out previously watched titles.
2. **Cold-Start Fallback (FAISS Semantic Search):**
   * Triggered when interaction history is missing.
   * Performs real-time vector similarity search on high-dimensional semantic movie embeddings using **FAISS** (Facebook AI Similarity Search).
3. **Global Trending Default:**
   * Serves top-ranked, high-fidelity catalog masterpieces when zero profile or contextual inputs are provided.

---

## ✨ Key Features

* **🎭 Triple-Tab Experience:**
  * **🏠 Home Portal Feed:** User-personalized feeds, global trending rows, and editor's choice carousels.
  * **🎭 Category Discovery Space:** Dynamic vector filtering across genre categories (Action, Sci-Fi, Horror, Animation, etc.).
  * **🔍 Universal Vector Search:** Unified search bar with **fuzzy typo matching** (`difflib.get_close_matches`) and vector expansion.
* **🎬 Interactive Modal Pop-ups:**
  * Powered by Streamlit's `@st.dialog` component to prevent layout jumping.
  * Displays high-res posters, director names, top 5 cast members, runtime, release date, and plot summaries.
* **🍿 Embedded YouTube Trailers:**
  * Automatically resolves official trailer keys via TMDb and embeds responsive video players with `st.video()`.
* **⚡ High-Performance TTL Caching:**
  * Utilizes `@st.cache_data(ttl=86400)` and `@st.cache_resource` to cache ML models and API payloads, eliminating network round-trips.

---

## 🛠️ Tech Stack & APIs

* **Frontend & Web Framework:** [Streamlit](https://streamlit.io/)
* **Vector Indexing & Similarity Search:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
* **Machine Learning & Computations:** Scikit-learn, SciPy (Sparse Matrices), NumPy, Pandas
* **Model Storage & Artifact Orchestration:** [Hugging Face Hub (`huggingface_hub`)](https://huggingface.co/)
* **External APIs:** 
  * **TMDb API (The Movie Database):** Single-roundtrip metadata and trailer key resolver via `append_to_response=credits,videos`.

---

## 📊 Model Evaluation & Metrics

Offline evaluation performed on historical test subsets:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Precision@10** | **33.30%** | Fraction of recommended items in Top-10 that were relevant to the user profile |
| **Recall@10** | **22.48%** | Fraction of total relevant items captured within the Top-10 recommendations |
| **Query Latency** | **< 15ms** | Sub-second FAISS vector index retrieval and vectorized matrix operations |

---

## 📂 Project Directory Structure

```text
Movie-Recommender/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── links.csv                 # MovieLens ID to TMDb ID mapping table
│   └── processed/
│       └── merged_movies.csv         # Cleaned catalog metadata
│
├── src/
│   ├── app.py                        # Streamlit web application & UI layer
│   ├── hybrid_engine.py              # Master hybrid engine & routing orchestrator
│   └── models/
│       ├── indexed_catalog.csv       # Indexed semantic lookup map
│       ├── user_item_matrix.npz      # Sparse user-item matrix
│       ├── user_cf.pkl               # Trained Collaborative Filtering k-NN model
│       ├── user_encoder.pkl          # User label encoder
│       ├── movie_encoder.pkl         # Movie label encoder
│       └── movie.index               # High-speed FAISS vector index
The hybrid recommendation engine follows a three-tier routing strategy to serve recommendations dynamically:
