import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz

# Ensure python environment includes root directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 1. CORE ENGINE ORCHESTRATOR (Loads models and computes recommendations)
# =====================================================================
class LiveRecommendationOrchestrator:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "src", "models")
        
        # 1. Load the exact dataframes used for each engine
        self.movies_df = pd.read_csv(os.path.join(self.base_dir, "data", "processed", "merged_movies.csv"))
        
        # FIX: Load the indexed catalog that maps row-for-row perfectly with the FAISS index!
        self.semantic_df = pd.read_csv(os.path.join(self.models_dir, "indexed_catalog.csv"))

        # NEW: Load links.csv to connect MovieLens IDs to TMDb IDs
        # Handle tmdbId as Int64 to gracefully ignore empty/missing mappings
        links_df = pd.read_csv(os.path.join(self.base_dir, "data", "raw", "links.csv"), dtype={'movieId': 'int32', 'tmdbId': 'Int64'})

        # Merge tmdbId directly into your active catalogs
        self.movies_df = self.movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
        self.semantic_df = self.semantic_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
        
        # Create a title lookup map based on the semantic catalog space
        self.title_to_idx = dict(zip(self.semantic_df["title"].str.lower(), self.semantic_df.index))
        
        # 2. Load the trained Collaborative Filtering artifacts
        self.user_model = joblib.load(os.path.join(self.models_dir, "user_cf.pkl"))
        self.user_encoder = joblib.load(os.path.join(self.models_dir, "user_encoder.pkl"))
        self.movie_encoder = joblib.load(os.path.join(self.models_dir, "movie_encoder.pkl"))
        self.user_item_matrix = load_npz(os.path.join(self.models_dir, "user_item_matrix.npz")).tocsr()
        
        self.userid_to_idx = dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_))))
        self.movie_idx_to_id = self.movie_encoder.classes_
        
        # 3. Load the pre-computed high-speed FAISS Semantic vector search index
        self.faiss_index = faiss.read_index(os.path.join(self.models_dir, "movie.index"))

    def recommend_user_cf(self, user_id, top_k=10):
        """Vectorized User-Collaborative array lookups running in milliseconds."""
        if user_id not in self.userid_to_idx:
            return None
            
        user_idx = self.userid_to_idx[user_id]
        distances, indices = self.user_model.kneighbors(self.user_item_matrix[user_idx], n_neighbors=21)
        
        sim_indices = indices.flatten()[1:]
        similarities = 1 - distances.flatten()[1:]
        sim_sum = np.sum(similarities) if np.sum(similarities) > 0 else 1e-6
        
        neighborhood = self.user_item_matrix[sim_indices].toarray()
        predicted_vector = np.dot(similarities, neighborhood) / sim_sum
        
        # Mask out movies this user has already seen
        predicted_vector[self.user_item_matrix[user_idx].indices] = -1.0
        
        top_movie_indices = np.argsort(predicted_vector)[::-1][:top_k]
        rec_ids = self.movie_idx_to_id[top_movie_indices]
        
        results = self.movies_df[self.movies_df["movieId"].isin(rec_ids)]

        # NEW: Ensure tmdbId is passed back in the results dictionary
        return results[["movieId", "title", "tmdbId"]].to_dict(orient="records")

    def recommend_semantic(self, movie_title, top_k=10):
        """Zero-shot search to bypass cold starts when user profile is brand new."""
        title_clean = movie_title.strip().lower()
        
        # FIX: Look up the row index directly from the matching title map
        movie_idx = self.title_to_idx.get(title_clean)
        if movie_idx is None:
            return None
            
        query_vector = np.zeros((1, self.faiss_index.d), dtype='float32')
        self.faiss_index.reconstruct(int(movie_idx), query_vector[0])
        
        _, indices = self.faiss_index.search(query_vector, top_k + 1)
        
        # FIX: Pull the real movie IDs directly using the FAISS row indices from semantic_df
        target_indices = indices[0][1:]
        matched_movies = self.semantic_df.iloc[target_indices]
        
       # NEW: Ensure tmdbId is passed back in the results dictionary
       
        return matched_movies[["movieId", "title", "tmdbId"]].to_dict(orient="records")

# =====================================================================
# 2. MASTER ROUTING POLICY GATEWAY (Your pasted code logic)
# =====================================================================
class ProductionHybridEngine:
    def __init__(self):
        logging.info("Initializing Master Hybrid Recommendation Engine...")
        self.orchestrator = LiveRecommendationOrchestrator()

    def get_hybrid_recommendations(self, user_id=None, current_movie_title=None, top_k=10):
        """
        Master Routing Policy:
        1. Try Personalized User-CF based on historical behavior maps.
        2. If the user is new (Cold-Start), fall back to Deep Semantic FAISS Search.
        3. If no inputs are submitted, default to top trending catalog items.
        """
        # Strategy 1: Behavioral Personalization
        if user_id is not None:
            cf_recommendations = self.orchestrator.recommend_user_cf(user_id, top_k=top_k)
            if cf_recommendations:
                logging.info(f"Successful behavioral match routed for User ID: {user_id}")
                return {"routing_strategy": "collaborative_filtering", "data": cf_recommendations}

        # Strategy 2: Contextual Fallback (Resolves Cold-Starts)
        if current_movie_title is not None:
            semantic_recommendations = self.orchestrator.recommend_semantic(current_movie_title, top_k=top_k)
            if semantic_recommendations:
                logging.info(f"Cold-Start / fallback triggered. Serving semantic context match for: '{current_movie_title}'")
                return {"routing_strategy": "semantic_content_fallback", "data": semantic_recommendations}

        # Strategy 3: Global Safe Default (Trending)
        logging.warning("Insufficient profile parameters submitted. Routing to global trending default baseline.")
        default_recs = self.orchestrator.movies_df.head(top_k)[["movieId", "title"]].to_dict(orient="records")
        return {"routing_strategy": "global_popularity_default", "data": default_recs}


if __name__ == "__main__":
    # Quick execution test harness
    hybrid_system = ProductionHybridEngine()
    
    print("\n--- Route Test 1: Active Profile (User #1) ---")
    print(hybrid_system.get_hybrid_recommendations(user_id=1))
    
    print("\n--- Route Test 2: Cold-Start Profile (New User watching 'Inception') ---")
    print(hybrid_system.get_hybrid_recommendations(current_movie_title="Inception"))