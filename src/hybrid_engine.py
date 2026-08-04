import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz
from huggingface_hub import hf_hub_download

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
        self.hf_repo_id = "Vaibhav28092004/movielens-artifacts"
        
        # 1. Load dataframes
        self.movies_df = pd.read_csv(os.path.join(self.base_dir, "data", "processed", "merged_movies.csv"))
        
        # Load indexed catalog via HF auto-download helper
        indexed_catalog_path = self._get_artifact_path("indexed_catalog.csv")
        self.semantic_df = pd.read_csv(indexed_catalog_path)

        # Load links.csv mapping
        links_df = pd.read_csv(os.path.join(self.base_dir, "data", "raw", "links.csv"), dtype={'movieId': 'int32', 'tmdbId': 'Int64'})

        # Merge tmdbId directly into active catalogs
        self.movies_df = self.movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
        self.semantic_df = self.semantic_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
        
        # Create title lookup map based on semantic catalog space
        self.title_to_idx = dict(zip(self.semantic_df["title"].str.lower(), self.semantic_df.index))
        
        # 2. Load Collaborative Filtering artifacts via HF helper
        user_cf_path = self._get_artifact_path("user_cf.pkl")
        user_encoder_path = self._get_artifact_path("user_encoder.pkl")
        movie_encoder_path = self._get_artifact_path("movie_encoder.pkl")
        matrix_path = self._get_artifact_path("user_item_matrix.npz")

        self.user_model = joblib.load(user_cf_path)
        self.user_encoder = joblib.load(user_encoder_path)
        self.movie_encoder = joblib.load(movie_encoder_path)
        self.user_item_matrix = load_npz(matrix_path).tocsr()
        
        self.userid_to_idx = dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_))))
        self.movie_idx_to_id = self.movie_encoder.classes_
        
        # 3. Load FAISS Semantic vector index via HF helper
        faiss_path = self._get_artifact_path("movie.index")
        self.faiss_index = faiss.read_index(faiss_path)

    def _get_artifact_path(self, filename):
        """Returns local path if exists, otherwise downloads from HF Hub."""
        local_path = os.path.join(self.models_dir, filename)
        if not os.path.exists(local_path):
            logging.info(f"Artifact {filename} not found locally. Downloading from Hugging Face Hub: {self.hf_repo_id}...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            local_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=filename,
                repo_type="dataset"
            )
        return local_path

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
        
        # Mask out movies user has seen
        predicted_vector[self.user_item_matrix[user_idx].indices] = -1.0
        
        top_movie_indices = np.argsort(predicted_vector)[::-1][:top_k]
        rec_ids = self.movie_idx_to_id[top_movie_indices]
        
        results = self.movies_df[self.movies_df["movieId"].isin(rec_ids)]
        return results[["movieId", "title", "tmdbId"]].to_dict(orient="records")

    def recommend_semantic(self, movie_title, top_k=10):
        """Zero-shot search to bypass cold starts when user profile is brand new."""
        title_clean = movie_title.strip().lower()
        
        movie_idx = self.title_to_idx.get(title_clean)
        if movie_idx is None:
            return None
            
        query_vector = np.zeros((1, self.faiss_index.d), dtype='float32')
        self.faiss_index.reconstruct(int(movie_idx), query_vector[0])
        
        _, indices = self.faiss_index.search(query_vector, top_k + 1)
        
        target_indices = indices[0][1:]
        matched_movies = self.semantic_df.iloc[target_indices]
        
        return matched_movies[["movieId", "title", "tmdbId"]].to_dict(orient="records")

# =====================================================================
# 2. MASTER ROUTING POLICY GATEWAY
# =====================================================================
class ProductionHybridEngine:
    def __init__(self):
        logging.info("Initializing Master Hybrid Recommendation Engine...")
        self.orchestrator = LiveRecommendationOrchestrator()

    def get_hybrid_recommendations(self, user_id=None, current_movie_title=None, top_k=10):
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
        default_recs = self.orchestrator.movies_df.head(top_k)[["movieId", "title", "tmdbId"]].to_dict(orient="records")
        return {"routing_strategy": "global_popularity_default", "data": default_recs}


if __name__ == "__main__":
    hybrid_system = ProductionHybridEngine()
    
    print("\n--- Route Test 1: Active Profile (User #1) ---")
    print(hybrid_system.get_hybrid_recommendations(user_id=1))
    
    print("\n--- Route Test 2: Cold-Start Profile (New User watching 'Inception') ---")
    print(hybrid_system.get_hybrid_recommendations(current_movie_title="Inception"))