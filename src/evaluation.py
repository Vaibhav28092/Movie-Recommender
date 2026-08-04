import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hybrid_engine import ProductionHybridEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridEvaluator:
    def __init__(self):
        logging.info("Loading engine deployment for evaluation harness...")
        self.engine = ProductionHybridEngine()
        
    def evaluate_system(self, test_ratings_path, sample_users=200, k=10):
        logging.info("Running evaluation with temporary validation split...")
        dtypes = {'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}
        test_df = pd.read_csv(test_ratings_path, dtype=dtypes)
        
        # Focus on movies the user highly enjoyed
        relevant_df = test_df[test_df['rating'] >= 4.0]
        
        trained_users = self.engine.orchestrator.user_encoder.classes_
        
        # Filter for users who have engaged with enough movies to realistically split
        user_counts = relevant_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= 15].index
        eval_users = list(set(valid_users).intersection(set(trained_users)))
        
        if not eval_users:
            logging.error("No valid users with sufficient interactions found.")
            return
            
        np.random.seed(42)
        sample_size = min(sample_users, len(eval_users))
        selected_users = np.random.choice(eval_users, size=sample_size, replace=False)
        
        precisions = []
        recalls = []
        
        logging.info(f"Simulating lookups across {sample_size} profile subsets...")
        for user_id in tqdm(selected_users, desc="Running Evaluation"):
            user_movies = relevant_df[relevant_df['userId'] == int(user_id)]['movieId'].tolist()
            
            # Split this user's favorites: hide 30% to serve as holdout test targets
            split_idx = int(len(user_movies) * 0.7)
            train_movies = set(user_movies[:split_idx])
            test_movies = set(user_movies[split_idx:])
            
            if not test_movies or not train_movies:
                continue
                
            user_idx = self.engine.orchestrator.userid_to_idx[user_id]
            
            # Fetch raw user cluster neighborhood mapping
            distances, indices = self.engine.orchestrator.user_model.kneighbors(
                self.engine.orchestrator.user_item_matrix[user_idx], n_neighbors=21
            )
            
            sim_indices = indices.flatten()[1:]
            similarities = 1 - distances.flatten()[1:]
            sim_sum = np.sum(similarities) if np.sum(similarities) > 0 else 1e-6
            
            neighborhood = self.engine.orchestrator.user_item_matrix[sim_indices].toarray()
            predicted_vector = np.dot(similarities, neighborhood) / sim_sum
            
            # MASKING STRATEGY: Only filter out the training split movies.
            # Leave the test movies unmasked so the model can openly recommend them!
            train_movie_indices = []
            for m_id in train_movies:
                if m_id in self.engine.orchestrator.movie_encoder.classes_:
                    train_movie_indices.append(self.engine.orchestrator.movie_encoder.transform([m_id])[0])
            
            predicted_vector[train_movie_indices] = -1.0
            
            top_indices = np.argsort(predicted_vector)[::-1][:k]
            recommended_ids = self.engine.orchestrator.movie_idx_to_id[top_indices]
            
            # Intersect predictions with hidden favorites
            hits = len(set(recommended_ids).intersection(test_movies))
            
            precisions.append(hits / k)
            recalls.append(hits / len(test_movies))
            
        print("\n" + "="*50)
        print("          OFFLINE EVALUATION METRICS REPORT          ")
        print("="*50)
        print(f"Total Profiles Evaluated : {len(precisions)}")
        print(f"Mean Precision@{k}        : {np.mean(precisions)*100:.2f}%")
        print(f"Mean Recall@{k}           : {np.mean(recalls)*100:.2f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_ratings = os.path.join(base_dir, "data", "processed", "ratings_clean.csv")
    
    evaluator = HybridEvaluator()
    evaluator.evaluate_system(processed_ratings, sample_users=100, k=10)