import os
import joblib
import logging
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# Standard professional logging structure

logging.basicConfig( level=logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")

class CollaborativeEngine:
    def __init__(self, n_neighbors= 20):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.user_model = NearestNeighbors( metric= "cosine", algorithm= "brute", n_neighbors=n_neighbors)
        self.item_model = NearestNeighbors( metric= "cosine", algorithm= "brute", n_neighbors=n_neighbors)

    def process_and_train( self, ratings_path, output_dir):

        """Loads continuous user ratings, compresses coordinates, trains CF models, and saves assets."""

        logging.info("Reading processed historical user interactions...")

        # Optimize memory consumption right off the bat by downcasting default pandas precisions

        dtypes = {"userId": "int32", "movieId": "int32", "ratings": "float32"}
        ratings_df = pd.read_csv( ratings_path, dtype= dtypes)

        logging.info("Encoding non-consecutive tracking IDs into tight vector indexes...")
        ratings_df["user_index"] = self.user_encoder.fit_transform(ratings_df["userId"]) 
        ratings_df["movie_index"] = self.movie_encoder.fit_transform(ratings_df["movieId"])


        logging.info("Compiling sparse User-Item interaction CSR Matrix...")
        user_item_matrix = csr_matrix(
            (
                ratings_df["rating"],

                (
                ratings_df["user_index"], 
                ratings_df["movie_index"]
                )
            ),
            dtype='float32'
        )


        logging.info(f"Fitting User-Based KNN map across {user_item_matrix.shape[0]} profiles...")
        self.user_model.fit(user_item_matrix)

        logging.info(f"Transposing structures and fitting Item-Based KNN across {user_item_matrix.shape[1]} inventories...")
        movie_user_matrix = user_item_matrix.T.tocsr()
        self.item_model.fit(movie_user_matrix)


        # Ensure dynamic directories exist before serialization
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Serializing analytical models and encoder metrics to disk...")
        joblib.dump(self.user_encoder, os.path.join(output_dir, "user_encoder.pkl"))
        joblib.dump(self.movie_encoder, os.path.join(output_dir, "movie_encoder.pkl"))
        joblib.dump(self.user_model, os.path.join(output_dir, "user_cf.pkl"))
        joblib.dump(self.item_model, os.path.join(output_dir, "item_cf.pkl"))
        save_npz(os.path.join(output_dir, "user_item_matrix.npz"), user_item_matrix)

        logging.info("Collaborative Filtering Engine models completely trained and deployed!")


if __name__ == "__main__":

    # Dynamically locate project folder levels
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CLEAN_RATINGS = os.path.join(ROOT_DIR, "data", "processed", "ratings_clean.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "src", "models")
    
    engine = CollaborativeEngine()
    engine.process_and_train(CLEAN_RATINGS, MODELS_DIR)
