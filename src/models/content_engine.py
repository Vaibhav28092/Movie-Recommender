import os
import logging
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize logger to match our data pipeline standards
logging.basicConfig( level=logging.INFO , format = " %(asctime)s - %(levelname)s - %(message)s ") 

class ContentSemanticEngine:
    def __init__(self, model_name= "all-MiniLM-L6-v2"):
        logging.info(f"Loading Sentence Transformer model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.movies_df = None

    def train_and_index(self, processed_data_path, models_dir, batch_size=128):

        """Loads processed data, generates semantic embeddings, and builds a FAISS index."""
        
        logging.info(f"Reading catalog data from: {processed_data_path}...")
        self.movies_df = pd.read_csv(processed_data_path) 

        # Double-check missing tags rows defensively

        original_count = len(self.movies_df)

        self.movies_df = self.movies_df.dropna(subset=["tags"]).copy()
        if len(self.movies_df) < original_count:
            logging.warning(f"Dropped {original_count - len(self.movies_df)} rows with missing tags.")

        logging.info(f"Generating semantic dense embeddings for {len(self.movies_df)} movies...")

        # Processing in batches prevents your computer from running out of RAM/GPU memory.
        embeddings = self.encoder.encode(
            self.movies_df['tags'].tolist(),
            batch_size = batch_size,
            show_progress_bar = True,
            convert_to_numpy = True
        ).astype("float32")

        dimension = embeddings.shape[1]
        logging.info(f"Building FAISS IndexFlatIP (Inner Product) with dimension: {dimension}...")

        # Normalize vectors inline so Inner Product calculations yield exact Cosine Similarity

        faiss.normalize_L2(embeddings)

        # Initialize and populate the index

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Ensure output model directory exists

        os.makedirs(models_dir, exist_ok=True)

        # Save structural index file using project relative pathing configurations

        index_save_path = os.path.join(models_dir, "movie.index")
        faiss.write_index(self.index, index_save_path)

        # Save a copy of our cleaned, synchronized dataframe for fast API lookups later

        catalog_save_path = os.path.join(models_dir, "indexed_catalog.csv")
        self.movies_df.to_csv(catalog_save_path, index=False)


        logging.info("Semantic Indexing Engine fully deployed and saved to disk!")


if __name__ == "__main__":
    # Dynamically resolve root project directory routes
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DATA = os.path.join(ROOT_DIR, "data", "processed", "merged_movies.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "src", "models")
    
    engine = ContentSemanticEngine()
    engine.train_and_index(PROCESSED_DATA, MODELS_DIR)