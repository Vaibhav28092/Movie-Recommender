import os
import ast
import pandas as pd
import numpy as np
import logging


# 1. Setup logging so your console tells you exactly what the script is doing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_clean_base_data(base_dir):

    """Translates Notebook 01: Dynamic loading and basic cleaning."""

    logging.info("Loading raw datasets dynamically...")

    # FIX: Use os.path.join instead of hardcoded C:\\ paths so it works anywhere

    raw_path = os.path.join(base_dir, "data", "raw")
    processed_path = os.path.join(base_dir, "data", "processed")

    ratings = pd.read_csv(os.path.join(raw_path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(raw_path, "movies.csv"))
    links = pd.read_csv(os.path.join(raw_path, "links.csv"))
    tmdb = pd.read_csv(os.path.join(raw_path, "tmdb.csv"))

    logging.info("Step-1: Processing timestamps and filling empty values in datasets...")
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    tmdb["director"] = tmdb["director"].fillna("")
    tmdb["overview"] = tmdb["overview"].fillna("")

    return ratings, movies, links, tmdb


def safe_literal_eval(val):

    """Defensive helper for feature engineering string-lists."""

    if pd.isna(val) or not isinstance(val, str):
        return []

    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def engineer_features_and_merge( ratings, movies, links, tmdb, base_dir):

    """Translates Notebook 02: Building the tags text signatures."""

    logging.info("Step-2: Structuring metadata string tags...")


    # 1. Safely convert string-lists to Python lists using our helper function
    genres_raw = tmdb['genres'].apply(safe_literal_eval)
    keywords_raw = tmdb['keywords'].apply(safe_literal_eval)
    cast_raw = tmdb['cast'].apply(safe_literal_eval)
    
    # 2. Process lists and collapse internal spaces natively
    genres = genres_raw.apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
    keywords = keywords_raw.apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
    cast = cast_raw.apply(lambda x: [i["name"].replace(" ", "") for i in x[:3] if "name" in i] if isinstance(x, list) else [])
    
    # 3. Defensive check: Ensure director is ALWAYS a list wrapper
    director = tmdb["director"].fillna("").apply(lambda x: [str(x).replace(" ", "")] if x else [])
    
    # 4. FIX: Force overview to string type before split to avoid float/NaN errors
    overview = tmdb["overview"].astype(str).fillna("").apply(lambda x: x.split())
    
    # 5. Combine lists into a single master tracking array safely
    combined_tags = overview + genres + keywords + cast + director
    
    # 6. Secure join execution (ensures x is a valid iterable object)
    logging.info("Compiling and string-joining uniform lowercase tags...")
    tmdb['tags'] = combined_tags.apply(lambda x: " ".join([str(i) for i in x]).lower() if isinstance(x, (list, np.ndarray)) else "")


    # Filter down features and merge catalogs

    new_tmdb = tmdb[[ "tmdb_id", "title", "tags"]].dropna(subset=["tags"])


    logging.info("Step-3: Merging catalog links together...")
    links = links.dropna(subset=["tmdbId"])
    movie_links = links.merge(movies, on="movieId", how="left")

    merged_movies = movie_links.merge(new_tmdb, left_on="tmdbId", right_on="tmdb_id", how="left")[["movieId","tmdb_id", "title_y", "tags"]].rename(columns={"title_y": "title"})


    # Save files to your data/processed/ directory

    processed_path = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    merged_movies.to_csv(os.path.join(processed_path, "merged_movies.csv"), index= False)
    ratings.to_csv(os.path.join(processed_path, "raings_clean.csv"), index= False)
    logging.info("Pipeline Successfull ! Output files saved to data/processed/" )


# This block ensures the code ONLY runs if you execute this file directly

if __name__ =="__main__":

    # Calculate the root folder path automatically relative to this file's position
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Run the execution sequence sequentially
    r, m, l, t = load_and_clean_base_data(ROOT_DIR)
    engineer_features_and_merge( r, m, l, t, ROOT_DIR)
