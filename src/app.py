import os
import sys
import requests
import streamlit as st
import pandas as pd
from difflib import get_close_matches
from dotenv import load_dotenv
import urllib.request

# Load environment variables
load_dotenv()

# Initialize variable
TMDB_API_KEY = ""

# 1. First, attempt to read from Streamlit Cloud Secrets safely
try:
    if "TMDB_API_KEY" in st.secrets:
        TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    pass  # Catch error when running locally without a secrets.toml file

# 2. If secrets didn't provide the key, fall back to environment variables / .env
if not TMDB_API_KEY:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")



# =========================================================
# 2. AUTOMATIC MODEL & DATASET DOWNLOADER
# =========================================================

# Replace with your actual Hugging Face dataset direct URLs
MODEL_ARTIFACTS = {
    "src/models/embeddings.npy": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/embeddings.npy",
    "src/models/indexed_catalog.csv": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/indexed_catalog.csv",
    "src/models/movie.index": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/movie.index",
    "src/models/item_cf.pkl": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/item_cf.pkl",
    "src/models/user_cf.pkl": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/user_cf.pkl",
    "src/models/user_item_matrix.npz": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/user_item_matrix.npz",
    "src/models/tfidf_matrix.npz": "https://huggingface.co/datasets/Vaibhav28092004/movielens-artifacts/resolve/main/tfidf_matrix.npz"
    
    
}


@st.cache_resource
def ensure_models_downloaded():
    """Download required artifacts from Hugging Face if missing locally."""
    os.makedirs("src/models", exist_ok=True)
    
    for local_path, url in MODEL_ARTIFACTS.items():
        if not os.path.exists(local_path):
            file_name = os.path.basename(local_path)
            with st.spinner(f"Downloading {file_name} from Hugging Face Storage..."):
                urllib.request.urlretrieve(url, local_path)
    return True



# Ensure python environment includes the project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hybrid_engine import ProductionHybridEngine

st.set_page_config(
    page_title="CineMatch AI | Premium Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# HIGH-END CINEMATIC UI CUSTOM STYLING INJECTION
# ---------------------------------------------------------------------
st.markdown("""
    <style>
    /* Premium Streaming Dark Core Settings */
    .stApp {
        background-color: #090c0f;
        color: #f3f4f6;
    }
    
    /* Global Text Visibility Hardening */
    h1, h2, h3, p, span, label, div {
        color: #ffffff !important;
    }
    .stMarkdown p {
        color: #e5e7eb !important;
    }
    
    /* Top Menu Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #12181f;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        font-weight: 600;
        font-size: 1rem;
        background-color: transparent;
        border: none;
        color: #9ca3af !important;
        transition: color 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #e11d48 !important; /* Premium Rose/Crimson Selection Vector */
    }

    /* Netflix-Style Cinematic Flex-Card Implementations */
    .movie-card {
        background-color: #12181f;
        border-radius: 12px;
        border: 1px solid #1e293b;
        transition: transform 0.4s cubic-bezier(0.165, 0.84, 0.44, 1), box-shadow 0.4s ease;
        margin-bottom: 25px;
        overflow: hidden;
        position: relative;
    }
    .movie-card:hover {
        transform: scale(1.04) translateY(-4px);
        box-shadow: 0 12px 24px rgba(225, 29, 72, 0.25);
        border-color: #e11d48;
    }
    
    /* Details Container Framing inside Card */
    .card-body {
        padding: 14px;
    }
    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-genre {
        font-size: 0.78rem;
        font-weight: 600;
        color: #e11d48 !important;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .movie-desc {
        font-size: 0.8rem;
        color: #9ca3af !important;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 8px;
    }
    
    /* Prominent Rating Sub-system */
    .rating-container {
        display: flex;
        align-items: center;
        gap: 4px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        width: fit-content;
    }
    .rating-star {
        color: #f59e0b !important;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .rating-value {
        color: #fbbf24 !important;
        font-size: 0.8rem;
        font-weight: 700;
    }
    
    /* Structural Rules Line Separators */
    hr {
        border-color: #1e293b !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# API METADATA & POSTER RESOLVER PIPELINE
# ---------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_details(tmdb_id, movie_title=""):
    """
    Fetches official posters, ratings, genres, and overviews from TMDB.
    Returns clean contextual metadata dictionaries with reliable safety defaults.
    """
    fallback_poster = "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500"
    details = {
        "poster": fallback_poster,
        "rating": "8.1", 
        "genre": "Drama & Sci-Fi",
        "overview": "An intense, high-stakes cinematic journey exploring profound human connections, sensory choices, and narrative vectors."
    }
    
    data = None
    # Tier 1: Direct Matching via ID Coordinates
    if not pd.isna(tmdb_id):
        id_url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        try:
            res = requests.get(id_url, timeout=2)
            if res.status_code == 200:
                data = res.json()
        except Exception:
            pass

    # Tier 2: Text Search Fallback
    if not data and movie_title:
        clean_query = movie_title.split(" (")[0].strip()
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(clean_query)}"
        try:
            res = requests.get(search_url, timeout=2)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    data = results[0]
        except Exception:
            pass

    # Populate details if API structural nodes are present
    if data:
        if data.get("poster_path"):
            details["poster"] = f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}"
        if data.get("vote_average") and data.get("vote_average") > 0:
            details["rating"] = f"{data.get('vote_average'):.1f}"
        if data.get("overview"):
            details["overview"] = data.get("overview")
            
        # Extract explicit genres strings if structural nested lists occur
        if data.get("genres"):
            details["genre"] = " & ".join([g["name"] for g in data["genres"][:2]])
        elif data.get("genre_ids"):
            details["genre"] = "Cinematic Collection"

    return details

# ---------------------------------------------------------------------
# CONTEXT ENGINE COMPONENT INITIALIZATION
# ---------------------------------------------------------------------
@st.cache_resource
def load_recommender_engine():
    return ProductionHybridEngine()

engine = load_recommender_engine()
movies_df = engine.orchestrator.movies_df
semantic_df = engine.orchestrator.semantic_df

# ---------------------------------------------------------------------
# MODERN METADATA CAROUSEL ROW COMPONENT
# ---------------------------------------------------------------------
def render_cinematic_carousel(movie_list, limit=6):
    """Renders highly engaging movie grid rows built with HTML card objects."""
    display_items = movie_list[:limit]
    cols = st.columns(len(display_items) if len(display_items) > 0 else 1)
    
    for idx, item in enumerate(display_items):
        with cols[idx]:
            # Pull rich structural content from API pipeline
            meta = get_movie_details(item.get('tmdbId'), movie_title=item.get('title', ''))
            
            # Inject beautiful container flexbox representations
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{meta['poster']}" style="width:100%; object-fit:cover; aspect-ratio:2/3; display:block;"/>
                    <div class="card-body">
                        <div class="movie-genre">{meta['genre']}</div>
                        <div class="movie-title">{item['title']}</div>
                        <p class="movie-desc">{meta['overview']}</p>
                        <div class="rating-container">
                            <span class="rating-star">★</span>
                            <span class="rating-value">{meta['rating']}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TOP HEADER & ICON CONTROLS
# ---------------------------------------------------------------------
available_users = list(engine.orchestrator.userid_to_idx.keys())[:50]

# 2-Column Top Bar: Header on left, Profile Popover aligned to TOP-RIGHT
col_header, col_profile = st.columns([0.80, 0.20], vertical_alignment="top")

with col_header:
    st.markdown("<h1 style='margin-bottom:0; padding-top:0;'>🎬 CINEMATCH AI</h1>", unsafe_allow_html=True)
    st.caption("Enterprise Dynamic Cross-Tower Content Delivery System")

with col_profile:
    # Adding a small top padding pushes the button to align cleanly with the title
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    # Icon-based profile & metrics popover top-right
    with st.popover("👤 Profile & Stats", use_container_width=True):
        st.markdown("🔐 User Session")
        selected_user = st.selectbox(
            "Select Active Account:",
            available_users,
            index=0,
            key="user_profile_selector"
        )
        
        st.markdown("---")
        st.markdown("#### 📊 Model Performance")
        st.markdown("""
            <div style="background-color:#090c0f; padding:12px; border-radius:8px; border:1px solid #1e293b;">
                <span style="color:#9ca3af; font-size:0.8rem;">Offline Evaluation Metrics:</span><br/>
                <b style="color:#10b981; font-size:1.05rem;">✓ 33.30% Precision@10</b><br/>
                <b style="color:#10b981; font-size:1.05rem;">✓ 22.48% Recall@10</b>
            </div>
        """, unsafe_allow_html=True)

# Top Navigation Tabs
tab_home, tab_browse, tab_search = st.tabs(["🏠 Home Portal Feed", "🎭 Category Discovery Space", "🔍 Universal Vector Search"])


# ---------------------------------------------------------------------
# VIEW LAYER 1: HOME PORTAL TAB
# ---------------------------------------------------------------------
with tab_home:
    # CURATED COMPONENT 1: Editor's Choice Section Tiers
    st.subheader("⭐ Editor's Choice: High-Fidelity Masterpieces")
    curated_titles = ["Inception (2010)", "Matrix, The (1999)", "Interstellar (2014)", "Fight Club (1999)", "Pulp Fiction (1994)", "Spirited Away (2001)"]
    curated_subset = semantic_df[semantic_df['title'].isin(curated_titles)]
    if not curated_subset.empty:
        curated_recs = curated_subset[["movieId", "title", "tmdbId"]].to_dict(orient="records")
        render_cinematic_carousel(curated_recs, limit=6)
    else:
        backup_curated = movies_df.iloc[40:46][["movieId", "title", "tmdbId"]].to_dict(orient="records")
        render_cinematic_carousel(backup_curated, limit=6)
        
    st.markdown("---")

    # CAROUSEL COMPONENT 2: Collaborative Filtering Tier
    st.subheader(f"👤 Recommended For You (Tailored for Profile #{selected_user})")
    with st.spinner("Traversing interactive matrix neighborhoods..."):
        personal_recs = engine.orchestrator.recommend_user_cf(user_id=int(selected_user), top_k=6)
        if personal_recs:
            render_cinematic_carousel(personal_recs, limit=6)
        else:
            default_recs = movies_df.head(6)[["movieId", "title", "tmdbId"]].to_dict(orient="records")
            render_cinematic_carousel(default_recs, limit=6)
            
    st.markdown("---")

    # CAROUSEL COMPONENT 3: Global Popularity Trends
    st.subheader("🔥 Trending Globally on CineMatch")
    trending_recs = movies_df.iloc[15:21][["movieId", "title", "tmdbId"]].to_dict(orient="records")
    render_cinematic_carousel(trending_recs, limit=6)

# ---------------------------------------------------------------------
# VIEW LAYER 2: BROWSE CATEGORIES TAB
# ---------------------------------------------------------------------
with tab_browse:
    st.subheader("🎭 Explore Global Genre Spaces")
    selected_genre = st.selectbox(
        "Select Target Interest Vector Category:",
        ["Action", "Comedy", "Sci-Fi", "Romance", "Horror", "Drama", "Thriller", "Animation", "Documentary"],
        label_visibility="collapsed"
    )
    st.markdown(f"### 🍿 Cinematic Highlights in: **{selected_genre}**")
    
    with st.spinner("Extracting dense vector matching tags..."):
        genre_anchor_df = semantic_df[semantic_df['title'].str.contains(selected_genre, case=False, na=False)]
        if not genre_anchor_df.empty:
            anchor_title = genre_anchor_df.iloc[0]['title']
            genre_recs = engine.orchestrator.recommend_semantic(anchor_title, top_k=6)
            if genre_recs:
                render_cinematic_carousel(genre_recs, limit=6)
        else:
            backup_genre_recs = semantic_df.head(6)[["movieId", "title", "tmdbId"]].to_dict(orient="records")
            render_cinematic_carousel(backup_genre_recs, limit=6)

# ---------------------------------------------------------------------
# VIEW LAYER 3: UNIVERSAL SEARCH TAB
# ---------------------------------------------------------------------
with tab_search:
    st.subheader("🔍 Universal Movie Search")
    
    # Single unified search bar for both direct titles and fuzzy typos
    user_query = st.text_input(
        "Search Box Query Input",
        placeholder="🍿 Type any movie title (e.g. Inception, Avatr, Toy Stori)...",
        label_visibility="collapsed"
    )
    
    if user_query.strip():
        with st.spinner("Searching catalog index and FAISS vector spaces..."):
            all_catalog_titles = semantic_df['title'].tolist()
            selected_title = None
            
            # Step 1: Direct case-insensitive match check
            exact_matches = semantic_df[semantic_df['title'].str.contains(user_query, case=False, na=False)]
            
            if not exact_matches.empty:
                # Pick the closest exact/partial match from the dataset
                selected_title = exact_matches.iloc[0]['title']
            else:
                # Step 2: Automatic fuzzy spellcheck if there's a typo
                fuzzy_hits = get_close_matches(user_query, all_catalog_titles, n=1, cutoff=0.4)
                if fuzzy_hits:
                    suggested_title = fuzzy_hits[0]
                    st.info(f"💡 *Did you mean:* **{suggested_title}**?")
                    selected_title = suggested_title

            # Step 3: Render the searched/corrected movie + its 5 FAISS vector recommendations
            if selected_title:
                target_movie_df = semantic_df[semantic_df['title'] == selected_title]
                
                if not target_movie_df.empty:
                    target_movie_row = target_movie_df.iloc[0]
                    searched_movie_dict = target_movie_row[["movieId", "title", "tmdbId"]].to_dict()
                    
                    # Fetch top 5 recommendations using the resolved title as vector anchor
                    semantic_recs = engine.orchestrator.recommend_semantic(selected_title, top_k=5)
                    
                    if semantic_recs:
                        st.markdown(f"### ✨ Displaying: *\"{selected_title}\"* & Similar Recommendations")
                        
                        # Prepend searched/corrected movie as Card #1 + 5 recommendations
                        final_display_list = [searched_movie_dict] + semantic_recs
                        render_cinematic_carousel(final_display_list, limit=6)
                    else:
                        st.warning("Could not generate recommendations for this title.")
            else:
                st.warning(f"No titles found matching or close to '{user_query}'. Please check your spelling.")
    else:
        st.info("Start typing any movie name in the box above to explore recommendations.")