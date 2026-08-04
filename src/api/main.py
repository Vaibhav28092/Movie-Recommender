import os
import sys
from fastapi import FastAPI, HTTPException, Query

# Ensure python environment includes root directories so your imports resolve perfectly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.hybrid_engine import ProductionHybridEngine

app = FastAPI(
    title="Hybrid Movie Recommendation Engine API",
    version="1.0.0",
    description="Scalable backend API serving pre-computed behavioral matrices and high-performance FAISS dense text vectors."
)

# Initialize the merged master engine once when the server boots up
hybrid_engine = ProductionHybridEngine()

@app.get("/api/v1/recommend")
def serve_hybrid_recommendations(
    user_id: int = Query(None, description="Target User ID profile to calculate collaborative filtering recommendations."),
    movie_title: str = Query(None, description="Baseline context movie title to execute semantic fallback routing.")
):
    # Prevent empty queries from breaking the backend
    if user_id is None and movie_title is None:
        raise HTTPException(
            status_code=400, 
            detail="Invalid Parameters. You must submit either a user_id or a movie_title query."
        )
        
    try:
        # Pass parameters directly to your master hybrid orchestrator routing class
        response = hybrid_engine.get_hybrid_recommendations(
            user_id=user_id, 
            current_movie_title=movie_title
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))