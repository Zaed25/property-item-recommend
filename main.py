import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import warnings
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
import time

# This is the crucial line. It imports the class definitions so joblib can find them.
from preprocessing_classes import InitialCleaner, FeatureEngineer, LogAndCapTransformer, FrequencyEncoder

# Ignore common warnings for a cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
#  1. API SETUP AND GLOBAL ARTIFACTS
# =============================================================================

# Define paths to the artifact files
PIPELINE_PATH = 'preprocessor.pkl'
MODEL_PATH = 'encoder_model_exp4.keras'
SIMILARITY_LOOKUP_PATH = 'similarity_lookup.pkl'
ALL_PROPERTIES_PATH = 'all_properties.csv'
PROPERTIES_API_URL = "http://wonederway.premiumasp.net/api/Recommendation/ListProperties"
TOP_N_SIMILAR = 50

# Define the full set of columns the pipeline was trained on
REQUIRED_COLUMNS = [
    'price', 'bed', 'bath', 'acre_lot', 'house_size', 'city', 'state',
    'zip_code', 'status', 'brokered_by', 'prev_sold_date'
]

# Initialize the FastAPI app
app = FastAPI(
    title="Real Estate Recommender API",
    description="An API to generate property embeddings and find similar properties.",
    version="3.2.0" # Version updated for memory optimization
)

# A dictionary to hold our loaded artifacts
artifacts = {}

# =============================================================================
#  2. API KEY AUTHENTICATION SETUP
# =============================================================================

# Define the API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Get the secret API key from environment variables
API_KEY = os.getenv("RECOMMENDATION_API_KEY")

if API_KEY is None:
    print("⚠️ WARNING: RECOMMENDATION_API_KEY environment variable not set. API security is disabled.")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Dependency function to validate the API key"""
    if API_KEY and api_key_header == API_KEY:
        return api_key_header
    else:
        if API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key"
            )
        else:
            return None


# =============================================================================
#  3. HELPER FUNCTION TO ENSURE DATA SCHEMA
# =============================================================================

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the incoming DataFrame has all the columns required by the pipeline.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df

# =============================================================================
#  4. BATCH SIMILARITY PROCESSING LOGIC (MEMORY OPTIMIZED)
# =============================================================================

def run_similarity_update_task():
    """
    The complete batch processing logic, now optimized for low memory usage.
    """
    print("--- Starting background similarity update task... ---")

    try:
        # Step 1: Fetch latest property data
        print(f"-> Fetching property data from {PROPERTIES_API_URL}...")
        response = requests.get(PROPERTIES_API_URL, timeout=60)
        response.raise_for_status()
        properties_df = pd.DataFrame(response.json()['data'])
        properties_df = ensure_schema(properties_df)
        properties_df.to_csv(ALL_PROPERTIES_PATH, index=False)
        print(f"✅ Fetched and saved {len(properties_df)} new properties.")

        # Step 2: Generate embeddings for all properties
        print("-> Generating new embeddings...")
        processed_data = artifacts['pipeline'].transform(properties_df)
        all_embeddings = artifacts['model'].predict(processed_data)
        property_ids = properties_df['propertyID'].tolist()
        print("✅ Embeddings generated.")

        # ================================================================= #
        #  START: MEMORY OPTIMIZATION - AVOID FULL SIMILARITY MATRIX        #
        # ================================================================= #
        
        # Steps 3 & 4: Calculate similarities and build the lookup iteratively
        print("-> Calculating similarities iteratively to save memory...")
        new_similarity_lookup = {}
        for i, prop_id in enumerate(property_ids):
            # Reshape the current embedding to a 2D array for cosine_similarity
            current_embedding = all_embeddings[i].reshape(1, -1)
            
            # Calculate similarity of the current property against all others
            # This creates a small (1, N) array instead of a large (N, N) matrix
            similarity_scores = cosine_similarity(current_embedding, all_embeddings)[0]
            
            # Get indices of top similar items, sorted descending
            similar_indices = np.argsort(similarity_scores)[::-1]
            
            # Build the list of top N similar properties, excluding the property itself
            top_similar = []
            for idx in similar_indices:
                if property_ids[idx] == prop_id:
                    continue
                if len(top_similar) >= TOP_N_SIMILAR:
                    break
                # Convert numpy float32 to native Python float for JSON serialization
                score = float(similarity_scores[idx])
                top_similar.append((property_ids[idx], score))
            
            new_similarity_lookup[prop_id] = top_similar

            # Optional: Add a progress indicator for long tasks
            if (i + 1) % 100 == 0:
                print(f"   ... processed {i + 1}/{len(property_ids)} properties for similarity.")

        # ================================================================= #
        #  END: MEMORY OPTIMIZATION                                         #
        # ================================================================= #

        # Step 5: Overwrite the old artifact files with the new data
        joblib.dump(new_similarity_lookup, SIMILARITY_LOOKUP_PATH)
        print("✅ New similarity lookup file saved.")
        
        # Step 6. IMPORTANT: Update the artifacts loaded in memory for immediate use
        artifacts['similarity_lookup'] = new_similarity_lookup
        artifacts['all_properties'] = properties_df.set_index('propertyID')
        artifacts['all_embeddings'] = all_embeddings
        artifacts['property_ids'] = property_ids
        
        print("--- ✅ Background similarity update complete. ---")

    except Exception as e:
        print(f"❌ ERROR during batch update: {e}")

# =============================================================================
#  5. API STARTUP LOGIC
# =============================================================================

@app.on_event("startup")
def load_artifacts():
    print("--- Loading pre-built artifacts... ---")
    artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
    artifacts['model'] = tf.keras.models.load_model(MODEL_PATH)
    artifacts['similarity_lookup'] = joblib.load(SIMILARITY_LOOKUP_PATH)
    all_props_df = pd.read_csv(ALL_PROPERTIES_PATH)
    all_props_df = ensure_schema(all_props_df)
    artifacts['all_properties'] = all_props_df.set_index('propertyID')
    print("-> Pre-calculating all property embeddings...")
    artifacts['all_embeddings'] = artifacts['model'].predict(artifacts['pipeline'].transform(all_props_df))
    artifacts['property_ids'] = all_props_df['propertyID'].tolist()
    print("--- All artifacts loaded. Ready to serve. ---")

# =============================================================================
#  6. PYDANTIC MODELS FOR DATA VALIDATION
# =============================================================================

class PropertyFeatures(BaseModel):
    brokered_by: Optional[float]=None; status: Optional[str]=None; price: Optional[float]=None
    bed: Optional[float]=None; bath: Optional[float]=None; acre_lot: Optional[float]=None
    street: Optional[str]=None; city: Optional[str]=None; state: Optional[str]=None
    zip_code: Optional[str]=None; house_size: Optional[float]=None; prev_sold_date: Optional[str]=None

class EmbeddingResponse(BaseModel): message: str; embedding: List[float]
class SimilarProperty(BaseModel): details: Any; similarity_score: float
class SimilarityResponse(BaseModel): source_property: Any; similar_properties: List[SimilarProperty]

# =============================================================================
#  7. API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Status"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Real Estate Recommender API."}

@app.post("/generate-embedding", response_model=EmbeddingResponse, tags=["Utilities"], dependencies=[Depends(get_api_key)])
def generate_embedding(features: PropertyFeatures):
    try:
        input_df = pd.DataFrame([features.dict()])
        input_df = ensure_schema(input_df)
        processed_features = artifacts['pipeline'].transform(input_df)
        embedding = artifacts['model'].predict(processed_features)[0]
        return {"message": "Embedding generated successfully.", "embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-similar-properties/{property_id}", response_model=SimilarityResponse, tags=["Recommendation"], dependencies=[Depends(get_api_key)])
def get_similar_properties(property_id: int):
    try:
        source_details = artifacts['all_properties'].loc[property_id].to_dict()
        source_details['propertyID'] = property_id
        similar_items = artifacts['similarity_lookup'][property_id]
        similar_list = []
        for sim_id, score in similar_items:
            if sim_id in artifacts['all_properties'].index:
                details = artifacts['all_properties'].loc[sim_id].to_dict()
                details['propertyID'] = int(sim_id)
                similar_list.append({"details": details, "similarity_score": score})
        return {"source_property": source_details, "similar_properties": similar_list}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Property with ID {property_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-to-new-property", response_model=SimilarityResponse, tags=["Recommendation"], dependencies=[Depends(get_api_key)])
def find_similar_to_new_property(features: PropertyFeatures):
    try:
        input_df = pd.DataFrame([features.dict()])
        input_df = ensure_schema(input_df)
        processed_features = artifacts['pipeline'].transform(input_df)
        new_embedding = artifacts['model'].predict(processed_features)
        similarity_scores = cosine_similarity(new_embedding, artifacts['all_embeddings'])[0]
        similar_indices = np.argsort(similarity_scores)[::-1]
        top_similar = []
        for idx in similar_indices[:TOP_N_SIMILAR]:
            prop_id = artifacts['property_ids'][idx]
            score = similarity_scores[idx]
            details = artifacts['all_properties'].loc[prop_id].to_dict()
            details['propertyID'] = int(prop_id)
            top_similar.append({"details": details, "similarity_score": float(score)})
        return {"source_property": features.dict(), "similar_properties": top_similar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-similarities", status_code=202, tags=["Admin"], dependencies=[Depends(get_api_key)])
def trigger_similarity_update(background_tasks: BackgroundTasks):
    print("Received request to update similarity matrix.")
    background_tasks.add_task(run_similarity_update_task)
    return {"message": "Similarity update task has been started in the background. It may take several minutes to complete."}