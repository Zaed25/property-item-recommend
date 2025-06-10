import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import warnings
import os

# Import the custom classes from our new file
from preprocessing_classes import InitialCleaner, FeatureEngineer, LogAndCapTransformer, FrequencyEncoder

# Ignore common warnings for a cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
#  1. API SETUP AND ARTIFACT LOADING
# =============================================================================

# Define paths to the saved artifacts
PIPELINE_PATH = 'preprocessor.pkl'
MODEL_PATH = 'encoder_model_exp4.keras'
SIMILARITY_LOOKUP_PATH = 'similarity_lookup.pkl' # <-- New artifact
ALL_PROPERTIES_PATH = 'all_properties.csv'      # <-- New artifact

# Initialize the FastAPI app
app = FastAPI(
    title="Real Estate Recommender API",
    description="An API to generate property embeddings and find similar properties.",
    version="3.0.0"
)

# A dictionary to hold our loaded artifacts
artifacts = {}

@app.on_event("startup")
def load_artifacts():
    """
    This function runs when the API server starts.
    It loads all necessary artifacts into memory for fast access.
    """
    print("--- Loading artifacts at startup... ---")
    
    # Load model and pipeline for embedding generation
    if os.path.exists(PIPELINE_PATH) and os.path.exists(MODEL_PATH):
        artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
        artifacts['model'] = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Embedding generator artifacts loaded.")
    else:
        print("⚠️ Warning: Could not find 'preprocessor.pkl' or 'encoder_model_exp4.keras'. The /generate-embedding endpoint will not work.")

    # Load similarity lookup and property data for recommendations
    if os.path.exists(SIMILARITY_LOOKUP_PATH) and os.path.exists(ALL_PROPERTIES_PATH):
        artifacts['similarity_lookup'] = joblib.load(SIMILARITY_LOOKUP_PATH)
        # Set propertyID as the index for fast lookups with .loc
        artifacts['all_properties'] = pd.read_csv(ALL_PROPERTIES_PATH).set_index('propertyID')
        print("✅ Recommendation artifacts loaded.")
    else:
        print("⚠️ Warning: Could not find 'similarity_lookup.pkl' or 'all_properties.csv'. The /get-similar-properties endpoint will not work.")
    
    print("--- Server startup complete. ---")


# =============================================================================
#  2. PYDANTIC MODELS FOR DATA VALIDATION
# =============================================================================

class PropertyFeatures(BaseModel):
    """Defines the structure for a single property for embedding generation."""
    brokered_by: Optional[float] = Field(None, example=56507.0)
    status: Optional[str] = Field(None, example="sold")
    price: Optional[float] = Field(None, example=210000.0)
    bed: Optional[float] = Field(None, example=3.0)
    bath: Optional[float] = Field(None, example=1.0)
    acre_lot: Optional[float] = Field(None, example=0.18)
    street: Optional[str] = Field(None, example="123 Main St")
    city: Optional[str] = Field(None, example="Amherst")
    state: Optional[str] = Field(None, example="New York")
    zip_code: Optional[str] = Field(None, example="14226")
    house_size: Optional[float] = Field(None, example=1230.0)
    prev_sold_date: Optional[str] = Field(None, example="2020-01-15")

class EmbeddingResponse(BaseModel):
    """Defines the successful embedding response structure."""
    message: str
    embedding: List[float]

# NEW MODELS FOR SIMILARITY RESPONSE
class SimilarProperty(BaseModel):
    """Defines the structure for a single similar property."""
    details: Any # Using 'Any' to allow for a flexible dictionary of property details
    similarity_score: float

class SimilarityResponse(BaseModel):
    """Defines the successful similarity response structure."""
    source_property: Any
    similar_properties: List[SimilarProperty]


# =============================================================================
#  3. API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the Real Estate Recommender API."}


@app.post("/generate-embedding", response_model=EmbeddingResponse, tags=["Embedding"])
def generate_embedding(property_features: PropertyFeatures):
    """
    Receives raw property data and generates a 10-dimensional embedding.
    """
    if 'pipeline' not in artifacts or 'model' not in artifacts:
        raise HTTPException(status_code=503, detail="Embedding artifacts are not loaded.")

    try:
        input_df = pd.DataFrame([property_features.dict()])
        processed_features = artifacts['pipeline'].transform(input_df)
        if processed_features.shape[1] != 18:
            raise HTTPException(status_code=500, detail="Preprocessing pipeline did not return 18 features.")
        
        embedding = artifacts['model'].predict(processed_features)[0]
        return {"message": "Embedding generated successfully.", "embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-similar-properties/{property_id}", response_model=SimilarityResponse, tags=["Recommendation"])
def get_similar_properties(property_id: int):
    """
    Takes a propertyID and returns a list of the most similar properties
    based on the pre-calculated similarity lookup.
    """
    if 'similarity_lookup' not in artifacts or 'all_properties' not in artifacts:
        raise HTTPException(status_code=503, detail="Recommendation artifacts are not loaded.")

    try:
        # Check if the requested property ID exists
        if property_id not in artifacts['similarity_lookup']:
            raise HTTPException(status_code=404, detail=f"Property with ID {property_id} not found in similarity lookup.")

        # 1. Get the source property's details
        source_property_details = artifacts['all_properties'].loc[property_id].to_dict()

        # 2. Get the list of similar property IDs and scores from the lookup
        similar_items = artifacts['similarity_lookup'][property_id]
        
        # 3. Prepare the list of similar properties with their full details
        similar_properties_list = []
        for similar_id, score in similar_items:
            # Find the details for the similar property
            if similar_id in artifacts['all_properties'].index:
                details = artifacts['all_properties'].loc[similar_id].to_dict()
                similar_properties_list.append({
                    "details": details,
                    "similarity_score": score
                })

        return {
            "source_property": source_property_details,
            "similar_properties": similar_properties_list
        }
    
    except KeyError:
         raise HTTPException(status_code=404, detail=f"Property with ID {property_id} not found in the property details dataset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
