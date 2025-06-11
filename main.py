import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import warnings
import os
import time
import kagglehub # <-- Import the new library
import shutil

# Ignore common warnings for a cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
#  1. CUSTOM TRANSFORMER CLASS DEFINITIONS
# =============================================================================

class InitialCleaner(BaseEstimator, TransformerMixin):
    """Performs initial data cleaning by setting invalid values to NaN."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        for col in ['bed', 'bath']:
            if col in X_copy.columns:
                mask = (X_copy[col] < 0) | (X_copy[col] > 20)
                X_copy.loc[mask, col] = np.nan
        if 'price' in X_copy.columns: X_copy.loc[X_copy['price'] < 1000, 'price'] = np.nan
        if 'house_size' in X_copy.columns: X_copy.loc[(X_copy['house_size'] < 100) | (X_copy['house_size'] > 15000), 'house_size'] = np.nan
        if 'acre_lot' in X_copy.columns: X_copy.loc[(X_copy['acre_lot'] < 0.01) | (X_copy['acre_lot'] > 1000), 'acre_lot'] = np.nan
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features and drops columns that are no longer needed."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        if 'prev_sold_date' in X_copy.columns:
            X_copy['has_prev_sale'] = X_copy['prev_sold_date'].notna()
        else:
            X_copy['has_prev_sale'] = False
        price_bins = [0, 100_000, 300_000, 600_000, 1_000_000, float('inf')]
        price_labels = ['<100k', '100k-300k', '300k-600k', '600k-1M', '1M+']
        X_copy['price_range'] = pd.cut(X_copy['price'], bins=price_bins, labels=price_labels)
        return X_copy.drop(columns=['prev_sold_date', 'street'], errors='ignore')

class LogAndCapTransformer(BaseEstimator, TransformerMixin):
    """Applies a log transform and then caps outliers using the IQR method."""
    def __init__(self):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
    def fit(self, X, y=None):
        X_log = np.log1p(pd.DataFrame(X))
        for i, col in enumerate(X_log.columns):
            Q1 = X_log[col].quantile(0.25)
            Q3 = X_log[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[i] = Q1 - 1.5 * IQR
            self.upper_bounds_[i] = Q3 + 1.5 * IQR
        return self
    def transform(self, X):
        X_log = np.log1p(pd.DataFrame(X))
        for i, col in enumerate(X_log.columns):
            X_log[col] = X_log[col].clip(self.lower_bounds_[i], self.upper_bounds_[i])
        return X_log.values

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features with their frequencies."""
    def __init__(self): self.freq_map_ = {}
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for i, col in enumerate(X_df.columns):
            self.freq_map_[i] = X_df[col].value_counts(normalize=True).to_dict()
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_copy = X_df.copy()
        for i, col in enumerate(X_df.columns):
            X_copy[f'{col}_encoded'] = X_copy[col].map(self.freq_map_[i]).fillna(0)
        return X_copy[[f'{col}_encoded' for col in X_df.columns]].values

# =============================================================================
#  2. API SETUP AND GLOBAL ARTIFACTS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'encoder_model_exp4.keras')
RAW_DATA_FILENAME = 'realtor-data.csv' # The name of the file inside the Kaggle dataset
LOCAL_RAW_DATA_PATH = os.path.join(BASE_DIR, RAW_DATA_FILENAME)
SIMILARITY_LOOKUP_PATH = os.path.join(BASE_DIR, 'similarity_lookup.pkl')
ALL_PROPERTIES_PATH = os.path.join(BASE_DIR, 'all_properties.csv')

PROPERTIES_API_URL = "http://wonederway.premiumasp.net/api/Recommendation/ListProperties"
TOP_N_SIMILAR = 50

app = FastAPI(title="Real Estate Recommender API", version="5.0.0")
artifacts = {}

# =============================================================================
#  3. BATCH PROCESSING LOGIC
# =============================================================================

def run_similarity_update():
    """
    This function contains all the logic from batch_similarity.py.
    It fetches all properties, processes them, and saves the similarity lookup.
    """
    global artifacts
    print("--- Starting background similarity update task... ---")
    
    # Step 1: Fetch all properties
    print("-> Fetching all properties...")
    try:
        response = requests.get(PROPERTIES_API_URL, timeout=60)
        response.raise_for_status()
        properties_df = pd.DataFrame(response.json()['data'])
        properties_df.to_csv(ALL_PROPERTIES_PATH, index=False)
        print(f"✅ Fetched and saved {len(properties_df)} properties.")
    except Exception as e:
        print(f"❌ ERROR fetching properties: {e}")
        return

    # Step 2: Generate embeddings for all properties
    print("-> Generating embeddings for all properties...")
    try:
        processed_data = artifacts['pipeline'].transform(properties_df)
        all_embeddings = artifacts['model'].predict(processed_data)
        property_ids = properties_df['propertyID'].tolist()
        print("✅ Embeddings generated.")
    except Exception as e:
        print(f"❌ ERROR generating embeddings: {e}")
        return

    # Step 3: Calculate cosine similarity
    print("-> Calculating similarity matrix...")
    similarity_matrix = cosine_similarity(all_embeddings)
    
    # Step 4: Build and save the lookup dictionary
    similarity_lookup = {}
    for i, prop_id in enumerate(property_ids):
        similar_indices = similarity_matrix[i].argsort()[::-1]
        top_similar = []
        for similar_idx in similar_indices:
            if similar_idx != i:
                similar_prop_id = property_ids[similar_idx]
                score = similarity_matrix[i][similar_idx]
                top_similar.append((similar_prop_id, score))
            if len(top_similar) >= TOP_N_SIMILAR:
                break
        similarity_lookup[prop_id] = top_similar
        
    joblib.dump(similarity_lookup, SIMILARITY_LOOKUP_PATH)
    print("✅ Similarity lookup file created.")
    
    # Step 5: Update the live artifacts in the running API
    artifacts['similarity_lookup'] = similarity_lookup
    artifacts['all_properties'] = properties_df.set_index('propertyID')
    print("--- Background similarity update task complete. ---")

# =============================================================================
#  4. API STARTUP LOGIC
# =============================================================================

@app.on_event("startup")
def startup_event():
    """
    Builds the pipeline, loads the model, and then either loads the 
    pre-calculated similarities or runs the batch job to create them.
    """
    print("--- Server starting up... ---")

    # Part 1: Download training data from Kaggle if it doesn't exist
    if not os.path.exists(LOCAL_RAW_DATA_PATH):
        print(f"⚠️ Raw data file not found at '{LOCAL_RAW_DATA_PATH}'.")
        print("-> Attempting to download from Kaggle...")
        try:
            # This requires KAGGLE_USERNAME and KAGGLE_KEY environment variables to be set
            path = kagglehub.dataset_download("ahmedshahriarsakib/usa-real-estate-dataset")
            # The actual CSV file path after extraction
            kaggle_csv_path = os.path.join(path, RAW_DATA_FILENAME)
            # Copy the file to our local directory for consistent access
            shutil.copy(kaggle_csv_path, LOCAL_RAW_DATA_PATH)
            print("✅ Successfully downloaded and saved raw data from Kaggle.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download data from Kaggle. Please check Kaggle credentials. Error: {e}")
    
    # Part 2: Build and fit the preprocessing pipeline
    print("-> Building preprocessing pipeline...")
    df_raw = pd.read_csv(LOCAL_RAW_DATA_PATH, low_memory=False)
    
    cols_log_cap = ['price', 'bed', 'bath', 'house_size']
    cols_boxcox = ['acre_lot']
    cols_freq = ['brokered_by', 'city', 'zip_code']
    cols_onehot = ['status', 'price_range']
    cols_label = ['state']
    
    numeric_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('log_and_cap', LogAndCapTransformer()), ('scaler', StandardScaler())])
    acre_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('boxcox', PowerTransformer(method='box-cox')), ('scaler', StandardScaler())])
    freq_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('frequency_encoder', FrequencyEncoder()), ('scaler', StandardScaler())])
    label_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), ('scaler', StandardScaler())])
    onehot_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_pipe, cols_log_cap), ('acre_lot', acre_pipe, cols_boxcox),
        ('frequency', freq_pipe, cols_freq), ('label', label_pipe, cols_label),
        ('onehot', onehot_pipe, cols_onehot)], remainder='passthrough')

    master_pipeline = Pipeline(steps=[('cleaner', InitialCleaner()), ('engineer', FeatureEngineer()), ('preprocessor', preprocessor)])
    
    X_train = df_raw.copy().dropna(subset=['price', 'state', 'zip_code'])
    master_pipeline.fit(X_train)
    artifacts['pipeline'] = master_pipeline
    print("✅ Pipeline built and fitted.")

    # Part 3: Load the ML model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Could not find Keras model at '{MODEL_PATH}'.")
    artifacts['model'] = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Keras model loaded.")

    # Part 4: Load or create the similarity data
    if os.path.exists(SIMILARITY_LOOKUP_PATH) and os.path.exists(ALL_PROPERTIES_PATH):
        print("-> Found existing similarity data. Loading into memory...")
        artifacts['similarity_lookup'] = joblib.load(SIMILARITY_LOOKUP_PATH)
        artifacts['all_properties'] = pd.read_csv(ALL_PROPERTIES_PATH).set_index('propertyID')
        print("✅ Recommendation artifacts loaded.")
    else:
        print("⚠️ Similarity data not found. Running initial batch job to create it...")
        run_similarity_update()
    
    print("--- Server startup complete. ---")


# =============================================================================
#  5. PYDANTIC MODELS & API ENDPOINTS
# =============================================================================

class PropertyFeatures(BaseModel):
    brokered_by: Optional[float]=None; status: Optional[str]=None; price: Optional[float]=None
    bed: Optional[float]=None; bath: Optional[float]=None; acre_lot: Optional[float]=None
    street: Optional[str]=None; city: Optional[str]=None; state: Optional[str]=None
    zip_code: Optional[str]=None; house_size: Optional[float]=None; prev_sold_date: Optional[str]=None

class EmbeddingResponse(BaseModel):
    message: str; embedding: List[float]

class SimilarProperty(BaseModel):
    details: Any; similarity_score: float

class SimilarityResponse(BaseModel):
    source_property: Any; similar_properties: List[SimilarProperty]

@app.post("/generate-embedding", response_model=EmbeddingResponse, tags=["On-Demand Embedding"])
def generate_embedding(property_features: PropertyFeatures):
    if 'pipeline' not in artifacts or 'model' not in artifacts:
        raise HTTPException(status_code=503, detail="Core artifacts not loaded.")
    try:
        input_df = pd.DataFrame([property_features.dict()])
        processed_features = artifacts['pipeline'].transform(input_df)
        embedding = artifacts['model'].predict(processed_features)[0]
        return {"message": "Embedding generated successfully.", "embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-similar-properties/{property_id}", response_model=SimilarityResponse, tags=["Recommendation"])
def get_similar_properties(property_id: int):
    if 'similarity_lookup' not in artifacts or 'all_properties' not in artifacts:
        raise HTTPException(status_code=503, detail="Recommendation data not ready.")
    try:
        if property_id not in artifacts['similarity_lookup']:
            raise HTTPException(status_code=404, detail=f"Property ID {property_id} not found.")
        source_details = artifacts['all_properties'].loc[property_id].to_dict()
        similar_items = artifacts['similarity_lookup'][property_id]
        similar_list = [{"details": artifacts['all_properties'].loc[sim_id].to_dict(), "similarity_score": score} for sim_id, score in similar_items if sim_id in artifacts['all_properties'].index]
        return {"source_property": source_details, "similar_properties": similar_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-similarity-matrix", status_code=202, tags=["Admin"])
def update_similarity_matrix(background_tasks: BackgroundTasks):
    """
    Triggers a background task to refresh the property data and recalculate similarities.
    """
    print("Received request to update similarity matrix.")
    background_tasks.add_task(run_similarity_update)
    return {"message": "Similarity update task has been started in the background."}
