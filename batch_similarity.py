import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import time

# =============================================================================
#  CONFIGURATION
# =============================================================================

# The URL of the API that lists all properties from your database
PROPERTIES_API_URL = "http://wonederway.premiumasp.net/api/Recommendation/ListProperties"

# The URL of the FastAPI server you are running locally
EMBEDDING_API_URL = "http://127.0.0.1:8000/generate-embedding"

# Number of top similar properties to find for each item
TOP_N = 50

# Output filenames for the artifacts we will create
ALL_PROPERTIES_FILE = 'all_properties.csv'
ALL_EMBEDDINGS_FILE = 'all_embeddings.npy'
SIMILARITY_LOOKUP_FILE = 'similarity_lookup.pkl'

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def fetch_all_properties(api_url):
    """Fetches all property data from the database API."""
    print(f"-> Fetching all properties from {api_url}...")
    try:
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        if not data.get('success') or 'data' not in data:
            print("❌ ERROR: API response was not successful or is missing the 'data' key.")
            return None
        
        properties_df = pd.DataFrame(data['data'])
        print(f"✅ Successfully fetched and loaded {len(properties_df)} properties.")
        
        # Save the raw properties data for later use in the API
        properties_df.to_csv(ALL_PROPERTIES_FILE, index=False)
        print(f"-> Saved raw property data to '{ALL_PROPERTIES_FILE}'")
        
        return properties_df

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to the properties API: {e}")
        return None

def generate_embeddings_for_all(properties_df, embedding_api_url):
    """Generates an embedding for each property using our FastAPI server."""
    all_embeddings = []
    property_ids = []
    
    print(f"\n-> Generating embeddings for {len(properties_df)} properties...")
    
    for index, row in properties_df.iterrows():
        # The payload must match the Pydantic model in your API
        payload = {
            "price": row.get('price'),
            "bed": row.get('bed'),
            "bath": row.get('bath'),
            "acre_lot": row.get('acre_lot'),
            "city": row.get('city'),
            "zip_code": str(row.get('zip_code')),
            "house_size": row.get('house_size'),
            "state": row.get('state'), # Assumes 'state' exists, add if needed
            # The API's pipeline handles missing 'status', 'prev_sold_date', etc.
        }
        
        try:
            response = requests.post(embedding_api_url, json=payload, timeout=10)
            if response.status_code == 200:
                embedding = response.json().get('embedding')
                all_embeddings.append(embedding)
                property_ids.append(row['propertyID'])
                
                # Print progress
                if (index + 1) % 50 == 0:
                    print(f"   ...processed {index + 1}/{len(properties_df)} properties.")
            else:
                print(f"   ⚠️ Warning: Failed to get embedding for propertyID {row['propertyID']}. Status: {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR: Request to embedding API failed for propertyID {row['propertyID']}: {e}")
    
    if not all_embeddings:
        print("❌ ERROR: No embeddings were generated. Halting process.")
        return None

    embeddings_array = np.array(all_embeddings)
    np.save(ALL_EMBEDDINGS_FILE, embeddings_array)
    print(f"\n✅ Successfully generated {len(embeddings_array)} embeddings.")
    print(f"-> Saved embeddings to '{ALL_EMBEDDINGS_FILE}'")
    
    return embeddings_array, property_ids


def calculate_and_save_similarities(embeddings, prop_ids, top_n):
    """Calculates cosine similarity and saves the top N for each property."""
    print(f"\n-> Calculating cosine similarity for {len(embeddings)} embeddings...")
    start_time = time.time()
    
    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"   ...similarity matrix calculated in {time.time() - start_time:.2f} seconds.")
    
    similarity_lookup = {}
    
    for i, prop_id in enumerate(prop_ids):
        # Get similarity scores for the current property, excluding itself
        # argsort returns indices from smallest to largest, so we reverse it
        similar_indices = similarity_matrix[i].argsort()[::-1]
        
        # Find the top N similar items, making sure to skip the item itself (which has a score of 1.0)
        top_similar = []
        for similar_idx in similar_indices:
            if similar_idx != i: # Don't include the item itself
                similar_prop_id = prop_ids[similar_idx]
                score = similarity_matrix[i][similar_idx]
                top_similar.append((similar_prop_id, score))
            if len(top_similar) >= top_n:
                break
        
        similarity_lookup[prop_id] = top_similar
        
    # Save the final lookup dictionary
    joblib.dump(similarity_lookup, SIMILARITY_LOOKUP_FILE)
    print(f"\n✅ Similarity lookup created with {len(similarity_lookup)} entries.")
    print(f"-> Saved similarity lookup to '{SIMILARITY_LOOKUP_FILE}'")


# =============================================================================
#  MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    # Step 1: Get all property data
    all_properties_df = fetch_all_properties(PROPERTIES_API_URL)
    
    if all_properties_df is not None:
        # Step 2: Generate embeddings for all of them
        # Make sure your FastAPI server (main.py) is running before you execute this.
        embeddings, property_ids = generate_embeddings_for_all(all_properties_df, EMBEDDING_API_URL)
        
        if embeddings is not None:
            # Step 3: Calculate and save the similarity scores
            calculate_and_save_similarities(embeddings, property_ids, TOP_N)
            print("\n--- Batch similarity processing complete! ---")
