# Loads and chunks HR policy document
# Generates embeddings and saves them with IDs
# Uploads to Vertex AI Vector Search and creates the index

import os
import uuid
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import storage, aiplatform

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
GCS_BUCKET = os.getenv("GCS_BUCKET")
DISPLAY_NAME = "hr-policy-index"

def load_and_chunk():
    loader = TextLoader("data/hr_policy.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

def generate_embeddings(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    records = []
    chunk_store = {}

    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        embedding = embeddings_model.embed_query(chunk.page_content)
        records.append({
            "id": chunk_id,
            "embedding": embedding
        })
        chunk_store[chunk_id] = chunk.page_content

    return records, chunk_store

def save_chunk_store(chunk_store):
    with open("data/chunk_store.json", "w") as f:
        json.dump(chunk_store, f)
    print("Chunk store saved to data/chunk_store.json")

def upload_to_gcs(records):
    # Save embeddings as JSONL
    jsonl_path = "data/embeddings.json"
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Upload to GCS
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob("hr-policy/embeddings.json")
    blob.upload_from_filename(jsonl_path)
    print(f"Embeddings uploaded to gs://{GCS_BUCKET}/hr-policy/embeddings.json")

def create_index():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=DISPLAY_NAME,
        contents_delta_uri=f"gs://{GCS_BUCKET}/hr-policy/",
        dimensions=384,                  # all-MiniLM-L6-v2 output size
        approximate_neighbors_count=10,
        distance_measure_type="COSINE_DISTANCE",
    )
    print(f"Index created: {index.resource_name}")
    return index

def create_index_endpoint(index):
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=f"{DISPLAY_NAME}-endpoint",
        public_endpoint_enabled=True,
    )
    print(f"Index Endpoint created: {endpoint.resource_name}")

    deployed = endpoint.deploy_index(
        index=index,
        deployed_index_id="hr_policy_deployed",
        display_name="hr-policy-deployed",
        min_replica_count=1,
        max_replica_count=2,
    )
    print(f"Index deployed to endpoint")
    print(f"INDEX_ENDPOINT_ID: {endpoint.name}")
    print(f"DEPLOYED_INDEX_ID: hr_policy_deployed")
    return endpoint

if __name__ == "__main__":
    print("Step 1: Loading and chunking documents...")
    chunks = load_and_chunk()
    print(f"Total chunks: {len(chunks)}")

    print("Step 2: Generating embeddings...")
    records, chunk_store = generate_embeddings(chunks)

    print("Step 3: Saving chunk store...")
    save_chunk_store(chunk_store)

    print("Step 4: Uploading embeddings to GCS...")
    upload_to_gcs(records)

    print("Step 5: Creating Vertex AI Vector Search Index...")
    index = create_index()

    print("Step 6: Creating Index Endpoint and deploying...")
    endpoint = create_index_endpoint(index)

    print("\nDone! Copy these into your .env file:")
    print(f"INDEX_ENDPOINT_ID={endpoint.name}")
    print(f"DEPLOYED_INDEX_ID=hr_policy_deployed")