from google.cloud import aiplatform
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import config

def init_vertex():
    aiplatform.init(
        project=config['project_id'],
        location=config["location"],
    )

def get_embeddings(text: str) -> list:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings.embed_query(text)

def retrieve_chunks(query: str, top_k: int = 3) -> list[str]:
    init_vertex()

    # Connect to deployed index endpoint
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=config['index_endpoint_id']
    )

    # Embed the query
    query_embedding = get_embeddings(query)

    # Query Vector Search
    response = index_endpoint.find_neighbors(
        deployed_index_id=config["deployed_index_id"],
        queries=[query_embedding],
        num_neighbors=top_k,
        )

    # Extract matched IDs and return
    neighbors = response[0]
    chunks = [neighbor.id for neighbor in neighbors]

    return chunks
    