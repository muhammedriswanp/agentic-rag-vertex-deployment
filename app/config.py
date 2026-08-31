import os
from dotenv import load_dotenv

load_dotenv()

def validate_config():
    required_vars = [
        "PROJECT_ID",
        "LOCATION",
        "INDEX_ENDPOINT_ID",
        "DEPLOYED_INDEX_ID",
        "GROQ_API_KEY",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise EnvironmentError(
             f"Missing required environment variables: {', '.join(missing)}"
        )

config = {
    "project_id": os.getenv("PROJECT_ID"),
    "location": os.getenv("LOCATION"),
    "index_endpoint_id": os.getenv("INDEX_ENDPOINT_ID"),
    "deployed_index_id": os.getenv("DEPLOYED_INDEX_ID"),
    "groq_api_key": os.getenv("GROQ_API_KEY"),
}