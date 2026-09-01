import os
import time
import uuid
import  json
import logging
from fastapi import FastAPI, Request
from app.retriever import retrieve_chunks
from app.config import validate_config
from app.generator import get_llm, evaluate_context, reformulate_query, generate_answer

class JSONLogger(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)

handler = logging.StreamHandler()
handler.setFormatter(JSONLogger)
logger = logging.getLogger("rag-app")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Validate env vars at startup
validate_config()

app = FastAPI()

chunk_store_path = "data/chunk_store.json"
if os.path.exists(chunk_store_path):
    with open(chunk_store_path, "r") as f:
        chunk_store = json.load(f)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    request_id = str(uuid.uuid4())
    body = await request.json()

    instances = body.get("instances", [])
    if not instances:
        return {"predictions": [{"error": "No instances provided"}]}

    query = instances[0].get("query", "")
    logger.info(f"request_id={request_id} query={query}")

    llm = get_llm()
    max_attempts = 3
    current_query = query
    all_context = ""
    total_retrievals = 0
    start = time.time()

    for attempt in range(1, max_attempts + 1):
        # Retrieve chunk IDs from Vector Search
        chunk_ids = retrieve_chunks(current_query, top_k=3)
        total_retrievals += 1

        # Look up actual text from chunk store
        context = "\n\n".join([
            chunk_store.get(cid, "") for cid in chunk_ids
        ])

        all_context = context if attempt == 1 else all_context + "\n\n" + context

        logger.info(f"request_id={request_id} attempt={attempt} chunks_retrieved={len(chunk_ids)}")

        is_sufficient = evaluate_context(query, all_context, llm)
        if is_sufficient:
            break
        if attempt < max_attempts:
            current_query = reformulate_query(query, all_context, llm)

    answer = generate_answer(query, all_context, llm)
    elapsed = round(time.time() - start, 2)

    logger.info(f"request_id={request_id} total_retrievals={total_retrievals} time_taken={elapsed}")

    return {
        "predictions": [
            {
                "answer": answer,
                "retrievals": total_retrievals,
                "time_taken": elapsed,
            }
        ]
    }