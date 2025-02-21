from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer



app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

client = MongoClient("mongodb://mongodb_admin:bc1724c9aba3adc3f38166fadba5af4d@147.79.83.71:27017/")
db = client['api']
collection = db['google_v0']


class QueryRequest(BaseModel):
    query_text: str
    top_n: int = 10

@app.get("/vector_search/")
async def vector_search(query_text: str, top_n: int = 5):
    """
    API to perform vector search and return recommendations.
    """
    query_vector = model.encode(query_text).reshape(1, -1)
    
    cursor = collection.find({}, {"vector": 1, "name": 1, "reviews": 1, "summary": 1, "types": 1})
    similarities = []

    for doc in cursor:
        stored_vector = np.array(doc.get("vector", []))
        if stored_vector.size > 0:
            score = cosine_similarity(query_vector, stored_vector.reshape(1, -1))[0][0]
            doc_copy = {key: doc[key] for key in ["name", "reviews", "summary", "types"] if key in doc}
            doc_copy["score"] = round(float(score), 2)
            similarities.append(doc_copy)

    sorted_results = sorted(similarities, key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_n]

