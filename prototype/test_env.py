#%%

import os

mongo_url = os.environ.get("MONGO_URL")
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_key = os.environ.get("QDRANT_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

print(mongo_url,'\n',qdrant_url,'\n',qdrant_key,'\n',openai_key)


from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
from qdrant_client.models import Distance,VectorParams


q_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key
)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

user_query = "Busco por um bar de música ao vivo"
query_vector = encoder.encode(user_query).tolist()

hits = q_client.search(
    collection_name="summary_db",
    query_vector=query_vector,
    limit=10,
)

print(hits)

#%%