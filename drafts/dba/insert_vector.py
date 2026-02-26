#Criação de uma collection no Qdrant com base nos dados do MongoDB

from pymongo import MongoClient
import pandas as pd
import json
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
from qdrant_client.models import Distance,VectorParams


mongo_url = os.environ.get("MONGO_URL")
qdrant_url = os.environ.get("QDRANT_CLUSTER_URL")
qdrant_key = os.environ.get("QDRANT_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

m_client = MongoClient(mongo_url)

qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_key,
)


encoder = SentenceTransformer('all-MiniLM-L6-v2')

db = m_client['api']
collection_m = db['google_v0']

mon_summ = {}
mon_rev = {}

for doc in collection_m.find():
    #vector dictionary for summary
    summary = doc.get('summary','')
    
    #encoder
    try:
        vector_s = encoder.encode(summary).tolist()
        mon_summ[doc.get('name','')] = vector_s
    except:
        vector_s = [0] * 384
        mon_summ[doc.get('name','')] = vector_s

    #dictionary for reviews
    reviews_text = (
    " ".join(review.get('text', '') for review in doc.get('reviews', []) if isinstance(doc.get('reviews', []), list))
    )
    try:
        vector_r = encoder.encode(reviews_text).tolist()
        mon_rev[doc.get('name','')] = vector_r
    except:
        vector_r = [0] * 384
        mon_rev[doc.get('name','')] = vector_r


summ_df = pd.DataFrame(list(mon_summ.items()), columns=['name', 'vector_summary'])
rev_df = pd.DataFrame(list(mon_rev.items()), columns=['name', 'vector_reviews'])

vect_df = summ_df.merge(rev_df, on='name', how='left')
vect_df.head()

qdrant_client.create_collection(
    collection_name = 'summary_db'
    ,vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

points_summ = [
    models.PointStruct(
        id=idx,
        vector=row['vector_summary'],
        payload={'name': row['name']}
    )
    for idx, row in vect_df.iterrows()
]

qdrant_client.upload_points(
    collection_name="summary_db",
    points=points_summ
)

user_query = "Busco por um bar de música ao vivo"
query_vector = encoder.encode(user_query).tolist()

hits = qdrant_client.search(
    collection_name="summary_db",
    query_vector=query_vector,
    limit=10,
)

print(hits)