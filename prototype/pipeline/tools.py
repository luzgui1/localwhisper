from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

class QdrantSearchTool:
    def __init__(self,qdrant_url,qdrant_key, collection: str = 'summary_db'):
        """
        This tool is used to search for candidates in Qdrant.

        Params:
            - qdrant_url: docker url
            - qdrant_key: credentials
        """
        print('Initializing QdrantSearchTool')

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key
        )

        self.collection = collection
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, query: str, top_k: int = 5):
        """
        This is the calling function.
        It's building a vectorial query to search in the Qdrant and return the names in the collection.

        Params:
            - query: user_string
            - top_k: how many results do I want to track. 5 by default.
        """

        vector = self.encoder.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k
        )
        return {
            "candidates": [hit.payload for hit in results]
        }
    
class MongoSearchTool:

    def __init__(self,mongo_url, database: str, collection: str):

        """"
        This tool is used to search for candidates in MongoDB.

        Params:
            - mongo_url: credential
            - database: database name
            - collection: collection name.
        """

        print('Initializing MongoSearchTool')

        
        self.client = MongoClient(mongo_url)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def __call__(self, query: dict, limit: int = 5):
        """
        This calling function returns a query made in MongoDB

        Params:
            - query: mongo query. Ex: {"name": name}
            - limit: how many results do I want. 5 by default.
        """

        results = self.collection.find(query).limit(limit)
        return {
            "candidates": list(results)
        }