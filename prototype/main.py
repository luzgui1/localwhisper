#%%

from pipeline.agents import Agents
from pipeline.tools import QdrantSearchTool, MongoSearchTool
import os
from dotenv import load_dotenv
import pprint

# Environment variables
mongo_url = os.environ.get("MONGO_URL")
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_key = os.environ.get("QDRANT_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

user_input = "busco por um bar de samba no centro de sao paulo"

agentic_result = {
    "intention":"",
    "detail":"",
    "response":""
}

result = {
    "candidates":""
    ,"memory":""
}

agent = Agents()
qdrant_tool = QdrantSearchTool(qdrant_url=qdrant_url, qdrant_key=qdrant_key)

mongo_tool = MongoSearchTool(mongo_url=mongo_url,database="api", collection="google_v0")

classification = agent.IntentionAgent(user_input, result)

agentic_result['intention'] = classification['intention']

if agentic_result['intention'] != "non_related_chat":
    
    # Call qdrant_tool
    candidates = qdrant_tool.__call__(user_input)
    candidate_names = [candidate['name'] for candidate in candidates['candidates']]
    
    # Initialize candidates list
    result['candidates'] = []
    
    # Search MongoDB for each candidate
    for name in candidate_names:
        mongo_result = mongo_tool.__call__({"name": name})

        candidate_info = {
            "name": mongo_result['candidates'][0]['name'],
            "place-description": mongo_result['candidates'][0]['summary'],
            "good-reviews": [review['text'] for review in mongo_result['candidates'][0]['reviews'] if review['rating'] > 4],
            "bad-reviews": [review['text'] for review in mongo_result['candidates'][0]['reviews'] if review['rating'] < 4],
            "place-address": mongo_result['candidates'][0]['formatted_address'],
            "website": mongo_result['candidates'][0]['website'],
            "price-level": mongo_result['candidates'][0]['price_level'],
            # "menu": mongo_result['candidates'][0]['menu']
            # "music": mongo_result['candidates'][0]['music']
        }
        
        result['candidates'].append(candidate_info)

    # Initialize DetailAgent
    detail = agent.DetailAgent(user_input,agentic_result)

    # Initialize ResponseAgent
    response = agent.ResponseAgent(user_input,agentic_result,result)

    pprint.pprint(agentic_result)


#%%