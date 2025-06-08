#%%

from pipeline.agents import Agents
from pipeline.tools import QdrantSearchTool, MongoSearchTool
from langchain_core.messages import HumanMessage, AIMessage
import os

# Environment variables
mongo_url = os.environ.get("MONGO_URL")
qdrant_url = os.environ.get("QDRANT_CLUSTER_URL")
qdrant_key = os.environ.get("QDRANT_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

agent = Agents()
qdrant_tool = QdrantSearchTool(qdrant_url=qdrant_url, qdrant_key=qdrant_key)
mongo_tool = MongoSearchTool(mongo_url=mongo_url,database="api", collection="google_v0")

def run_pipeline(user_input:str, chat_history:list):
    
    agentic_result = {
        "intention":"",
        "detail":"",
        "response":""
    }
    
    result = {
        "candidates":[]
        ,"memory":""
    }


    classification = agent.IntentionAgent(user_input, agentic_result)
    agentic_result['intention'] = classification['intention']

    if agentic_result['intention'] != "non_related_chat":

        # Vetor -> nomes
        candidates = qdrant_tool.__call__(user_input)
        candidate_names = [candidate['name'] for candidate in candidates['candidates']]
        
        # Nomes -> Dados
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

        # Detalhes e resposta
        agent.DetailAgent(user_input,agentic_result)
        result["memory"] = "\n".join([f"{m.type.upper()}: {m.content}" for m in chat_history])
        response = agent.ResponseAgent(user_input,agentic_result,result)

    else:
        agentic_result["detail"] = []
        response = agent.ResponseAgent(user_input,agentic_result,result)
        result["memory"] = "\n".join([f"{m.type.upper()}: {m.content}" for m in chat_history])

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    return response, agentic_result, chat_history


#%%