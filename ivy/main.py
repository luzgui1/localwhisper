#%%

from pipeline.agents import Agents


agent = Agents(model_name="gpt-4", temperature=0.4)

my_input = "Gostaria de um bom restaurante para almoçar"

session_state = {
    "chat_history": [],
    "user_location": (-23.585537, -46.637549),
    "places_nearby": [],
    "debug_log": [],
    "historic_recommendation": 0,
}

response = agent.respond(my_input,session_state)
print(response)




#%%