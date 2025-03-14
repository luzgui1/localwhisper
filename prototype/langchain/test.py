#%%
from langchain_openai import ChatOpenAI
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import os


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
)

messages = [
    SystemMessage("Translate the following from English into Italian")
    ,HumanMessage("hey bitch!")
]


model.invoke(messages)
response = model.invoke(messages)
print(response.content)

#%%