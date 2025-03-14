#%%
from langchain_openai import OpenAI
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage

dotenv.load_dotenv()

model = OpenAI("gpt-4o-mini")

messages = [
    SystemMessage("Translate the following from English into Italian")
    ,HumanMessage("hi!")
]

response = model.invoke(messages)
print(response.content)

#%%