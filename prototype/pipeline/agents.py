#%%

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

class Agents():

    """
    A class to manage and interact with different LangChain agents.
    This class provides functionality to create, configure, and run various
    language model agents using the LangChain framework.
    """
    
    def __init__(self, model_name="gpt-4", temperature=0.7):
        """
        Initialize the Agents class.        
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )    
    
    def update_model_settings(self, model_name=None, temperature=None):
        """
        Update the model settings and reinitialize the language model.
        """
        if model_name:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
            
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )

    def IntentionAgent(self, user_query: str, dictionary: dict):

        """"
        This will be the first agent of the project.
        His role is to evaluate user's intention based on the request made.
        """

        self.update_model_settings(model_name="gpt-4",temperature=0.2)

        intention_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an agent in a chain of agents for a city leisure recommender system. "
                        "Based on the user's message, classify the intent using one of the following options: "
                        '"pub_request", "geographic_request", "non_related_chat", "restaurant_request", '
                        '"nightclub_request", "outside_activity_request", "coffee_request". Respond only the label of intention.'),
                ("user", "{input}")
                ])
        
        chain = intention_prompt | self.llm
        
        dictionary["intention"] = str(chain.invoke({"input": user_query}).content)

        return dictionary
    

    def ResponseAgent(self, user_query: str, params:dict):

        """"
        This is my response agent.
        His role is to structure a response for the final user based on tools and other agents decisions.

        Parameters:
            - user_query: User's request
            - The final parameters of the user
        """

        self.update_model_settings(model_name='gpt-4',temperature=0.7)

        response_prompt = ChatPromptTemplate.from_messages([
            (
                "system","You are the final agent among a chain of agent for city laisure recommender system."
                "You recieved a final result treated data based on everything you need to know about user's request."
                f"Here's everything the system used to better understand user's request:\n\n "
                f"USER'S INTENTION: {params['intention']}\n"
                f"MATCHES IN THE DATABASE: {params['candidates']}\n"
                f"DESCRIPTION OF THE DATABASE: {params['place-description']}\n"
                f"TOP REVIEWS OF THE DATABASE: {params['top-reviews']}\n"
                f"ADDRESS IN THE DATABASE: {params['place-address']}\n"
                f"MENU OPTIONS IF APPLIABLE: {params['menu']}\n"
                f"PLACE SUMMARY: {params['summary']}\n"
                f"WEBSITE: {params['website']} \n"
                f"PRICE LEVEL: {params['price-level']}"
                "Based on the above structure analyze if you can help the user with it's request.\n"
            ),
            ("user", "{input}")
        ])

        chain = response_prompt | self.llm
        
        return str(chain.invoke({"input": user_query}).content)

#%%




