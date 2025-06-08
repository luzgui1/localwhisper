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

    def IntentionAgent(self, user_query: str, agents_dictionary: dict):

        """"
        This will be the first agent of the project.
        His role is to evaluate user's intention based on the request made.

        Params:
            - user_query: User's request in a string.
            = dictionary: The agents_dict that comes from main.py
        """

        self.update_model_settings(model_name="gpt-4",temperature=0.2)

        intention_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                    You are an agent in a chain of agents for a city leisure recommender system. 
                    Based on the user's message, classify the intent using one of the *following options:*

                    "generic_pub_request"
                    "detailed_place_request"
                    "non_related_chat"

                    *OPTION'S DESCRIPTION:*
                        "generic_pub_request": When user is not asking for a specific place in a specific location. Ex: "I'm looking for a place with cheap beer."
                        "detailed_place_request": When user is looking for a specific place or a place with very specific characteristics. Ex: "I'm looking for a place nearby metro Ana Rosa and with vegetarian food."
                        "non_related_chat": When user is not talking about anything related with urban leisure.

                    Respond **only** the label of intention and nothing else.
                    Use your memory to help you classify the intention:
                    {memory}
                """
                ),
                
                ("user", "{input}")
                ])
        
        chain = intention_prompt | self.llm
        
        response = chain.invoke({"input": user_query, "memory": agents_dictionary['memory']}).content
        intent = str(response).strip().strip('"').strip("'")
        agents_dictionary["intention"] = intent

        return agents_dictionary
    
    def DetailAgent(self, user_query: str, agents_dictionary:dict):

        """
        This agent will be the one detailing what the user is really asking about in it's request.

        Params:
            - user_query: User's request in a string.

        Output: a dictionary trigger. 
        """

        self.update_model_settings(model_name="gpt-4", temperature=0.5)

        detailing_prompt = ChatPromptTemplate.from_messages([
            (
                "system", """
                    You are an agent in chain of agents for city laisure recommender.
                    Your role is to bring more details about the user request.
                    I already understood that user is talking about urban leisure.
                    Please, classify it's intent in one or more of the *following options*:

                    'geographic_detail'
                    'reviews_detail'
                    'menu_detail'
                    'musical_detail'
                    
                    *OPTION'S DESCRIPTION:*
                        'geographic_detail': When user is asking for a place based on previous geographic requests. Ex.: "I'm looking for a place nearby Vila Mariana São Paulo"
                        'reviews_detail': When user needs to have more historic information about the place he's talking about. Ex.: "Is this place treating well the costumers? What does people have to say about it?"
                        'menu_detail': When user is asking more about the menu options. Ex.: "Does it have caipirinha in the menu?"
                        'musical_detail': When user is asking about the musical genre it will be playing in the place. Ex.: "Will this place have rock music playing?"

                    *DISCLAIMER:*
                        - You can bring more than one option if needed.
                        - *Only respond with those options, and nothing more.*
                    """
            ),
            (
                "user", "{input}"
            )
        ])

        chain = detailing_prompt | self.llm
        response = str(chain.invoke({"input": user_query}).content)
        
        details = [detail.strip().strip("'").strip('"') for detail in response.strip('[]').split(',')]
        agents_dictionary['detail'] = details
    

    def ResponseAgent(self, user_query: str, agents_dictionary:dict, result_dictionary: dict):

        """"
        This is my response agent.
        His role is to structure a response for the final user based on tools and other agents decisions.

        Parameters:
            - user_query: User's request
            - The final parameters of the user
        """

        self.update_model_settings(model_name='gpt-4',temperature=0.7)

        if agents_dictionary['intention'] != 'non_related_chat':

            response_data = {}
            filtered_dict = result_dictionary['candidates']

            for d in filtered_dict:
                if agents_dictionary['detail'] == 'geographic_detail':
                    response_data[d['name']] = {'summary':d['place-description'],'address':d['place-address'], 'website':d['website']}

                elif agents_dictionary['detail'] == 'reviews_detail':
                    response_data[d['name']] = {'summary':d['place-description'],'bad-reviews':d['bad-reviews'],"good-reviews":d['good-reviews'],'website':d['website']}
                
                elif agents_dictionary['detail'] == 'menu_detail':
                    response_data[d['name']] = {'summary':d['place-description'],'description':d['menu'],"good-reviews":d['good-reviews'],'website':d['website']}

                elif agents_dictionary['detail'] == 'music-detail':
                    response_data[d['name']] = {'summary':d['place-description'],'description':d['menu'],"good-reviews":d['good-reviews'],'website':d['website']}
                else:
                    response_data[d['name']] = {'summary':d['place-description'],'website':d['website']}


            response_prompt = ChatPromptTemplate.from_messages([
                (
                    "system", """
                        You are the final agent in a chain of agents for urban leisure.
                        If it's not your first user message of this session, *this is your memory*:

                        {memory}

                        And this is the final data prepared for you by other agents and tools:

                        {data}

                        Based on those information and the user_input, provide the user a very well structured message for it's request.
                        *RULES:*
                        - Always speak in the same language of the user.
                    """
                ),
                ("user", "{input}")
            ])

            chain = response_prompt | self.llm
            
            agents_dictionary["response"] = str(chain.invoke({
                "input": user_query,
                "memory": result_dictionary['memory'],
                "data": response_data
                }).content)
            return agents_dictionary["response"]
            
        else:
            
            response_prompt = ChatPromptTemplate.from_messages([
                (
                    "system", """
                        You are the final agent in a chain of agents for urban leisure.
                        User is not talking about urban leisure.
                        Please, provide a very well structured message for it's request.
                        Be persuasive and friendly, try to talk about the city and it's leisure options.
                        *RULES:*
                        - Always speak in the same language of the user.
                    """
                ),
                ("user", "{input}")
            ])

            chain = response_prompt | self.llm
            
            agents_dictionary["response"] = str(chain.invoke({
                "input": user_query
            }).content)
            return agents_dictionary["response"]


#%%




