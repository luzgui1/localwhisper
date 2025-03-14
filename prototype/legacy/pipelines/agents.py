# pipeline.py

import streamlit as st
from qdrant_client import QdrantClient
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# ✅ Initialize global components
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Qdrant Database
q_client = QdrantClient(
    url="http://147.79.83.71:6333",
    api_key="KNF9zCKn15NOK9QOIWtgaYR4dpVPqHDq89f1qvzMSMCeY47Gr6"
)

# ✅ MongoDB Database
m_client = MongoClient("mongodb://mongodb_admin:bc1724c9aba3adc3f38166fadba5af4d@147.79.83.71:27017/")
data = m_client['api']
collection_m = data['google_v0']

# ✅ OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 🧠 MEMORY MANAGEMENT
# -----------------------------
class MemoryManager:
    def __init__(self):
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "stored_places" not in st.session_state:
            st.session_state.stored_places = {}  # ✅ Stores past recommended places

    def store_message(self, role, content):
        """Stores messages in session memory and Streamlit"""
        message = {"role": role, "content": content}
        st.session_state.conversation_history.append(message)
        st.session_state.messages.append(message)

    def get_last_messages(self, n=3):
        """Retrieves the last N messages"""
        return st.session_state.conversation_history[-n:]

    def reset_memory(self):
        """Resets conversation history and stored places"""
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.session_state.stored_places = {}

    def store_place_details(self, place_name, details):
        """Stores place details for later reference"""
        st.session_state.stored_places[place_name.lower()] = details

    def get_place_details(self, place_name):
        """Retrieves stored place details if available"""
        return st.session_state.stored_places.get(place_name.lower(), None)


# -----------------------------
# 🎯 UNDERSTANDING AGENT
# -----------------------------
class UnderstandingAgent:
    def __init__(self, user_query):
        self.user_query = user_query
        self.encoder = encoder

    def classify_intent(self):

        ### entender melhor como aplicar essa função
        ###########

        """Classifies the user intent into predefined categories"""
        categories = {
            "bares": ["bar", "cerveja", "drink", "pub", "barzinho", "musica ao vivo"],
            "restaurante": ["restaurante", "comida", "jantar", "almoço"]
        }
        for category, keywords in categories.items():
            if any(word in self.user_query.lower() for word in keywords):
                return category
        return "general"

    def check_context_continuation(self, memory_manager):
        """Checks if the new message is related to the previous topic"""
        last_messages = memory_manager.get_last_messages()

        if not last_messages:
            return "new_query"

        last_user_message = last_messages[-1]["content"]

        last_vector = self.encoder.encode(last_user_message)
        current_vector = self.encoder.encode(self.user_query)

        similarity = sum(last_vector * current_vector) / (sum(last_vector**2)**0.5 * sum(current_vector**2)**0.5)

        return "continuation" if similarity > 0.75 else "new_query"

    def extract_referenced_place(self, memory_manager):
        """Tries to extract a place name from past messages safely."""
        last_messages = memory_manager.get_last_messages(5)

        possible_places = []
        for message in reversed(last_messages):
            if message["role"] == "assistant":
                # ✅ Ensure message["content"] is a string before processing
                content = message["content"]
                if isinstance(content, str):  # ✅ Check if it's a string
                    words = content.split()
                    for i in range(len(words)):
                        for j in range(i + 2, min(i + 5, len(words) + 1)):  
                            phrase = " ".join(words[i:j])
                            if phrase.istitle():  
                                possible_places.append(phrase)

        # ✅ Check if user query contains a referenced place
        for place in possible_places:
            if place.lower() in self.user_query.lower():
                return place  

        return None  


# -----------------------------
# 🔍 VECTOR SEARCH AGENT
# -----------------------------
class VectorSearchAgent:
    def __init__(self, user_query, memory_manager):
        self.user_query = user_query
        self.memory_manager = memory_manager

    def execute(self):
        """Searches Qdrant for similar establishments and fetches full details from MongoDB"""
        query_vector = encoder.encode(self.user_query).tolist()
        hits = q_client.search(
            collection_name="summary_db",
            query_vector=query_vector,
            limit=10
        )

        threshold = 0.4
        establishment_names = [
            hit.payload.get("name", "") for hit in hits if hit.score > threshold
        ]

        if not establishment_names:
            return ["No matching places found."]

        # ✅ Fetch details from MongoDB
        results = list(collection_m.find({"name": {"$in": establishment_names}}))

        # ✅ Store results in memory
        for place in results:
            self.memory_manager.store_place_details(place["name"], place)

        return results if results else ["No matching places found."]

# -----------------------------
# 📝 INSIDE TEXT REFERENCE AGENT
# -----------------------------

class ReferenceAgent:
    def __init__(self, referenced_place, memory_manager):
        self.referenced_place = referenced_place
        self.memory_manager = memory_manager

    def fetch_details(self):
        """Fetches place details from memory first, then MongoDB if needed"""
        if not self.referenced_place:
            return None

        # ✅ Check if we already stored this place
        stored_details = self.memory_manager.get_place_details(self.referenced_place)
        if stored_details:
            return stored_details

        # ✅ If not stored, search MongoDB (fallback)
        result = collection_m.find_one({"name": {"$regex": self.referenced_place, "$options": "i"}})

        # ✅ Store in memory for next time
        if result:
            self.memory_manager.store_place_details(self.referenced_place, result)

        return result if result else None


# -----------------------------
# 📝 CONTEXTUAL REWRITING AGENT
# -----------------------------
class ContextualRewritingAgent:
    def __init__(self, user_query, enriched_data):
        self.user_query = user_query
        self.enriched_data = enriched_data

    def generate_response(self):
        """Generates a natural, persuasive response while ensuring data security."""

        # 🛠 Handling non-leisure topics gracefully
        if not self.enriched_data or self.enriched_data == ["No matching places found."]:
            prompt = f"""
            Você é um assistente de recomendações de lazer.
            O usuário enviou a seguinte mensagem: "{self.user_query}".
            
            Caso a mensagem esteja fora do tema de lazer (como bares, restaurantes, parques, eventos), responda educadamente explicando seu papel e tente guiar a conversa de volta ao lazer urbano de forma natural.
            
            Se possível, sugira um estabelecimento relevante relacionado ao contexto.
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": prompt}],
                    stream=True,  # ✅ Enable streaming
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content  # ✅ Ensure it streams properly

            except Exception:
                yield "Desculpe, houve um erro ao processar sua mensagem. Tente novamente!"

            return  # ✅ Ensure function exits correctly

        # ✅ Securely format the retrieved establishments
        extracted_info = []
        for place in self.enriched_data:
            place_info = {
                "Nome": place.get("name", "Desconhecido"),
                "Endereço": place.get("formatted_address", "Endereço não disponível"),
                "Status": place.get("business_status", "Status não informado"),  # ✅ Fix KeyError
                "Avaliação": f"{place.get('rating', 'N/A')}⭐ ({place.get('user_ratings_total', 'N/A')} avaliações)",
                "Horários": place.get("opening_hours", {}).get("weekday_text", "Não disponível"),
                "Acessível": "Sim" if place.get("wheelchair_accessible_entrance", False) else "Não",
                "Nível de Preço": place.get("price_level", "Não disponível"),
                "Tipos": ", ".join(place.get("types", [])) if place.get("types") else "Não especificado",
                "Resumo": place.get("summary", "Não disponível"),
                "Serviço de Retirada": "Sim" if place.get("curbside_pickup", False) else "Não",
                "Serviço de Entrega": "Sim" if place.get("takeout", False) else "Não",
                "Serve Cerveja": "Sim" if place.get("serves_beer", False) else "Não",
                "Serve Vinho": "Sim" if place.get("serves_wine", False) else "Não",
                "Website": place.get("website", "Não disponível"),
                "Reviews": " ~ ".join(
                    [f"Review {i+1}: {review.get('text', 'Sem texto')}" for i, review in enumerate(place.get("reviews", []))]
                ) if place.get("reviews") else "Sem avaliações disponíveis."
            }
            extracted_info.append(place_info)

        # ✅ Format extracted information for the LLM prompt
        formatted_places = "\n\n".join(
            [
                f"📍 *{place['Nome']}*\n"
                f"📌 **Endereço:** {place['Endereço']}\n"
                f"🛑 **Status:** {place['Status']}\n"
                f"⭐ **Avaliação:** {place['Avaliação']}\n"
                f"🕒 **Horários:** {place['Horários']}\n"
                f"♿ **Acessível:** {place['Acessível']}\n"
                f"💰 **Nível de Preço:** {place['Nível de Preço']}\n"
                f"🍽️ **Tipos:** {place['Tipos']}\n"
                f"📝 **Resumo:** {place['Resumo']}\n"
                f"🚗 **Serviço de Retirada:** {place['Serviço de Retirada']}\n"
                f"🏠 **Serviço de Entrega:** {place['Serviço de Entrega']}\n"
                f"🍺 **Serve Cerveja:** {place['Serve Cerveja']}\n"
                f"🍷 **Serve Vinho:** {place['Serve Vinho']}\n"
                f"🌐 **Website:** {place['Website']}\n"
                f"🗣️ **Opiniões:** {place['Reviews']}\n"
                for place in extracted_info
            ]
        )

        prompt = f"""
        Você é um assistente de recomendações de lazer.
        O usuário perguntou sobre: "{self.user_query}"
        
        Aqui estão algumas opções que encontramos:
        {formatted_places}
        
        **INSTRUÇÕES:**
        - 🚫 **NÃO** inclua informações técnicas ou dados brutos como JSON, links internos ou códigos de referência.
        - ✅ Apenas use as informações acima para criar uma resposta natural e envolvente.
        - 🎯 Mantenha sua resposta concisa, clara e relevante para o usuário.
        - 🔥 Destaque os melhores aspectos dos lugares sem inventar informações.

        Agora, escreva uma resposta natural com base nos dados acima:
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                stream=True,  # ✅ Enable streaming
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception:
            yield "Desculpe, houve um erro ao gerar a resposta. Tente novamente mais tarde!"
