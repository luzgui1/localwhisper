# 🗺️ LocalWhisper — Your Urban Leisure AI Guide

[🔗 Try the live app](https://localwhisper-v001.streamlit.app/)

**LocalWhisper** is an intelligent conversational agent built with LangChain, LangGraph, and Streamlit that acts as a smart city guide. It understands natural language queries, identifies user intent, searches across structured and semantic databases, and delivers personalized recommendations for bars, restaurants, and cultural venues in São Paulo.

---

## 💡 What it does

LocalWhisper simulates a hyper-local leisure assistant. Whether the user asks something vague like _"looking for a cheap beer spot"_ or specific like _"recommend a samba bar near Vila Mariana"_, the system routes the request through agents and tools to generate a meaningful and location-aware response.

---

## 🧠 Tech Stack

- **LangChain** – Prompt engineering and agent interfaces
- **LangGraph** – Multi-agent orchestration
- **Streamlit** – Conversational web interface
- **MongoDB** – Structured data store for places
- **QdrantDB** – Vector store for semantic similarity search
- **OpenAI GPT-4** – Core LLM powering the agents

---

## 🤖 Agent Pipeline

1. **IntentionAgent**  
   Classifies the user’s intention: general, detailed, or irrelevant to urban leisure.

2. **DetailAgent**  
   Extracts user preferences: geographic region, menu details, music genre, reviews.

3. **QdrantSearchTool**  
   Performs semantic search to identify the most contextually relevant places.

4. **MongoSearchTool**  
   Retrieves structured data: address, reviews, pricing, website, etc.

5. **ResponseAgent**  
   Crafts a fluent, localized response using retrieved data and conversational memory.

---

## 🧪 Examples

**User**: "Recommend a samba bar in Vila Mariana"  
**Response**: A curated list with names, descriptions, and website links.

**User**: "Is it well rated?"  
**Response**: An explanation based on top and bottom reviews.

**User**: "sup?"  
**Response**: A friendly nudge toward discovering cool places in town.

---

## 🚀 Running Locally

```bash
# Clone this repository
git clone https://github.com/your-user/localwhisper.git
cd localwhisper

# Install dependencies
pip install -r requirements.txt

# Add your API keys in a .env file
touch .env
# Set:
# MONGO_URL=
# QDRANT_URL=
# QDRANT_KEY=
# OPENAI_API_KEY=

# Run the app
streamlit run app.py


