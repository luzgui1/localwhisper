# app.py
import streamlit as st
from main import run_pipeline
from langchain_core.messages import HumanMessage, AIMessage
import os

mongo_url = os.environ.get("MONGO_URL")
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_key = os.environ.get("QDRANT_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="LocalWhisper", layout="centered")
st.title("🗺️ LocalWhisper - seu guia de lazer urbano")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent_result = {}

user_input = st.chat_input("O que você procura hoje?")

if user_input:
    with st.spinner("Consultando agentes..."):
        response, agent_result, updated_history = run_pipeline(user_input, st.session_state.chat_history)
        st.session_state.chat_history = updated_history
        agent_result["user_input"] = user_input

# Renderiza o histórico de chat
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Só mostra dados brutos se houver algo
if agent_result:
    with st.expander("🔍 Dados brutos dos agentes"):
        st.json(agent_result)
