
import streamlit as st
from pipeline.tools import get_user_location
from pipeline.agents import Agents
from pipeline import logger

logger.info("APP: App starting.")
st.set_page_config(page_title="Ivy", layout="centered")
st.title("Ivy 🌿")

# ------------------------
# Session init
# ------------------------
logger.info("Default session_state defined.")

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("user_location", {})
st.session_state.setdefault("places_nearby", [])

# ------------------------
# Sidebar: location & controls
# ------------------------
with st.sidebar:
    st.header("📍 Localização")

    user_location = get_user_location()
    st.session_state["user_location"] = user_location

    if isinstance(user_location, dict) and "lat" in user_location and "lng" in user_location:
        st.success(f"OK: {user_location['lat']:.6f}, {user_location['lng']:.6f}")
    else:
        st.info("Clique em **📍 Usar minha localização** para continuar.")

    logger.info(f"APP: Location defined: '{st.session_state['user_location']}'")

    st.divider()

    if st.button("🧹 Limpar cache de lugares"):
        st.session_state["places_nearby"] = []
        st.success("Cache limpo.")

    if st.button("🗑️ Limpar conversa"):
        st.session_state["chat_history"] = []
        st.success("Conversa limpa.")
        st.rerun()

# ------------------------
# Location status in main area
# ------------------------

current_loc = st.session_state.get("user_location")

if current_loc:
    st.success(f"Localização capturada: {current_loc['lat']:.6f}, {current_loc['lng']:.6f}")
else:
    st.info("Nenhuma localização capturada ainda.")

# ------------------------
# Agent init
# ------------------------

agent = Agents(model_name="gpt-4", temperature=0.4)

# ------------------------
# Render chat history
# ------------------------

for msg in st.session_state["chat_history"]:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------
# Chat input
# ------------------------

user_input = st.chat_input("O que você procura hoje? (ex: bar animado, almoço rápido, date...)")

if user_input:
    # user message
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    logger.info(f"APP: 'user-input': {user_input}'")

    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent answer (routing + tools handled inside Agents)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            reply = agent.respond(user_input, st.session_state)
            st.markdown(reply)

    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
