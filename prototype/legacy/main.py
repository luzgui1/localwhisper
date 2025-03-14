#%%

import streamlit as st
from pipelines.agents import MemoryManager, UnderstandingAgent, VectorSearchAgent, ContextualRewritingAgent, ReferenceAgent

# Initialize Memory
memory = MemoryManager()

st.title("LocalWhisper - Recomendador de lazer urbano.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Digite sua busca:"):
    memory.store_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Step 1: Understanding Agent - Classify intent & check context
    understanding_agent = UnderstandingAgent(user_input)
    intent = understanding_agent.classify_intent()
    context = understanding_agent.check_context_continuation(memory)
    referenced_place = understanding_agent.extract_referenced_place(memory)

    if context == "new_query":
        memory.reset_memory()

    # Step 2: If user asks about a past place, retrieve details
    if referenced_place:
        ref_agent = ReferenceAgent(referenced_place, memory)
        place_details = ref_agent.fetch_details()

        if place_details:
            response = f"Sobre {referenced_place}: {place_details.get('formatted_address', 'Endereço desconhecido')}. O estabelecimento possui as seguintes características: {place_details}."
        else:
            response = f"Infelizmente, não encontrei detalhes sobre {referenced_place}, mas posso sugerir lugares similares!"

    else:
        # Step 3: Vector Search Agent - Find relevant results
        search_agent = VectorSearchAgent(user_input, memory)
        establishments = search_agent.execute()

        # Step 4: Contextual Rewriting Agent - Generate a natural response
        rewriting_agent = ContextualRewritingAgent(user_input, establishments)
        response = rewriting_agent.generate_response()
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in response:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")  # Simulate typing effect

        response_placeholder.markdown(full_response)  # ✅ Finalize response display

    memory.store_message("assistant", full_response)




#%%