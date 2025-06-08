# 🗺️ LocalWhisper — Seu Guia Urbano com Inteligência Artificial

[🔗 Acesse o projeto em produção](https://localwhisper-v001.streamlit.app/)

LocalWhisper é um agente conversacional desenvolvido com LangChain, LangGraph e Streamlit que atua como um concierge urbano inteligente. Ele entende seu pedido, classifica a intenção, busca em bancos de dados vetoriais e relacionais, e te entrega sugestões precisas de bares, restaurantes e atividades culturais — tudo com linguagem natural e contextualizada.

---

## 💡 Visão geral

O LocalWhisper simula um especialista em lazer urbano, capaz de interagir como um guia local, entendendo desde pedidos vagos como *"quero um lugar com cerveja barata"* até consultas específicas como *"me recomenda um bar de samba na Vila Mariana?"*.

---

## 🧠 Tecnologias utilizadas

- **LangChain**: modelagem de agentes e prompts
- **LangGraph**: orquestração de múltiplos agentes em cadeia
- **Streamlit**: interface conversacional simples e rápida
- **MongoDB**: banco relacional com dados estruturados dos locais
- **QdrantDB**: base vetorial para buscas semânticas por similaridade
- **OpenAI GPT-4**: motor principal dos agentes

---

## 🔁 Arquitetura dos agentes

1. **IntentionAgent**  
   Classifica a intenção do usuário: genérica, detalhada ou irrelevante.

2. **DetailAgent**  
   Entende detalhes relevantes: localização, cardápio, avaliação ou música.

3. **QdrantSearchTool**  
   Realiza busca semântica baseada na descrição da entrada.

4. **MongoSearchTool**  
   Puxa informações estruturadas como endereço, menu, reviews, etc.

5. **ResponseAgent**  
   Compõe a resposta final, baseada no histórico e nos dados encontrados.

---

## 🧪 Exemplos de uso

**Usuário**: "Quero um bar com samba na Vila Mariana"  
**Resposta**: Lista de bares com descrição, link e recomendação personalizada.

**Usuário**: "Ele tem música ao vivo?"  
**Resposta**: Detalhamento específico do lugar anterior, com memória de contexto.

**Usuário**: "Tá vivo?"  
**Resposta**: Agente responde em tom leve e convida o usuário a explorar o que a cidade tem a oferecer.

---

## 🚀 Como rodar localmente

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/localwhisper.git
cd localwhisper

# Instale as dependências
pip install -r requirements.txt

# Crie um .env com suas chaves
touch .env
# Adicione as variáveis:
# MONGO_URL=
# QDRANT_URL=
# QDRANT_KEY=
# OPENAI_API_KEY=

# Rode localmente
streamlit run app.py

