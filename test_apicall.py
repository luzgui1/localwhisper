import requests

def ai_agent(user_query):
    """
    AI Agent that takes user input, queries the Vector Search API,
    and returns structured recommendations.
    """
    api_url = "http://127.0.0.1:8000/vector_search/"
    params = {"query_text": user_query, "top_n": 5}

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        recommendations = response.json()
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"

    # Format the response for better readability
    if not recommendations:
        return "No recommendations found."

    formatted_response = "\n".join(
        [f"📍 {rec['name']} ({rec['score']}⭐)\n📝 {rec['types']} \n {rec['summary']}" for rec in recommendations]
    )
    return formatted_response

# Example Usage:
if __name__ == "__main__":
    user_query = "tenho 27 anos, quero buscar um bar onde eu consiga ouvir um bom samba"
    print(ai_agent(user_query))
