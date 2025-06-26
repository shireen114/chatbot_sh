from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests

app = Flask(__name__)

# 1. إعداد embedding (نفس الموديل اللي خزّنت به)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. تحميل قاعدة البيانات Chroma
vector_store = Chroma(
    persist_directory="/home/shorouk/Documents/pro/chatbot/chroma_bge_db",
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)

# API Key
HF_API_KEY = "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"

def call_llm_api(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"error {e}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data['query']
        answer = ask_question(user_query)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def ask_question(query):
    docs = retriever.invoke(query) 
    
    if not docs:
        return "I don't know."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert assistant. Your ONLY source of information is the provided "Context".
You MUST answer questions using ONLY the information explicitly given in the "Context".
If the question is a greeting (e.g., "hi", "hello", "hey", "greetings", "how are you"), reply with exactly: "Hello! How can I assist you today? 😊".
If the answer can be found directly or inferred clearly from the "Context", provide that answer concisely.
If the answer is NOT in the "Context" or cannot be directly inferred, you MUST reply with exactly: "Sorry,I don't have enough information about your question"
Do NOT add extra explanations, guesses, or unrelated information.

Context:
{context}

Question:
{query}
"""
    answer = call_llm_api(prompt)
    return answer


if __name__ == '__main__':
    app.run(debug=True)
