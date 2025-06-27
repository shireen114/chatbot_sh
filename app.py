# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import requests
import zipfile
import shutil

app = Flask(__name__)

# --- 0. فك ضغط قاعدة البيانات عند الحاجة ---
db_directory = "chroma_db"
zip_file = "chroma_bge_db.zip"

if not os.path.isdir(db_directory):
    print(f"Directory '{db_directory}' not found, extracting from '{zip_file}'...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(db_directory)
        print("Extraction complete.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to extract '{zip_file}': {e}")
        exit()
else:
    print(f"Database directory '{db_directory}' already exists. Skipping extraction.")

# --- 1. إعداد نموذج التضمين (Embedding Model) ---
print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # نموذج سريع
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding model initialized.")

# --- 2. تحميل قاعدة البيانات إلى Chroma ---
try:
    print(f"Loading vector store from: '{db_directory}'...")
    vector_store = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model
    )
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load the Chroma database. {e}")
    exit()

# --- 3. إعداد الـ Retriever ---
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)
print("Retriever is ready.")

# --- 4. إعداد API Key ---
HF_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_llm_api(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
    }
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return f"Error communicating with the API: {e}"
    except (KeyError, IndexError) as e:
        print(f"API Response Parsing Error: {e}")
        return "Error parsing the response from the API."
    except Exception as e:
        print(f"Unexpected error in call_llm_api: {e}")
        return "An unexpected error occurred."

# --- 5. إعداد Flask routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Invalid request. "query" field is missing.'}), 400

        user_query = data['query']
        print(f"Received query: {user_query}")
        answer = ask_question(user_query)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500

def ask_question(query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    print(f"Retrieved context for query '{query}':\n---\n{context}\n---")

    prompt = f"""You are an expert assistant. Your ONLY source of information is the provided "Context".
You MUST answer questions using ONLY the information explicitly given in the "Context".
If the question is a greeting (e.g., "hi", "hello", "hey", "greetings", "how are you"), reply with exactly: "Hello! How can I assist you today? 😊".
If the answer can be found directly or inferred clearly from the "Context", provide that answer concisely.
If the answer is NOT in the "Context" or cannot be directly inferred, you MUST reply with exactly: "Sorry, I don't have enough information about your question"
Do NOT add extra explanations, guesses, or unrelated information.

Context:
{context}

Question:
{query}
"""
    answer = call_llm_api(prompt)
    return answer

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
