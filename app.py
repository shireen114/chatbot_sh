# -*- coding: utf-8 -*-

import os
import zipfile
from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests

app = Flask(__name__)

# --- إعداد قاعدة البيانات تلقائياً ---
# تحديد المسارات بشكل ديناميكي بالنسبة لموقع السكربت
# This makes the application portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_bge_db")
DB_ZIP_PATH = os.path.join(SCRIPT_DIR, "chroma_bge_db.zip")

def setup_database():
    """
    يفحص وجود مجلد قاعدة البيانات. إذا لم يكن موجودًا،
    يحاول فك ضغطه من ملف .zip محلي.
    This function checks for the database directory. If not found,
    it tries to unzip it from a local .zip file.
    """
    # إذا كان مجلد قاعدة البيانات غير موجود
    if not os.path.exists(DB_DIR):
        print(f"المجلد '{DB_DIR}' غير موجود. يتم البحث عن '{DB_ZIP_PATH}'.")
        # تحقق من وجود الملف المضغوط
        if os.path.exists(DB_ZIP_PATH):
            print(f"تم العثور على '{DB_ZIP_PATH}'. جاري فك الضغط...")
            try:
                # قم بفك ضغط الملف
                with zipfile.ZipFile(DB_ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(SCRIPT_DIR)
                print("تم فك الضغط بنجاح.")
            except Exception as e:
                print(f"حدث خطأ أثناء فك ضغط الملف: {e}")
                # إنهاء البرنامج إذا كانت قاعدة البيانات ضرورية
                exit()
        else:
            # إذا لم يتم العثور على المجلد أو الملف المضغوط
            print(f"خطأ: مجلد قاعدة البيانات والملف المضغوط غير موجودين.")
            exit()

# قم بتشغيل الإعداد عند بدء تشغيل التطبيق
setup_database()
# --- نهاية إعداد قاعدة البيانات ---


# 1. إعداد embedding (نفس الموديل اللي خزّنت به)
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding model loaded.")

# 2. تحميل قاعدة البيانات Chroma من المسار المحلي
print("Loading Chroma vector store...")
vector_store = Chroma(
    persist_directory=DB_DIR,  # استخدام المسار الديناميكي الجديد
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)
print("Chroma vector store loaded.")


# API Key - من الأفضل تحميله من متغيرات البيئة لمزيد من الأمان
# It's better to load the API key from environment variables for security
HF_API_KEY = os.getenv("OPENROUTER_API_KEY")
if "sk-or-v1" in HF_API_KEY:
    print("Warning: Using a hardcoded API key. Consider using environment variables.")


def call_llm_api(prompt: str):
    """Calls the OpenRouter LLM API."""
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
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx responses
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return f"Error communicating with the language model API: {e}"
    except (KeyError, IndexError) as e:
        print(f"API response parsing error: {e}")
        return f"Error parsing the response from the language model API: {e}"
    except Exception as e:
        print(f"An unexpected error occurred in call_llm_api: {e}")
        return f"An unexpected error occurred: {e}"


@app.route('/')
def home():
    """Renders the main chat page."""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """Handles the user's query and returns the answer."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Invalid request. "query" is required.'}), 400

        user_query = data['query']
        answer = ask_question(user_query)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500


def ask_question(query: str) -> str:
    """
    Retrieves context from the vector store and generates an answer using the LLM.
    """
    print(f"Retrieving documents for query: '{query}'")
    docs = retriever.invoke(query)

    if not docs:
        print("No relevant documents found in the vector store.")
        return "I don't know."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert assistant. Your ONLY source of information is the provided "Context".
You MUST answer questions using ONLY the information explicitly given in the "Context".
If the question is a greeting (e.g., "hi", "hello", "hey", "greetings", "how are you"), reply with exactly: "Hello! How can I assist you today? �".
If the answer can be found directly or inferred clearly from the "Context", provide that answer concisely.
If the answer is NOT in the "Context" or cannot be directly inferred, you MUST reply with exactly: "Sorry, I don't have enough information about your question"
Do NOT add extra explanations, guesses, or unrelated information.

Context:
{context}

Question:
{query}
"""
    print("Sending prompt to LLM...")
    answer = call_llm_api(prompt)
    return answer


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
