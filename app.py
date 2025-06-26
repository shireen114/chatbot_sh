from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import gdown
import os
import requests
import zipfile

app = Flask(__name__)

# إعداد نموذج Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

zip_url = "https://drive.google.com/uc?id=1TRCTZ_txfmdzSfEGr_YXS9h4Kx4ZWNEx"  # رابط تحميل مباشر
zip_path = "chroma_dataset.zip"
extract_path = "chroma_dataset"

if not os.path.exists(extract_path):
    print("Downloading and extracting dataset...")
    gdown.download(url=zip_url, output=zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed.")

# تحميل قاعدة البيانات إلى Chroma
vector_store = Chroma(
    persist_directory=extract_path,
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"
)

# مفتاح OpenRouter API (من الأفضل وضعه في متغير بيئة)
HF_API_KEY = os.environ.get("HF_API_KEY", "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")

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
        return f"error: {e}"

@app.route('/')
def home():
    return render_template('index.html')  # تأكد أن index.html موجود في مجلد templates

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
    return call_llm_api(prompt)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
