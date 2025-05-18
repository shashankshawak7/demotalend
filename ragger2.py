import os
from flask import Flask, request, render_template_string
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# ========== CONFIGURATION ========== #

# Tree-sitter
TREE_SITTER_LIB_PATH = "build/my-languages.so"
TREE_SITTER_LANGUAGE = "java"
TREE_SITTER_GRAMMAR_PATHS = ["tree-sitter-java"]

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB
CHROMA_DIR = "./chroma"
CHROMA_COLLECTION_NAME = "java_chunks"

# Ollama
OLLAMA_MODEL = "mistrals"
OLLAMA_API_URL = "http://localhost:11435/api/generate"

# Flask
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# ==================================== #

# === Tree-sitter setup ===
# if not os.path.exists(TREE_SITTER_LIB_PATH):
#     Language.build_library(TREE_SITTER_LIB_PATH, TREE_SITTER_GRAMMAR_PATHS)
import tree_sitter_java as tsj
JAVA_LANGUAGE = Language(tsj.language())
parser = Parser(JAVA_LANGUAGE)
# parser.set_language(JAVA_LANGUAGE)

# === Embedding model ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === ChromaDB setup ===
chroma_client = chromadb.Client(Settings(
    persist_directory=CHROMA_DIR,
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# === Function to parse Java file and extract class/method chunks ===
def extract_chunks(code: str):
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    chunks = []

    def walk(node):
        if node.type in ["class_declaration", "method_declaration"]:
            chunks.append(code[node.start_byte:node.end_byte])
        for child in node.children:
            walk(child)

    walk(root)
    return chunks

# === Function to embed and store chunks in Chroma ===
def embed_and_store(chunks, filename):
    collection.delete(where={"file": filename})  # Safe deletion
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"file": filename}] * len(chunks)
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

# === Function to query relevant chunks and ask Ollama ===
def ask_ollama(query):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    context = "\n\n".join(results["documents"][0]) if results["documents"] else "No relevant context found."

    prompt = f"""You are a helpful Java assistant. Given the following context, answer the query.

Context:
{context}

Query:
{query}

Answer:"""

    response = requests.post(OLLAMA_API_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })

    return response.json()["response"]

# === Flask web app ===
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<title>Java RAG</title>
<h2>Upload Java File</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=javafile>
  <input type=submit value=Upload>
</form>
{% if chunks %}
  <h3>Extracted Chunks:</h3>
  <ul>
  {% for c in chunks %}
    <li><pre>{{ c }}</pre></li>
  {% endfor %}
  </ul>
{% endif %}

<h2>Ask a Question</h2>
<form method=post>
  <input type=text name=query style="width:400px">
  <input type=submit value=Ask>
</form>
{% if answer %}
<h3>Answer:</h3>
<pre>{{ answer }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    chunks = []
    answer = None

    if request.method == "POST":
        if "javafile" in request.files:
            f = request.files["javafile"]
            code = f.read().decode("utf-8")
            chunks = extract_chunks(code)
            embed_and_store(chunks, f.filename)

        if "query" in request.form:
            query = request.form["query"]
            answer = ask_ollama(query)

    return render_template_string(TEMPLATE, chunks=chunks, answer=answer)

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
s