import os
import requests
from flask import Flask, request, render_template_string
from tree_sitter import Language, Parser
import chromadb
import shutil
import tree_sitter_java as tsj

# === CONFIG ===
UPLOAD_FOLDER = "uploads"
OLLAMA_HOST = "http://localhost:11435"
EMBEDDING_MODEL = "mistral"
GENERATION_MODEL = "mistral"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === INIT TREE-SITTER JAVA ===
# if not os.path.exists("build/my-languages.so"):
#     Language.build_library("build/my-languages.so", ["tree-sitter-java"])
JAVA_LANGUAGE = Language(tsj.language())
parser = Parser(JAVA_LANGUAGE)

# === INIT CHROMADB ===
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="java_chunks")

# === HELPERS ===
def read_java_code(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def extract_chunks(source_code):
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    chunks = []

    def visit(node):
        if node.type in ["class_declaration", "method_declaration", "constructor_declaration", "method_invocation"]:
            snippet = source_code[node.start_byte:node.end_byte]
            chunks.append(snippet.strip())
        for child in node.children:
            visit(child)

    visit(root_node)
    return chunks

def embed(text):
    res = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": EMBEDDING_MODEL, "prompt": text})
    return res.json()["embedding"]

def generate_response(context, question):
    prompt = f"""Use the following Java code to answer the question.

### Code:
{context}

### Question:
{question}
"""
    res = requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": GENERATION_MODEL, "prompt": prompt, "stream": False})
    return res.json()["response"]

# === FLASK APP ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HTML_TEMPLATE = """
<!doctype html>
<title>Java RAG Assistant</title>
<h2>📁 Upload a Java file</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=java_file>
  <input type=submit value="Upload and Index">
</form>

{% if uploaded %}
<p><b>✅ Uploaded and indexed:</b> {{ uploaded }}</p>
<form method=post>
  <input name=question placeholder="Ask a question about the code..." style="width: 60%;">
  <input type=submit value="Ask">
</form>
{% endif %}

{% if answer %}
<h3>🤖 Answer:</h3>
<pre>{{ answer }}</pre>

<h4>🔍 Retrieved Code Chunks:</h4>
{% for chunk in chunks %}
<pre style="background:#f0f0f0;padding:10px;border-radius:8px;">{{ chunk }}</pre>
{% endfor %}
{% endif %}
"""

# === APP STATE ===
uploaded_filename = None

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_filename
    answer = ""
    chunks = []
    uploaded = uploaded_filename

    # Handle upload
    if request.method == "POST" and "java_file" in request.files:
        f = request.files["java_file"]
        if f.filename.endswith(".java"):
            uploaded_filename = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
            f.save(uploaded_filename)
            uploaded = f.filename

            # Clear old index
            collection.delete(where={})
            code = read_java_code(uploaded_filename)
            code_chunks = extract_chunks(code)

            for i, chunk in enumerate(code_chunks):
                emb = embed(chunk)
                collection.add(documents=[chunk], embeddings=[emb], ids=[f"{uploaded}_{i}"])
    
    # Handle question
    elif request.method == "POST" and "question" in request.form:
        q = request.form["question"]
        emb = embed(q)
        results = collection.query(query_embeddings=[emb], n_results=5)
        chunks = results["documents"][0]
        context = "\n\n".join(chunks)
        answer = generate_response(context, q)

    return render_template_string(HTML_TEMPLATE, answer=answer, chunks=chunks, uploaded=uploaded)

if __name__ == "__main__":
    app.run(debug=True)
