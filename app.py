import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests, os

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2',  use_auth_token=False)

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    def split_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            # advance but keep overlap
            next_start = end - self.chunk_overlap
            if next_start <= start:
                start = end
            else:
                start = next_start
        return chunks

def chunk_text(text):
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=100,
      length_function=len
  )
  return splitter.split_text(text)

def embed_chunks(chunks):
    return embedder.encode(chunks)

def build_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search(query, index, chunks, top_k=3):
    q_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(q_embedding), top_k)
    return [chunks[i] for i in indices[0]]

try:
    load_dotenv()
except Exception:
    pass

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise EnvironmentError(
        'HF_TOKEN not found. Set HF_TOKEN as an environment variable or in a local .env file. In VS Code you can add it to the workspace .env or set it in Run/Debug configurations.'
    )

API_URL = 'https://router.huggingface.co/v1/chat/completions'

def query_hf_llm(prompt, max_tokens=300):
    headers = { 'Authorization': f'Bearer {HF_TOKEN}' }
    payload = {
        'messages': [ { 'role': 'user', 'content': prompt } ],
        'model': 'Qwen/Qwen3-1.7B:featherless-ai',
        'parameters': { 'max_new_tokens': max_tokens, 'temperature': 0.3 }
    }
    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()  # raise an exception for HTTP errors
    data = resp.json()
    return data['choices'][0]['message']['content']

def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"""You are a helpful assistant that answers questions about a research paper.

Context:
{context}

Question:
{question}

Answer concisely based only on the context above.
"""

# --- STREAMLIT UI ---

st.title("📄 Ask Your Paper")

uploaded_pdf = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_pdf:
    st.success("PDF uploaded! Extracting text...")
    raw_text = extract_text(uploaded_pdf)
    chunks = chunk_text(raw_text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    st.success(f"Processed {len(chunks)} chunks.")

    st.write("You can now ask a question about the paper:")
    question = st.text_input("Ask your question")

    if question:
        with st.spinner("Thinking..."):
            retrieved = search(question, index, chunks)
            prompt = build_prompt(question, retrieved)
            answer = query_llm(prompt)

        st.markdown("### 💬 Answer")
        st.write(answer)

        with st.expander("🔎 Show retrieved context"):
            for i, chunk in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}**: {chunk[:300]}...")
