import streamlit as st
import os
import re
import faiss
import numpy as np
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

# ---------------- LOAD ENV ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 Nexus IQ Solutions – HR Policy Assistant")
st.caption("Hybrid RAG-based HR chatbot (Accurate • Safe • Professional)")

# ---------------- PDF PATH ----------------
PDF_PATH = "Sample_HR_Policy.pdf"

if not os.path.exists(PDF_PATH):
    st.error("HR Policy PDF not found. Please upload the file.")
    st.stop()

# ---------------- LOAD & PROCESS PDF ----------------
@st.cache_resource
def load_rag_pipeline():

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return texts, embedder, index

texts, embedder, index = load_rag_pipeline()

# ---------------- KEYWORD FILTER ----------------
def keyword_filter(query, texts):
    keywords = re.findall(r"\w+", query.lower())
    filtered = [
        t for t in texts
        if any(k in t.lower() for k in keywords)
    ]
    return filtered if filtered else texts

# ---------------- HYBRID SEARCH ----------------
def hybrid_search(query, top_k=3):

    filtered_texts = keyword_filter(query, texts)

    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, top_k)

    results = [texts[i] for i in I[0]]
    return results

# ---------------- LLM ANSWER ----------------
def generate_answer(context, question):

    prompt = f"""
You are an HR policy assistant.

Rules:
- Answer ONLY from the provided HR policy context.
- Do NOT guess or add information.
- If the answer is not present, say politely that it is not mentioned.

HR Policy Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ---------------- UI ----------------
question = st.text_input("Ask your HR-related question:")

if st.button("Get Answer") and question:

    retrieved_chunks = hybrid_search(question)
    context = "\n\n".join(retrieved_chunks)

    if not context.strip():
        st.info(
            "I checked the HR policy document, but this information is not mentioned. "
            "Please reach out to the HR team for further clarification."
        )
    else:
        answer = generate_answer(context, question)
        st.success(answer)
