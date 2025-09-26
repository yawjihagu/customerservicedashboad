# streamlit_rag_chatbot.py
# Simple RAG (Retrieval-Augmented Generation) chatbot built with Streamlit + OpenAI.
# Features:
# - Upload your own PDF or TXT files
# - Automatic text extraction + chunking
# - Create embeddings using OpenAI Embeddings
# - Very small in-memory vector store (no external DB required)
# - Retrieve top-k chunks and build a context to send to the LLM
# - Fallback: if no docs uploaded OR user chooses, answer without docs

import streamlit as st
import openai
import numpy as np
from typing import List, Dict
from pypdf import PdfReader

# -----------------------------
# Helper functions
# -----------------------------

def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
        return "\n\n".join(text)
    except Exception:
        return ""


def read_text_file(file) -> str:
    try:
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return raw
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def get_embeddings(openai_client, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    embeddings = []
    resp = openai_client.Embedding.create(model=model, input=texts)
    for item in resp.data:
        embeddings.append(item.embedding)
    return embeddings


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def retrieve_top_k(query_embedding: List[float], store: List[Dict], k: int = 3) -> List[Dict]:
    if not store:
        return []
    q = normalize(np.array(query_embedding))
    scores = []
    for item in store:
        vec = np.array(item["embedding"])
        vec = normalize(vec)
        score = cosine_sim(q, vec)
        scores.append(score)
    idxs = np.argsort(scores)[::-1][:k]
    results = [store[i] for i in idxs]
    for i, r in enumerate(results):
        r["score"] = float(scores[idxs[i]])
    return results


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="Simple RAG Chatbot", layout="wide")
st.title("📚 Simple RAG Chatbot — Streamlit + OpenAI")

# API key input
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API key", type="password", value=st.session_state.openai_api_key)
    if api_key:
        st.session_state.openai_api_key = api_key

    embedding_model = st.selectbox("Embedding model", options=["text-embedding-3-small", "text-embedding-3-large"], index=0)
    chat_model = st.selectbox("Chat model", options=["gpt-3.5-turbo", "gpt-4o-mini"], index=0)
    top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=10, value=3)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=200, step=50)

st.markdown("Upload PDF or TXT files (you can upload multiple). The app will index them in memory.")
uploaded_files = st.file_uploader("Upload PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if "store" not in st.session_state:
    st.session_state.store = []

# Indexing step
if uploaded_files:
    if not st.session_state.openai_api_key:
        st.warning("Please set your OpenAI API key in the sidebar before uploading files.")
    else:
        openai.api_key = st.session_state.openai_api_key
        new_chunks = []
        new_metadatas = []
        for f in uploaded_files:
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(f)
            else:
                text = read_text_file(f)

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            for i, c in enumerate(chunks):
                new_chunks.append(c)
                new_metadatas.append({"source": f.name, "chunk_index": i})

        if new_chunks:
            with st.spinner("Creating embeddings..."):
                embeddings = get_embeddings(openai, new_chunks, model=embedding_model)

            for txt, meta, emb in zip(new_chunks, new_metadatas, embeddings):
                st.session_state.store.append({"content": txt, "metadata": meta, "embedding": emb})

            st.success(f"Indexed {len(new_chunks)} chunks from {len(uploaded_files)} file(s).")
        else:
            st.info("No text found in the uploaded files.")

# -----------------------------
# Chat UI
# -----------------------------

st.subheader("Ask questions")
col1, col2 = st.columns([3, 1])
with col1:
    user_question = st.text_area("Your question", height=120)
    ask_button = st.button("Ask")
with col2:
    # Toggle: use docs or not
    use_docs = st.radio("Answer mode:", ["Use uploaded documents (RAG)", "General AI answer"], index=0)

    if st.session_state.store:
        st.write(f"Indexed chunks: {len(st.session_state.store)}")
        sources = {}
        for item in st.session_state.store:
            src = item["metadata"]["source"]
            sources[src] = sources.get(src, 0) + 1
        st.write("**Sources:**")
        for s, c in sources.items():
            st.write(f"- {s}: {c} chunks")

# -----------------------------
# Handle Q&A
# -----------------------------
if ask_button:
    if not st.session_state.openai_api_key:
        st.warning("Set your OpenAI API key in the sidebar first.")
    elif not user_question:
        st.warning("Please type a question.")
    else:
        openai.api_key = st.session_state.openai_api_key

        # Case 1: Use RAG if chosen and docs exist
        if use_docs == "Use uploaded documents (RAG)" and st.session_state.store:
            with st.spinner("Retrieving..."):
                q_emb = get_embeddings(openai, [user_question], model=embedding_model)[0]
                top_chunks = retrieve_top_k(q_emb, st.session_state.store, k=int(top_k))

            context = "\n\n".join(
                [f"Source: {c['metadata']['source']} (chunk {c['metadata']['chunk_index']})\n{c['content']}" for c in top_chunks]
            )

            system_prompt = (
                "You are a helpful assistant. Use the provided CONTEXT to answer the user question. "
                "If the answer is not contained in the context, say you don't know and avoid hallucination. "
                "Cite the source filename and chunk index when referencing the document."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n---\nQUESTION:\n{user_question}"},
            ]

        # Case 2: General AI answer
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question as best as you can."},
                {"role": "user", "content": user_question},
            ]
            top_chunks = []

        # Generate response
        with st.spinner("Generating answer from the LLM..."):
            try:
                resp = openai.ChatCompletion.create(
                    model=st.session_state.get("chat_model", chat_model),
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                answer = None

        if answer:
            st.markdown("### Answer")
            st.write(answer)

        # Show snippets only if docs were used
        if use_docs == "Use uploaded documents (RAG)" and top_chunks:
            st.markdown("---")
            st.markdown("### Retrieved snippets (ranked)")
            for i, c in enumerate(top_chunks, start=1):
                st.write(
                    f"**{i}. Source:** {c['metadata']['source']} "
                    f"(chunk {c['metadata']['chunk_index']}) — score: {c.get('score', 0):.4f}"
                )
                st.write(c['content'][:1000] + ("..." if len(c['content']) > 1000 else ""))

# -----------------------------
# Sidebar notes
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Notes**:\n"
    "- This app uses a tiny in-memory vector store. It's not persistent across sessions.\n"
    "- For production, use a vector DB (Chroma, Pinecone, Weaviate) and batching for embeddings.\n"
    "- Keep your API key safe."
)
