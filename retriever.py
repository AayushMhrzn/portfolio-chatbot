"""
retriever.py
------------
Handles semantic search against Supabase pgvector.
Given a user question, returns the most relevant chunks
from your portfolio data.

Can be run standalone to test retrieval:
  python retriever.py
"""

import os
import json
#import torch
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from transformers import AutoTokenizer
#from peft import PeftModel
#import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForFeatureExtraction

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ONNX_MODEL = "models/portfolio-onnx"
MATCH_COUNT     = 5      # how many chunks to retrieve
MATCH_THRESHOLD = 0.15   # lowered — our model scores in 0.2-0.5 range
MAX_LENGTH      = 256

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── SINGLETON MODEL LOADER ───────────────────────────────────────────────────
# Load model once and reuse — avoids reloading on every request in FastAPI

_tokenizer = None
_model     = None

def get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading ONNX model...")
        _tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL)
        _model     = ORTModelForFeatureExtraction.from_pretrained(ONNX_MODEL)
        print("ONNX model ready ✅")
    return _tokenizer, _model


# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list:
    tokenizer, model = get_model()

    tokens = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="np"   # IMPORTANT: use numpy directly
    )

    outputs = model(**tokens)

    embedding = outputs.last_hidden_state  # already numpy in ORT

    mask = tokens["attention_mask"][..., None]

    sum_emb = (embedding * mask).sum(axis=1)
    sum_mask = np.clip(mask.sum(axis=1), 1e-9, None)

    emb = sum_emb / sum_mask

    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    return emb[0].tolist()


# ─── SUPABASE CLIENT ──────────────────────────────────────────────────────────

def get_supabase():
    """
    Use anon key for reading — matches our RLS policy.
    (Service key for writing in embed.py, anon key for reading here)
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")   # anon key — read only
    return create_client(url, key)


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def retrieve(query: str,
             match_count: int = MATCH_COUNT,
             match_threshold: float = MATCH_THRESHOLD) -> list[dict]:
    """
    Main retrieval function.
    1. Embed the query
    2. Search Supabase for similar chunks
    3. Return ranked list of relevant chunks

    Returns list of dicts:
    [
      {
        "content": "Aayush built a RAG chatbot...",
        "similarity": 0.82,
        "source": "projects",
        "chunk_id": 5
      },
      ...
    ]
    """
    supabase       = get_supabase()
    query_embedding = embed_query(query)

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count":     match_count,
        }
    ).execute()

    if not result.data:
        return []

    chunks = []
    for doc in result.data:
        try:
            metadata = json.loads(doc.get("metadata", "{}"))
        except Exception:
            metadata = {}

        chunks.append({
            "content":    doc["content"],
            "similarity": round(doc.get("similarity", 0), 4),
            "source":     metadata.get("source", "unknown"),
            "chunk_id":   metadata.get("chunk_id", -1),
        })

    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string
    to be injected into the LLM prompt.

    Groups chunks by source for cleaner context.
    """
    if not chunks:
        return "No relevant information found."

    # Group by source
    by_source = {}
    for chunk in chunks:
        src = chunk["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(chunk["content"])

    # Format
    parts = []
    source_labels = {
        "about":    "About Aayush",
        "projects": "Projects",
        "resume":   "Resume & Skills",
        "unknown":  "General Info",
    }
    for source, contents in by_source.items():
        label = source_labels.get(source, source.title())
        combined = " ".join(contents)
        parts.append(f"[{label}]\n{combined}")

    return "\n\n".join(parts)


# ─── RERANKING ────────────────────────────────────────────────────────────────

def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """
    Simple keyword-based reranking on top of vector search.
    Boosts chunks that contain exact keywords from the query.

    In production you'd use a cross-encoder model for this
    (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2).
    For our portfolio project, keyword boosting is sufficient.
    """
    query_words = set(query.lower().split())

    # Remove common stop words
    stop_words = {"what", "who", "where", "when", "how", "is", "are",
                  "did", "does", "do", "the", "a", "an", "about",
                  "tell", "me", "his", "her", "their", "has", "have",
                  "can", "could", "would", "aayush", "he", "his"}
    keywords = query_words - stop_words

    for chunk in chunks:
        content_lower = chunk["content"].lower()
        # Count keyword hits
        hits = sum(1 for kw in keywords if kw in content_lower)
        # Boost similarity score by keyword matches
        boost = hits * 0.05
        chunk["reranked_score"] = round(chunk["similarity"] + boost, 4)

    # Sort by reranked score
    chunks.sort(key=lambda x: x["reranked_score"], reverse=True)
    return chunks


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────

def get_context(query: str) -> tuple[str, list[dict]]:
    """
    Full retrieval pipeline:
    1. Vector search → top 5 chunks
    2. Keyword reranking → reorder by relevance
    3. Format into context string for LLM

    Returns (context_string, chunks_list)
    """
    # Step 1: Vector retrieval
    chunks = retrieve(query)

    if not chunks:
        # Fallback: try with even lower threshold
        chunks = retrieve(query, match_threshold=0.05)

    if not chunks:
        return "No relevant information found in portfolio data.", []

    # Step 2: Rerank
    chunks = rerank(query, chunks)

    # Step 3: Format context
    context = format_context(chunks)

    return context, chunks


# ─── STANDALONE TEST ──────────────────────────────────────────────────────────

def test_retriever():
    """Test retrieval with several queries."""
    print("=" * 60)
    print("  Retriever Test")
    print("=" * 60)

    test_queries = [
        "What projects has Aayush built?",
        "Does Aayush know Docker?",
        "What is Aayush's educational background?",
        "Tell me about the Network Intrusion Detection project",
        "What certifications does Aayush have?",
        "Why should I hire Aayush?",
        "What tech stack does Aayush use for AI projects?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        context, chunks = get_context(query)

        if chunks:
            print(f"Retrieved {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks[:3]):  # show top 3
                print(f"  [{i+1}] score={chunk.get('reranked_score', chunk['similarity']):.3f} "
                      f"| source={chunk['source']} "
                      f"| {chunk['content'][:80]}...")
        else:
            print("  No chunks retrieved ❌")

    print("\n" + "=" * 60)
    print("  Context formatting test")
    print("=" * 60)
    query   = "What AI projects has Aayush built?"
    context, chunks = get_context(query)
    print(f"\nQuery: {query}")
    print(f"Context ({len(context.split())} words):\n")
    print(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    test_retriever()