"""
embed.py
--------
Chunks your portfolio data, embeds it using your fine-tuned model,
and stores the vectors in Supabase pgvector.

Run once after training:  python embed.py
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BASE_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LORA_MODEL  = "models/portfolio-distilbert"
DATA_DIR    = "data"
CHUNK_SIZE  = 150    # words per chunk
OVERLAP     = 30     # overlapping words between chunks
BATCH_SIZE  = 32     # embed N chunks at once

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── CONNECT TO SUPABASE ──────────────────────────────────────────────────────

def get_supabase():
    """Use service_role key for writing to Supabase."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY missing in .env")
    return create_client(url, key)


# ─── LOAD YOUR FINE-TUNED MODEL ───────────────────────────────────────────────

def load_model():
    """
    Load base MiniLM + your LoRA adapters on top.
    This is YOUR fine-tuned model, not the generic one.
    """
    print("\nLoading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)

    # Load base model
    base = AutoModel.from_pretrained(BASE_MODEL)

    # Load YOUR LoRA weights on top
    model = PeftModel.from_pretrained(base, LORA_MODEL)

    # Merge LoRA weights into base for faster inference
    # After merging: no LoRA overhead, just one clean model
    model = model.merge_and_unload()
    model.eval()
    model.to(device)

    print("Model loaded and LoRA weights merged ✅")
    return tokenizer, model


# ─── EMBEDDING FUNCTION ───────────────────────────────────────────────────────

def mean_pool(token_embeddings, attention_mask):
    mask     = attention_mask.unsqueeze(-1).float()
    sum_emb  = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_emb / sum_mask


def embed_texts(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """
    Convert a list of text strings into embedding vectors.
    Processes in batches for efficiency.
    Returns: numpy array of shape (n_texts, 384)
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(
            batch,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs    = model(**tokens)
            embeddings = mean_pool(outputs.last_hidden_state,
                                   tokens["attention_mask"])
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split text into overlapping chunks.
    Overlap ensures context isn't lost at chunk boundaries.

    Example with overlap:
      Chunk 1: "Aayush built a RAG chatbot using LangChain..."
      Chunk 2: "...using LangChain and Groq for document QA..."
                ^^^^^^^^^^^^^ overlap keeps context
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "content":  chunk_text,
            "source":   source,
            "start":    start,
            "end":      end,
            "n_words":  end - start,
        })

        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def load_and_chunk_all():
    """Load all 3 data files and chunk them."""
    all_chunks = []

    files = {
        "about.txt":    "about",
        "projects.txt": "projects",
        "resume.txt":   "resume",
    }

    for filename, source in files.items():
        filepath = Path(DATA_DIR) / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"{filename}: {len(text.split())} words → {len(chunks)} chunks")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    return all_chunks


# ─── STORE IN SUPABASE ────────────────────────────────────────────────────────

def clear_existing(supabase):
    """Clear old embeddings before re-inserting."""
    print("\nClearing existing embeddings from Supabase...")
    supabase.table("documents").delete().neq("id", 0).execute()
    print("Cleared ✅")


def store_chunks(supabase, chunks, embeddings):
    """
    Store each chunk + its embedding vector in Supabase.
    embedding is stored as a list (pgvector accepts this format).
    """
    print("\nStoring chunks in Supabase...")
    success = 0
    failed  = 0

    for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings),
                                                total=len(chunks))):
        try:
            supabase.table("documents").insert({
                "content":   chunk["content"],
                "embedding": embedding.tolist(),  # numpy → python list
                "metadata":  json.dumps({
                    "source":  chunk["source"],
                    "chunk_id": i,
                    "n_words": chunk["n_words"],
                })
            }).execute()
            success += 1
        except Exception as e:
            print(f"\nFailed chunk {i}: {e}")
            failed += 1

    print(f"\nStored: {success} chunks ✅  |  Failed: {failed}")
    return success


# ─── VERIFY ───────────────────────────────────────────────────────────────────

def verify(supabase, tokenizer, model):
    """
    Quick test: embed a question and do similarity search in Supabase.
    Confirms the full embed → store → retrieve pipeline works.
    """
    print("\n--- Verification: Testing retrieval ---")

    test_query = "What projects has Aayush built?"
    q_embedding = embed_texts([test_query], tokenizer, model)[0]

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": q_embedding.tolist(),
            "match_threshold": 0.3,
            "match_count":     3,
        }
    ).execute()

    if result.data:
        print(f"\nQuery: '{test_query}'")
        print(f"Top {len(result.data)} results:\n")
        for i, doc in enumerate(result.data):
            similarity = doc.get("similarity", "N/A")
            content    = doc["content"][:120]
            source     = json.loads(doc["metadata"])["source"]
            print(f"  [{i+1}] similarity={similarity:.3f} | source={source}")
            print(f"       {content}...")
            print()
        print("Retrieval working ✅")
    else:
        print("No results returned — check match_documents function in Supabase")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Embedding Portfolio Data → Supabase pgvector")
    print("=" * 60)

    # 1. Connect to Supabase
    print("\n[1/5] Connecting to Supabase...")
    supabase = get_supabase()
    print("Connected ✅")

    # 2. Load fine-tuned model
    print("\n[2/5] Loading your fine-tuned model...")
    tokenizer, model = load_model()

    # 3. Load and chunk data files
    print("\n[3/5] Loading and chunking data files...")
    chunks = load_and_chunk_all()

    # 4. Embed all chunks
    print("\n[4/5] Embedding chunks...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(texts, tokenizer, model)
    print(f"Embeddings shape: {embeddings.shape}")  # (n_chunks, 384)

    # 5. Store in Supabase
    print("\n[5/5] Storing in Supabase...")
    clear_existing(supabase)
    stored = store_chunks(supabase, chunks, embeddings)

    # 6. Verify
    verify(supabase, tokenizer, model)

    print("\n" + "=" * 60)
    print("  EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"  Chunks embedded:  {len(chunks)}")
    print(f"  Stored in DB:     {stored}")
    print(f"  Embedding dim:    {embeddings.shape[1]}")
    print(f"  Model used:       {LORA_MODEL}")
    print("\nNext step: run python retriever.py")


if __name__ == "__main__":
    main()