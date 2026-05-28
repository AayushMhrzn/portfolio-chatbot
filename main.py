"""
main.py
-------
FastAPI backend for the Portfolio RAG Chatbot.
Exposes a /chat endpoint that:
  1. Receives user question
  2. Retrieves relevant chunks from Supabase
  3. Builds context-aware prompt
  4. Calls Groq LLM for answer
  5. Returns response

Run locally:
  uvicorn main:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What projects has Aayush built?"}'
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from retriever import get_context

load_dotenv()

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Portfolio RAG Chatbot API",
    description="AI-powered chatbot that answers questions about Aayush Maharjan",
    version="1.0.0",
)

# CORS — allows your portfolio website to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []  # optional — for multi-turn memory

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What projects has Aayush built?",
                "conversation_history": []
            }
        }

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant on Aayush Maharjan's portfolio website.
Your ONLY job is to answer questions about Aayush based on the context provided.

STRICT RULES:
- Answer naturally and professionally, as if you are Aayush's personal assistant
- Keep answers clear, specific, and professional
- Keep answers concise but complete
- Maximum 3-4 sentences unless listing items
- Always maintain a positive and professional tone about Aayush
- When asked "who is Aayush", "tell me about Aayush Maharjan", "introduce Aayush", or similar 
  broad identity questions — give a 3-4 sentence summary covering:
  name, location, degree, top skills, notable projects, and what he is looking for
- When asked "what projects" or broadly about projects or experience — list the projects given inside the triple backticks and ask if they want details on any specific one
```
1. Audio-Driven Facial Animation System - generates Avatar speaking animation through audio input only using CNN-TCN deep learning approach
2. Network Intrusion Detection System (NIDS) - multi-classification of Web Traffic Attacks using Random Forest Classifier
3. Interview ChatBot — Practice Interview with Relevance scoring, Sentiment Analysis, and Feedback Reporting
4. RAG Chatbot  — LangChain, FAISS, HuggingFace embeddings, Groq LLM, multi-mode, PDF Q&A
5. Portfolio RAG Chatbot — Fine-tuned sentence-transformers using LoRA, FastAPI, Supabase pgvector, Docker, Groq LLM
6. DIY HOME CCTV project — built using Spare Webcam,Person Detection - YOLOv8, Alert Notification, Remote CCTV feed access
7. PONG game — using only C Programming Language, interactive user interface MENU, Difficulty Levels, Player 1-2 Mode, Scoreboard
```
- When asked about a specific project — give details: problem, tech stack, results of that specific project
- When asked about certifications — list them all specifically
- When asked about skills - include terms tech/AI frameworks like TensorFlow, Keras, PyTorch, HuggingFace, Scikit-learn, LangChain, Computer Vision, LLMs, RAG, Be specific not bluff, include Soft Skills
- When asked about location - give the location he is based in
- When asked why hire Aayush — highlight: production AI skills,certifications, hackathon winner, top 60 learner, real projects, Soft Skills, scholarships, and more — but be specific and avoid generic fluff
- NEVER say "a project" or "one of his projects" — always name it specifically
- NEVER make up information not in the context
- If truly no information exists — say:  "I don't have specific details about that, but you can reach Aayush directly at aayushmaharjan.94@gmail.com"
-
"""


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

def build_prompt(question: str, context: str, history: list[dict]) -> list[dict]:
    """
    Build the full message list for Groq including:
    - System prompt (who the assistant is)
    - Conversation history (for multi-turn memory)
    - Current question with retrieved context injected
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 6 turns max to avoid token overflow)
    if history:
        messages.extend(history[-6:])

    # Add current question with context
    user_message = f"""Context from Aayush's portfolio:
---
{context}
---

Question: {question}

Please answer based on the context above."""

    messages.append({"role": "user", "content": user_message})
    return messages


# ─── CHAT ENDPOINT ────────────────────────────────────────────────────────────
 
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a question, retrieves context, generates answer.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
 
    if len(request.message) > 500:
        raise HTTPException(status_code=400, detail="Message too long (max 500 chars)")
 
    try:
        # Step 1: Retrieve relevant context
        context, chunks = get_context(request.message)
 
        # Step 2: Build prompt
        messages = build_prompt(
            question=request.message,
            context=context,
            history=request.conversation_history,
        )
 
        # Step 3: Call Groq LLM
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=512,
            temperature=0.3,   # low temp = more factual, less creative
        )
 
        answer  = response.choices[0].message.content.strip()
        sources = list(set(c["source"] for c in chunks))
 
        return ChatResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
 
 
# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
 
@app.get("/health")
async def health():
    """Health check endpoint — used by Render to verify app is running."""
    return {
        "status": "healthy",
        "model":  "llama-3.3-70b-versatile",
        "db":     "supabase-pgvector",
    }
 
 
@app.get("/")
async def root():
    return {
        "message": "Portfolio RAG Chatbot API",
        "docs":    "/docs",
        "chat":    "/chat",
        "health":  "/health",
    }
 
 
# ─── STARTUP EVENT ────────────────────────────────────────────────────────────
 
# @app.on_event("startup")
# async def startup_event():
#     """
#     Pre-load the embedding model when FastAPI starts.
#     This way the first user request isn't slow.
#     """
#     print("Pre-loading embedding model on startup...")
#     from retriever import get_model
#     get_model()
#     print("API ready ✅")
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)