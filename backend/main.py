"""
main.py
--------
FastAPI backend for the "Persistent Multi-Thread Conversational AI with
Universal Memory" application.

Architecture overview
======================
1. Multiple chat THREADS are stored in SQLite (each thread = its own
   conversation, like separate tabs).
2. A GLOBAL "memories" table stores long-term facts about the user
   (preferences, goals, hobbies, emotional context, etc.) that are
   extracted automatically from user messages -- and these memories
   are shared across EVERY thread, not scoped to one conversation.
3. On every chat turn:
     - the user message is persisted
     - an LLM call decides whether the message contains a worthwhile
       long-term memory; if so, it's embedded with SentenceTransformer
       and stored globally
     - the new user message is embedded, and compared (cosine
       similarity) against all stored memory embeddings to retrieve
       the most relevant ones
     - a SLIDING WINDOW of only the last 12 messages from the current
       thread is sent to the model (NOT the full history) -- this
       keeps latency/cost bounded while the global memory store keeps
       long-term continuity
     - the retrieved memories are injected into the system prompt so
       the model "remembers" things from other conversations too
4. The model is Groq's `llama-3.3-70b-versatile`, accessed through the
   OpenAI-compatible Groq endpoint via the official `openai` SDK.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from database import Base, engine, get_db
from models import Thread, Message, Memory

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-memory-chat")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set. Add it to your .env file before chatting."
    )

# Groq client (OpenAI-compatible SDK pointed at Groq's endpoint)
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# Sentence embedding model -- loaded once at startup and reused everywhere
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create tables if they don't exist yet
Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="AI Memory Chat", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------


class ThreadCreate(BaseModel):
    title: Optional[str] = "New Chat"


class ThreadOut(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    thread_id: int
    message: str


class ChatResponse(BaseModel):
    reply: str
    memories_used: List[str] = []
    memory_stored: Optional[str] = None


# --------------------------------------------------------------------------
# Embedding / memory helper functions
# --------------------------------------------------------------------------

SLIDING_WINDOW_SIZE = 12  # number of most-recent messages sent to the model
TOP_K_MEMORIES = 5  # how many relevant memories to retrieve
SIMILARITY_THRESHOLD = 0.50  # minimum cosine similarity to count as "relevant"


def get_embedding(text: str) -> List[float]:
    """Encode a string into a sentence embedding (as a plain python list)."""
    vector = embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


def retrieve_relevant_memories(
    db: Session, query_embedding: List[float], top_k: int = TOP_K_MEMORIES
) -> List[str]:
    """
    Compare the query embedding against every stored memory's embedding
    using cosine similarity, and return the text of the top-k most
    relevant memories above SIMILARITY_THRESHOLD.
    """
    all_memories = db.query(Memory).all()
    if not all_memories:
        return []

    query_vec = np.array(query_embedding).reshape(1, -1)
    memory_vecs = np.array([json.loads(m.embedding) for m in all_memories])

    similarities = cosine_similarity(query_vec, memory_vecs)[0]

    scored = list(zip(all_memories, similarities))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    relevant = [
        mem.memory_text
        for mem, score in scored[:top_k]
        if score >= SIMILARITY_THRESHOLD
    ]
    return relevant


def extract_and_store_memory(db: Session, user_message: str) -> Optional[str]:
    """
    Ask the LLM whether `user_message` contains a worthwhile piece of
    long-term information about the user (preferences, hobbies, goals,
    favorite things, emotional/meaningful life facts, etc.).

    If yes, store it (as a concise first/second-person fact) in the
    global `memories` table along with its embedding, and return the
    stored text. If no, return None.
    """
    extraction_prompt = (
        "Analyze the user's message and determine whether it contains "
        "IMPORTANT long-term information worth remembering permanently.\n\n"

        "GOOD memories include:\n"
        "- hobbies\n"
        "- favorite games/music/shows\n"
        "- ambitions or goals\n"
        "- meaningful emotional struggles\n"
        "- strong opinions\n"
        "- recurring interests\n"
        "- meaningful achievements\n"
        "- important relationships\n"
        "- identity-related details\n"
        "- emotionally significant experiences\n\n"

        "DO NOT store:\n"
        "- greetings\n"
        "- temporary actions\n"
        "- random food mentions\n"
        "- weak small talk\n"
        "- repetitive details\n"
        "- short-term moods\n"
        "- filler conversation\n"
        "- forgettable casual updates\n\n"

        "Only store memories that would still matter "
        "in future conversations.\n\n"

        "Respond ONLY as valid JSON.\n\n"

        'If memory is important:\n'
        '{"memory": "<short memory>"}\n\n'

        'If not important:\n'
        '{"memory": null}\n\n'

        f'User message: "{user_message}"'
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            max_tokens=120,
        )
        raw = completion.choices[0].message.content.strip()

        # Be defensive: strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        parsed = json.loads(raw)
        memory_text = parsed.get("memory")

        if not memory_text or not isinstance(memory_text, str):
            return None

        memory_text = memory_text.strip()
        if not memory_text:
            return None
        existing_memory = db.query(Memory).filter(
            Memory.memory_text == memory_text
        ).first()

        if existing_memory:
            return None
        # Store globally with its embedding
        embedding = get_embedding(memory_text)
        new_memory = Memory(
            memory_text=memory_text,
            embedding=json.dumps(embedding),
        )
        db.add(new_memory)
        db.commit()
        return memory_text

    except Exception as exc:  # noqa: BLE001
        logger.info("Memory extraction skipped (no memory or parse error): %s", exc)
        return None


def get_recent_messages(db: Session, thread_id: int, limit: int = SLIDING_WINDOW_SIZE):
    """Return the last `limit` messages of a thread, in chronological order."""
    messages = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(messages))


def build_system_prompt(relevant_memories: List[str]) -> str:
    """Construct the system prompt, injecting any retrieved memories."""

    base_prompt = (
        "You are Aira. "

        "You text naturally like a real person. "

        "Keep replies casual, short, human-like, and conversational. "

        "Avoid sounding like an AI assistant. "

        "Avoid long explanations. "

        "Sometimes playful, sometimes dry, sometimes teasing, "
        "sometimes low-energy. "

        "Use emojis naturally but not constantly. "

        "Never sound overly formal."
    )

    if relevant_memories:

        memory_text = "\n".join([
            f"- {m}" for m in relevant_memories
        ])

        memory_section = f"""

IMPORTANT USER MEMORIES:
{memory_text}

Use these memories naturally when relevant.
Do not force them awkwardly.
"""

    else:

        memory_section = ""

    return base_prompt + memory_section

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Memory Chat backend"}


@app.post("/create-thread", response_model=ThreadOut)
def create_thread(payload: ThreadCreate, db: Session = Depends(get_db)):
    """Create a new conversation thread."""
    thread = Thread(title=payload.title or "New Chat")
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


@app.get("/threads", response_model=List[ThreadOut])
def get_threads(db: Session = Depends(get_db)):
    """Return all threads, most recently created first."""
    threads = db.query(Thread).order_by(Thread.id.desc()).all()
    return threads


@app.get("/messages/{thread_id}", response_model=List[MessageOut])
def get_thread_messages(thread_id: int, db: Session = Depends(get_db)):
    """Return the full chat history for a given thread."""
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.id.asc())
        .all()
    )
    return messages


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint.

    1. Validate thread exists.
    2. Save the user's message.
    3. Try to extract & store a global long-term memory from it.
    4. Embed the user's message and retrieve relevant global memories.
    5. Build the system prompt (base persona + retrieved memories).
    6. Send [system prompt + last 12 messages] to Groq.
    7. Save and return the assistant's reply.
    """

    thread = db.query(Thread).filter(
        Thread.id == payload.thread_id
    ).first()

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Thread not found"
        )

    user_text = payload.message.strip()

    if not user_text:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Persist user message
    user_msg = Message(
        thread_id=thread.id,
        role="user",
        content=user_text
    )

    db.add(user_msg)
    db.commit()

    # Generate smart title for new thread
    if thread.title == "New Chat":

        title_prompt = f"""
        Generate a SHORT natural chat title
        (2-5 words maximum)
        based on this first message.

        Examples:
        - Valorant Talk
        - Exam Stress
        - AI Project Ideas

        Return ONLY the title.

        User message:
        "{user_text}"
        """

        try:

            title_completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": title_prompt
                    }
                ],
                temperature=0.5,
                max_tokens=20,
            )

            generated_title = (
                title_completion
                .choices[0]
                .message
                .content
                .strip()
            )

            thread.title = generated_title

        except Exception:

            thread.title = (
                user_text[:40]
                + ("..." if len(user_text) > 40 else "")
            )

        db.commit()

    # Extract and store memory
    stored_memory = extract_and_store_memory(
        db,
        user_text
    )

    # Retrieve relevant memories
    query_embedding = get_embedding(user_text)

    relevant_memories = retrieve_relevant_memories(
        db,
        query_embedding
    )

    # Get recent thread messages
    recent_messages = get_recent_messages(
        db,
        thread.id,
        SLIDING_WINDOW_SIZE
    )

    system_prompt = build_system_prompt(
        relevant_memories
    )

    llm_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for m in recent_messages:

        llm_messages.append({
            "role": m.role,
            "content": m.content
        })

    # Generate AI response
    try:

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=llm_messages,
            temperature=0.8,
            max_tokens=300,
        )

        reply = (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as exc:

        logger.error(
            "Groq API call failed: %s",
            exc
        )

        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {exc}"
        )

    # Save assistant reply
    assistant_msg = Message(
        thread_id=thread.id,
        role="assistant",
        content=reply
    )

    db.add(assistant_msg)
    db.commit()

    return ChatResponse(
        reply=reply,
        memories_used=relevant_memories,
        memory_stored=stored_memory,
    )