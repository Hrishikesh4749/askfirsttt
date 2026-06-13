# 🧠 AI Memory Chat

A persistent, multi-thread conversational AI with **universal long-term memory**,
built with FastAPI, Streamlit, SQLite, Groq (Llama 3.3 70B), and Sentence
Transformers.

## How it works

- **Threads**: each conversation is its own thread, stored in SQLite.
- **Sliding window**: only the last 12 messages of the *current* thread are
  sent to the model — keeps things fast and cheap.
- **Universal memory**: whenever you mention something durable (a hobby,
  goal, preference, favorite game, meaningful life fact, etc.), the backend
  uses the LLM to extract it and stores it globally — across *all* threads.
- **Semantic retrieval**: every new message is embedded with
  `all-MiniLM-L6-v2`, compared via cosine similarity against all stored
  memories, and the most relevant ones are injected into the system prompt.
- **Result**: the AI can recall things from *other* conversations naturally,
  while staying fast and conversational.

## Project structure

```
ai-memory-chat/
├── backend/
│   ├── main.py          # FastAPI app, chat logic, memory system
│   ├── database.py       # SQLAlchemy engine/session setup
│   ├── models.py          # Thread, Message, Memory tables
│   ├── requirements.txt
│   └── .env               # GROQ_API_KEY goes here
├── frontend/
│   └── app.py             # Streamlit UI
└── requirements.txt
```

## Setup

1. **Create a virtual environment and install dependencies**

```bash
cd ai-memory-chat
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Add your Groq API key**

Edit `backend/.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at https://console.groq.com/keys

3. **Run the backend** (from the `backend/` folder)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

This creates `memory_chat.db` (SQLite) automatically on first run.

4. **Run the frontend** (in a new terminal, from the `frontend/` folder)

```bash
cd frontend
streamlit run app.py
```

5. Open the Streamlit URL printed in the terminal (usually
   http://localhost:8501), click **New Chat**, and start talking.

## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/create-thread` | Create a new chat thread |
| GET | `/threads` | List all threads |
| POST | `/chat` | Send a message, get AI reply (handles memory extraction + retrieval) |
| GET | `/messages/{thread_id}` | Get full history of a thread |

## Notes

- Memories are stored with their embeddings precomputed, so retrieval is a
  single cosine-similarity pass over the `memories` table — no re-embedding
  on every request.
- The system prompt is tuned for short, casual, human-sounding replies
  rather than a formal "assistant" tone.
