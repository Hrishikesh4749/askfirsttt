"""
models.py
----------
SQLAlchemy ORM models for the AI Memory Chat application.

Tables:
- threads:  individual conversation threads
- messages: chat messages belonging to a thread
- memories: long-term, GLOBAL memories shared across all threads,
            each stored with a precomputed sentence-embedding for
            semantic similarity search.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from database import Base


class Thread(Base):
    """A single conversation thread (like a 'chat tab')."""

    __tablename__ = "threads"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    """A single message (user or assistant) inside a thread."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    thread = relationship("Thread", back_populates="messages")


class Memory(Base):
    """
    A piece of long-term, GLOBAL knowledge about the user
    (preferences, goals, hobbies, facts, etc.), shared across
    every thread.

    `embedding` stores the sentence-transformer embedding for this
    memory, serialized as a JSON string (list of floats), so we can
    compute cosine similarity against new user messages without
    re-encoding every memory on every request.
    """

    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    memory_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON-encoded list[float]
    created_at = Column(DateTime, default=datetime.utcnow)
