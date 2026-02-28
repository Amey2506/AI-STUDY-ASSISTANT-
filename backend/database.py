from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./study_assistant.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class UserQuiz(Base):
    __tablename__ = "user_quizzes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True) # Isolation field
    filename = Column(String)
    score = Column(Integer)
    total_questions = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class Flashcard(Base):
    __tablename__ = "flashcards"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True) # Isolation field
    question = Column(String)
    answer = Column(String)
    explanation = Column(String)
    source = Column(String)
    # SM-2 Fields
    interval = Column(Integer, default=0)  # Interval in days
    repetition = Column(Integer, default=0)  # Number of times reviewed
    easiness = Column(Float, default=2.5)  # E-Factor
    next_review = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# SuperMemo-2 Algorithm
def sm2_algorithm(quality: int, interval: int, repetition: int, easiness: float):
    """
    quality: 0-5 (0 = total blackout, 5 = perfect response)
    """
    if quality >= 3:
        if repetition == 0:
            interval = 1
        elif repetition == 1:
            interval = 6
        else:
            interval = round(interval * easiness)
        repetition += 1
    else:
        repetition = 0
        interval = 1

    easiness = easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if easiness < 1.3:
        easiness = 1.3
        
    return interval, repetition, easiness

# DB Utils
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
