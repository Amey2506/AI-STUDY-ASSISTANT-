import os
from typing import Optional
import io
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from pypdf import PdfReader
from sqlalchemy.orm import Session
import logging
from passlib.context import CryptContext
from jose import JWTError, jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Local imports
from database import SessionLocal, User, UserQuiz, Flashcard, get_db, sm2_algorithm

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Study Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class QuizScore(BaseModel):
    filename: str
    score: int
    total_questions: int

class FlashcardReview(BaseModel):
    card_id: int
    quality: int # 0-5

class ChatRequest(BaseModel):
    message: str
    filename: Optional[str] = None

# Auth Setup
SECRET_KEY = "super_secret_study_key" # In production, this should be in .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user.username # Return username as the unique identifier for isolation

# AI Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# FAISS index storage base path
FAISS_BASE_PATH = "./faiss_index"

# FAISS index storage base path
FAISS_BASE_PATH = "./faiss_index"

# Initialize OCR Reader
ocr_reader = None
try:
    import easyocr
    logger.info("Initializing EasyOCR Model for scanned PDFs...")
    # Load into memory once
    ocr_reader = easyocr.Reader(['en'], gpu=False) # Start with CPU to avoid immediate OOM or complex CUDA setup on user machine
except ImportError:
    logger.warning("EasyOCR not installed. OCR fallback will be disabled.")


def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        # OCR Fallback for scanned/image-based PDFs
        if len(text.strip()) < 50 and ocr_reader is not None:
            logger.info("PDF extraction yielded little text. Attempting OCR fallback...")
            try:
                import fitz # PyMuPDF
                # Reset stream position to read bytes again for fitz
                pdf_file.seek(0)
                pdf_bytes = pdf_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                ocr_text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Render page to an image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    
                    # Run EasyOCR
                    results = ocr_reader.readtext(img_bytes, detail=0)
                    if results:
                        ocr_text += " ".join(results) + "\n\n"
                        
                if len(ocr_text.strip()) > 50:
                    text = ocr_text
                    logger.info("OCR successfully extracted text from the document.")
            except Exception as ocr_e:
                logger.error(f"OCR fallback failed: {ocr_e}")
                
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

@app.get("/")
def read_root():
    return {"status": "online", "engine": "Groq + Local RAG + SQLite + Auth"}

class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer", "username": new_user.username}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user)
):
    print(f"DEBUG: Processing upload for user {username}, file {file.filename}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        content = await file.read()
        extracted_text = extract_text_from_pdf(io.BytesIO(content))
        print(f"DEBUG: Extracted {len(extracted_text)} characters from PDF")
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The PDF appears to be empty or unscannable.")

        # 2. Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(extracted_text)
        print(f"DEBUG: Split into {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not create text chunks from the PDF.")

        # Vector storage
        user_index_path = os.path.join(FAISS_BASE_PATH, username)
        logger.info(f"Using vector storage at {user_index_path}")
        
        if os.path.exists(user_index_path):
            vdb = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
            vdb.add_texts(texts=chunks, metadatas=[{"source": file.filename} for _ in chunks])
        else:
            vdb = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": file.filename} for _ in chunks])
        
        vdb.save_local(user_index_path)
        
        # RAG Retrieval - simplified
        query = "Generate key study questions and flashcards from this text."
        try:
            # Simple similarity search without filters to avoid compatibility issues
            retrieved_docs = vdb.query(query, k=5) if hasattr(vdb, 'query') else vdb.similarity_search(query, k=5)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            retrieved_docs = []
            
        logger.info(f"Retrieved {len(retrieved_docs)} docs")
        
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        if not context_text.strip():
            context_text = "\n\n".join(chunks[:3])

        logger.info("Calling Groq for quiz generation...")
        system_prompt = "You are a specialized study assistant. Respond ONLY with a JSON object."
        human_prompt = 'Create a 5-question MCQ quiz based on the context. Your entire response must be a single JSON object with this exact structure: {{"questions": [{{"question": "text", "options": ["A", "B", "C", "D"], "answer": "correct_option", "explanation": "why"}}]}} \n\n Context: {context}'
        
        chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        formatted_prompt = chat_prompt.format_messages(context=context_text)
        response = llm.invoke(formatted_prompt)
        
        raw_content = response.content
        logger.info(f"Raw LLM response length: {len(raw_content)}")
        
        # Log to file for deep inspection
        with open("last_llm_response.json", "w", encoding="utf-8") as f:
            f.write(raw_content)

        # Robust JSON extraction
        try:
            import re
            # Extract everything between the first { and the last }
            match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
            if match:
                json_str = match.group(1)
                # Aggressive cleaning
                json_str = re.sub(r'//.*', '', json_str)
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                quiz_data = json.loads(json_str)
            else:
                raise ValueError("JSON not found in response")
        except Exception as e:
            logger.error(f"Parsing failed: {e}. Raw content: {raw_content[:400]}")
            # Fallback to a valid dummy quiz to prevent the UI from being stuck
            quiz_data = {
                "questions": [
                    {
                        "question": "The AI had trouble formatting the quiz, but RAG is working! What is the primary focus of your notes?",
                        "options": ["Quantum Computing", "Biology", "History", "General Study"],
                        "answer": "General Study",
                        "explanation": "This is a fallback quiz because the LLM returned malformed JSON."
                    }
                ]
            }

        # Create flashcards
        if "questions" in quiz_data:
            for q in quiz_data["questions"][:3]:
                card = Flashcard(
                    user_id=username,
                    question=q.get("question", "N/A"),
                    answer=q.get("answer", "N/A"),
                    explanation=q.get("explanation", "N/A"),
                    source=file.filename
                )
                db.add(card)
            db.commit()
            print(f"DEBUG: Saved {len(quiz_data['questions'])} questions and created flashcards")
        
        return {"filename": file.filename, "quiz": quiz_data}
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-score")
async def submit_score(data: QuizScore, db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    quiz_record = UserQuiz(user_id=username, filename=data.filename, score=data.score, total_questions=data.total_questions)
    db.add(quiz_record)
    db.commit()
    return {"status": "saved"}

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    quizzes = db.query(UserQuiz).filter(UserQuiz.user_id == username).order_by(UserQuiz.created_at.desc()).limit(10).all()
    total_cards = db.query(Flashcard).filter(Flashcard.user_id == username).count()
    due_today = db.query(Flashcard).filter(Flashcard.user_id == username, Flashcard.next_review <= datetime.utcnow()).count()
    
    return {
        "recent_quizzes": quizzes,
        "total_flashcards": total_cards,
        "due_today": due_today
    }

@app.get("/review")
async def get_review_cards(db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    cards = db.query(Flashcard).filter(Flashcard.user_id == username, Flashcard.next_review <= datetime.utcnow()).all()
    return {"cards": cards}

@app.post("/review/submit")
async def submit_review(data: FlashcardReview, db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    card = db.query(Flashcard).filter(Flashcard.id == data.card_id, Flashcard.user_id == username).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    
    interval, repetition, easiness = sm2_algorithm(
        data.quality, card.interval, card.repetition, card.easiness
    )
    
    card.interval = interval
    card.repetition = repetition
    card.easiness = easiness
    card.next_review = datetime.utcnow() + timedelta(days=interval)
    
    db.commit()
    return {"next_review": card.next_review}

@app.post("/chat")
async def chat_with_notes(data: ChatRequest, username: str = Depends(get_current_user)):
    print(f"DEBUG: Chat request from {username} for file {data.filename}")
    try:
        user_index_path = os.path.join(FAISS_BASE_PATH, username)
        if not os.path.exists(user_index_path):
            return {"response": "You haven't uploaded any notes yet. Please upload a PDF to start learning!", "context_used": []}
        
        vdb = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
        # Removing filter to avoid FAISS KeyError: 'vector' as isolation is per user_id anyway
        retrieved_docs = vdb.query(data.message, k=4) if hasattr(vdb, 'query') else vdb.similarity_search(data.message, k=4)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Prompt for generation
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful study assistant. Answer the user's question based ONLY on the provided context. If the answer isn't in the context, say 'I couldn't find that in your notes, but based on general knowledge...'"),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        formatted_prompt = chat_prompt.format_messages(context=context_text, question=data.message)
        response = llm.invoke(formatted_prompt)
        return {"response": response.content, "context_used": [doc.metadata for doc in retrieved_docs]}
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
