from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from langchain.memory import ConversationBufferMemory
from app.services.rag_service import rag_service
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Portfolio AI Assistant",
    description="Portfolio assistant using Groq + RAG",
    version="0.0.1"
)

# CORS - Allow localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
sessions = {}

# Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
    active_sessions: int
    environment: str
    documents_path: str

# Startup event
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("🚀 Starting Portfolio AI Assistant")
    print("="*50)
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"❌ Error initializing RAG: {str(e)}")
        print("\n📝 Quick fix:")
        print(f"1. Make sure documents exist in: {settings.documents_path}")
        print(f"2. Check your .env file has GROQ_API_KEY set")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Portfolio AI Assistant API",
        "version": "1.0.0",
        "environment": settings.environment,
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if rag_service.is_initialized else "initializing",
        rag_initialized=rag_service.is_initialized,
        active_sessions=len(sessions),
        environment=settings.environment,
        documents_path=settings.documents_path
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        qa_chain = rag_service.get_chain()
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG system not initialized: {str(e)}"
        )
    
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create memory for this session
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    try:
        print(f"\n💬 Question: {request.message}")
        
        # Get response from RAG chain
        result = qa_chain({
            "question": request.message,
            "chat_history": sessions[session_id].load_memory_variables({})["chat_history"]
        })
        
        print(f"✅ Answer: {result['answer'][:100]}...")
        
        # Update memory
        sessions[session_id].save_context(
            {"question": request.message},
            {"answer": result["answer"]}
        )
        
        # Extract sources
        sources = []
        if "source_documents" in result:
            sources = list(set([
                doc.metadata.get("source", "Unknown") 
                for doc in result["source_documents"]
            ]))
            print(f"📚 Sources: {sources}")
        
        return ChatResponse(
            response=result["answer"],
            session_id=session_id,
            sources=sources
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing chat: {str(e)}"
        )

@app.post("/api/chat/new-session")
async def new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    print(f"🆕 New session created: {session_id}")
    return {"session_id": session_id}

@app.delete("/api/chat/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in sessions:
        del sessions[session_id]
        print(f"🗑️  Session deleted: {session_id}")
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/documents/refresh")
async def refresh_documents():
    """Rebuild the vector store with updated documents"""
    try:
        print("\n🔄 Refreshing documents...")
        rag_service.initialize()
        return {"message": "Documents refreshed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error refreshing documents: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Starting development server...")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )