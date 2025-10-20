from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from langchain.memory import ConversationBufferMemory
from app.services.rag_service import rag_service
import os

app = FastAPI(
    title="Portfolio AI Assistant",
    description="Portfolio assistant using Groq + RAG",
    version="0.0.1"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.amplifyapp.com",
        "https://*.cloudfront.net",
        # "https://your-custom-domain.com"
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

# Initialize RAG on startup
@app.on_event("startup")
async def startup_event():
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"❌ Error initializing RAG: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Portfolio AI Assistant API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_service.qa_chain is not None,
        "active_sessions": len(sessions),
        "platform": "AWS Lambda" if os.getenv('AWS_EXECUTION_ENV') else "Local"
    }

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
        # Get response from RAG chain
        result = qa_chain({
            "question": request.message,
            "chat_history": sessions[session_id].load_memory_variables({})["chat_history"]
        })
        
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
        
        return ChatResponse(
            response=result["answer"],
            session_id=session_id,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing chat: {str(e)}"
        )

@app.post("/api/chat/new-session")
async def new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.delete("/api/chat/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/documents/refresh")
async def refresh_documents():
    """Rebuild the vector store with updated documents"""
    try:
        rag_service.initialize()
        return {"message": "Documents refreshed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error refreshing documents: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)