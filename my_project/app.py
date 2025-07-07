from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import os
import base64
import tempfile
import asyncio
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Pipeline API",
    description="MOSDAC/ISRO Document RAG Pipeline with Knowledge Graph Generation",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (will be loaded in background)
retriever = None
llm = None
prompt = None
extract_entities = None
build_and_save_kg = None
whisper_model = None
models_loaded = False

# Ensure output directory exists
os.makedirs("output/knowledge_graphs", exist_ok=True)

class QueryResponse(BaseModel):
    query: str
    context: str
    answer: str
    entities: List[Tuple[str, str]]
    kg_image: str
    kg_image_url: str  # URL to access the image

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

def load_models():
    """Load models in a background thread"""
    global retriever, llm, prompt, extract_entities, build_and_save_kg, whisper_model, models_loaded, process_query

    logger.info("Loading RAG pipeline components in background...")

    try:
        from my_project.rag import (
            retriever as _retriever,
            llm as _llm,
            prompt as _prompt,
            extract_entities as _extract_entities,
            build_and_save_kg as _build_and_save_kg,
            whisper_model as _whisper_model,
            process_query as _process_query,
        )

        retriever = _retriever
        llm = _llm
        prompt = _prompt
        extract_entities = _extract_entities
        build_and_save_kg = _build_and_save_kg
        whisper_model = _whisper_model
        process_query = _process_query
        models_loaded = True

        logger.info("RAG pipeline components loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load RAG components: {str(e)}")
        import traceback
        traceback.print_exc()  # <--- Add this line to print the full stack trace
        models_loaded = False

# Start loading models in background thread
loading_thread = threading.Thread(target=load_models, daemon=True)
loading_thread.start()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Pipeline API is running!",
        "version": "1.0.0",
        "models_loaded": models_loaded,
        "endpoints": {
            "POST /rag": "Main RAG pipeline endpoint",
            "POST /rag/voice-upload": "Upload audio file for voice queries",
            "GET /kg-image/{filename}": "Get knowledge graph images",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if not models_loaded:
            return {
                "status": "loading",
                "message": "RAG components are still loading..."
            }
        
        # Test if models are loaded
        test_query = "test"
        docs = retriever.invoke(test_query)
        return {
            "status": "healthy",
            "models_loaded": True,
            "documents_loaded": len(docs) > 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/rag", response_model=QueryResponse)
async def rag(
    mode: str = Form(..., description="text or voice"),
    query: Optional[str] = Form(None, description="Query text (if mode=text)"),
    duration: Optional[int] = Form(5, description="Recording duration in seconds (if mode=voice)")
):
    """
    Run full RAG pipeline.
    If mode=text => use the provided query.
    If mode=voice => record audio on server, transcribe it to query.
    """
    
    # Check if models are loaded
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is still loading. Please wait a moment and try again."
        )
    
    try:
        # Validate mode
        if mode not in {"text", "voice"}:
            raise HTTPException(
                status_code=400, 
                detail="mode must be 'text' or 'voice'"
            )
        
        # Handle voice input
        if mode == "voice":
            logger.info(f"Recording audio for {duration} seconds...")
            
            # Record audio with error handling
            try:
                record_audio(filename="input.wav", duration=duration)
                result = whisper_model.transcribe("input.wav")
                query = result['text'].strip()
                logger.info(f"Transcribed query: {query}")
            except Exception as e:
                logger.error(f"Voice recording/transcription failed: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Voice processing failed: {str(e)}"
                )
        else:
            if not query or not query.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="Query text is required for mode=text"
                )
            query = query.strip()

        if not query:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract query from input"
            )

        logger.info(f"Processing query: {query}")

        # Use process_query for clarity
        try:
            context, answer, entities = process_query(query)
            logger.info("Generated answer successfully")
        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"RAG processing failed: {str(e)}"
            )

        # Extract entities and build knowledge graph
        try:
            safe_query = "".join(c if c.isalnum() else "_" for c in query[:20])
            kg_filename = f"knowledge_graph_{safe_query}.png"
            kg_filepath = f"output/knowledge_graphs/{kg_filename}"
            build_and_save_kg(entities, filename=kg_filepath)
            kg_image_url = f"/kg-image/{kg_filename}"
            logger.info(f"Knowledge graph created: {kg_filepath}")
        except Exception as e:
            logger.error(f"Knowledge graph generation failed: {str(e)}")
            entities = []
            kg_filename = "error_generating_kg.png"
            kg_image_url = ""

        return QueryResponse(
            query=query,
            context=context,
            answer=answer,
            entities=entities,
            kg_image=kg_filename,
            kg_image_url=kg_image_url
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/rag/voice-upload", response_model=QueryResponse)
async def rag_voice_upload(
    audio: UploadFile = File(..., description="Audio file for transcription")
):
    """
    Upload audio file for voice query processing.
    Supports common audio formats (wav, mp3, m4a, etc.)
    """
    
    # Check if models are loaded
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is still loading. Please wait a moment and try again."
        )
    
    try:
        # Validate file type
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an audio file"
            )
        
        logger.info(f"Processing uploaded audio: {audio.filename}")
        
        # Read uploaded file
        audio_bytes = await audio.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        try:
            # Transcribe audio
            result = whisper_model.transcribe(temp_filename)
            query = result['text'].strip()
            
            logger.info(f"Transcribed query from upload: {query}")
            
            if not query:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not extract text from audio"
                )
            
            # Process the query through RAG pipeline
            docs = retriever.invoke(query)
            context = "\n".join(doc.page_content for doc in docs)
            
            final_prompt = prompt.format(context=context, query=query)
            response = llm.invoke(final_prompt)
            answer = response.content.strip()
            
            # Extract entities and build knowledge graph
            entities = extract_entities(answer)
            
            safe_query = "".join(c if c.isalnum() else "_" for c in query[:20])
            kg_filename = f"knowledge_graph_{safe_query}.png"
            kg_filepath = f"output/knowledge_graphs/{kg_filename}"
            
            build_and_save_kg(entities, filename=kg_filepath)
            kg_image_url = f"/kg-image/{kg_filename}"
            
            return QueryResponse(
                query=query,
                context=context,
                answer=answer,
                entities=entities,
                kg_image=kg_filename,
                kg_image_url=kg_image_url
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice upload processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Voice upload processing failed: {str(e)}"
        )

@app.get("/kg-image/{filename}")
async def get_kg_image(filename: str):
    """
    Serve knowledge graph images.
    """
    
    try:
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        filepath = f"output/knowledge_graphs/{safe_filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=404, 
                detail="Knowledge graph image not found"
            )
        
        return FileResponse(
            filepath,
            media_type="image/png",
            filename=safe_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving KG image: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error serving image"
        )

@app.get("/kg-image/{filename}/base64")
async def get_kg_image_base64(filename: str):
    """
    Get knowledge graph image as base64 string.
    Useful for embedding in JSON responses.
    """
    
    try:
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        filepath = f"output/knowledge_graphs/{safe_filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=404, 
                detail="Knowledge graph image not found"
            )
        
        # Read and encode image
        with open(filepath, "rb") as image_file:
            image_bytes = image_file.read()
            base64_string = base64.b64encode(image_bytes).decode()
        
        return {
            "filename": safe_filename,
            "base64": base64_string,
            "mime_type": "image/png"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting KG image to base64: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error processing image"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)