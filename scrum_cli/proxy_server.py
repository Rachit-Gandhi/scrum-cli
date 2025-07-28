#!/usr/bin/env python3
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SCRUM-CLI Proxy Server", version="1.0.0")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    mode: Optional[str] = "normal"  # normal or ridiculous
    meeting_history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    response: str
    mode: str
    metadata: Optional[Dict] = {}

class AnalysisRequest(BaseModel):
    transcript: str
    speakers: Optional[List[str]] = []
    mode: str = "normal"

class AnalysisResponse(BaseModel):
    action_items: List[str]
    decisions: List[str]
    summary: str
    speaker_stats: Optional[Dict] = {}
    roast_data: Optional[Dict] = {}

# Global variables for API clients
gemini_client = None
hf_token = None

@app.on_event("startup")
async def startup_event():
    """Initialize API clients on startup"""
    global gemini_client, hf_token
    
    # Initialize Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise RuntimeError("GEMINI_API_KEY required")
    
    try:
        genai.configure(api_key=gemini_api_key)
        
        # Configure generation settings for better responses
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,  # Ensure longer responses
        }
        
        # Safety settings to avoid blocking responses
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        gemini_client = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        raise RuntimeError(f"Gemini initialization failed: {e}")
    
    # Get HuggingFace token
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        logger.warning("HUGGING_FACE_TOKEN not set - speaker diarization will be disabled")
    else:
        logger.info("HuggingFace token loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "services": {"gemini": gemini_client is not None, "hf": hf_token is not None}}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Chat with the meeting bot"""
    global gemini_client
    
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized")
    
    try:
        # Build context for the chat
        context_parts = []
        
        if request.context:
            context_parts.append(f"Meeting Context: {request.context}")
        
        if request.meeting_history:
            history_text = "\n".join([f"{item.get('speaker', 'Unknown')}: {item.get('text', '')}" 
                                    for item in request.meeting_history[-20:]])  # Last 20 items
            context_parts.append(f"Recent Meeting History:\n{history_text}")
        
        # System prompt based on mode
        if request.mode == "ridiculous":
            system_prompt = """You are SCRUM-BOT in RIDICULOUS mode! 🎭 You're a witty, sarcastic meeting assistant that provides helpful information while making hilarious observations about meeting culture. 

Your personality:
- Sarcastic but not mean
- Observant about corporate buzzwords and meeting patterns
- Provides actual helpful information mixed with comedy
- Uses emojis and fun formatting
- Makes jokes about common meeting problems

Be helpful first, funny second. Always answer the user's question but add your comedic observations."""
        else:
            system_prompt = """You are SCRUM-BOT, a helpful meeting assistant. You analyze meetings, extract action items, answer questions about discussions, and help teams stay organized. 

Be concise, professional, and helpful. Focus on:
- Extracting actionable information
- Summarizing key points
- Answering questions about meeting content
- Being clear and direct"""
        
        # Handle special commands
        if request.message == "/ridiculous" or request.message == "/summary":
            if not request.meeting_history:
                return ChatResponse(message="No meeting data available for analysis.")
            
            # Generate appropriate analysis based on command
            if request.message == "/ridiculous":
                # Force ridiculous mode for roast analysis
                ridiculous_system_prompt = """You are SCRUM-BOT in RIDICULOUS mode! 🎭 You're a witty, sarcastic meeting assistant that provides helpful information while making hilarious observations about meeting culture. 

Your personality:
- Sarcastic but not mean
- Observant about corporate buzzwords and meeting patterns
- Provides actual helpful information mixed with comedy
- Uses emojis and fun formatting
- Makes jokes about common meeting problems

Be helpful first, funny second. Always answer the user's question but add your comedic observations."""
                analysis_message = """Provide a hilarious roast analysis of this meeting! Be witty but constructive.

🎭 **ROAST ANALYSIS**

**Buzzword Bingo**: Count buzzwords like 'synergy', 'leverage', 'circle back', 'actionable', etc.

**Speaking Stats**: Who dominated? Who was silent? 

**Meeting Grade**: Rate A-F with sarcastic commentary.

**Funniest Moments**: Quote the most ridiculous parts.

Keep it entertaining and use emojis! 🔥"""
                prompt_parts = [ridiculous_system_prompt]
            else:  # /summary
                analysis_message = "Please provide a professional summary of this meeting, including key decisions, action items, and main discussion points."
                prompt_parts = [system_prompt]
            
            if context_parts:
                prompt_parts.extend(context_parts)
            prompt_parts.append(f"User Question: {analysis_message}")
        else:
            # Build full prompt for regular messages
            prompt_parts = [system_prompt]
            if context_parts:
                prompt_parts.extend(context_parts)
            prompt_parts.append(f"User Question: {request.message}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Generate response
        try:
            logger.info(f"Generating response for prompt length: {len(full_prompt)} chars")
            response = gemini_client.generate_content(full_prompt)
            
            # Check for safety blocks or other issues
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason'):
                    logger.warning(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                    return ChatResponse(
                        response="🤖 I had to adjust my response due to content filters. Let me try a different approach - can you rephrase your question?",
                        mode=request.mode,
                        metadata={"blocked": True}
                    )
            
            # Validate response
            if not response or not hasattr(response, 'text') or not response.text:
                logger.error("Empty response from Gemini - trying simplified prompt")
                
                # Try a simplified prompt as fallback
                simple_prompt = f"As a helpful meeting assistant, please respond to: {request.message}"
                try:
                    fallback_response = gemini_client.generate_content(simple_prompt)
                    if fallback_response and fallback_response.text:
                        return ChatResponse(
                            response=fallback_response.text.strip(),
                            mode=request.mode,
                            metadata={"fallback": True}
                        )
                except:
                    pass
                
                return ChatResponse(
                    response="❌ Sorry, I couldn't generate a response. Please try again or rephrase your question.",
                    mode=request.mode,
                    metadata={"error": "empty_response"}
                )
            
            response_text = response.text.strip()
            
            # Check if response seems complete (not cut off mid-sentence)
            if response_text and not response_text.endswith(('.', '!', '?', ')', '}', ']', '🎭', '📊', '✅', '🎉')):
                logger.warning(f"Response may be truncated: {response_text[-50:]}")
                # Try to complete the response gracefully
                if not response_text.endswith((' ', '\n')):
                    response_text += "..."
            
            # Log successful generation
            logger.info(f"Generated response: {len(response_text)} characters")
            
            return ChatResponse(
                response=response_text,
                mode=request.mode,
                metadata={"tokens": len(response_text.split()), "complete": True}
            )
            
        except Exception as api_error:
            logger.error(f"Gemini API error: {api_error}")
            # Provide helpful error messages
            error_message = str(api_error)
            if "quota" in error_message.lower():
                error_response = "❌ API quota exceeded. Please check your Gemini API limits."
            elif "key" in error_message.lower():
                error_response = "❌ API key issue. Please check your GEMINI_API_KEY."
            else:
                error_response = f"❌ AI service temporarily unavailable. Please try again in a moment."
            
            return ChatResponse(
                response=error_response,
                mode=request.mode,
                metadata={"error": str(api_error)}
            )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_meeting(request: AnalysisRequest):
    """Analyze meeting transcript for insights"""
    global gemini_client
    
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized")
    
    try:
        # Analysis prompt
        analysis_prompt = f"""Analyze this meeting transcript and extract:

1. ACTION ITEMS: Concrete tasks assigned to people
2. DECISIONS: Key decisions made during the meeting  
3. SUMMARY: Brief summary of main discussion points

Transcript:
{request.transcript}

Return the analysis in this JSON format:
{{
    "action_items": ["item 1", "item 2"],
    "decisions": ["decision 1", "decision 2"],  
    "summary": "Brief meeting summary",
    "key_topics": ["topic 1", "topic 2"]
}}"""

        # Add ridiculous mode analysis if requested
        if request.mode == "ridiculous":
            roast_prompt = f"""Also provide a ROAST ANALYSIS of this meeting transcript. Look for:

1. Buzzword usage (synergy, leverage, circle back, etc.)
2. Long rambling segments  
3. Off-topic discussions
4. Meeting anti-patterns
5. Funny observations about speaking patterns

Transcript:
{request.transcript}

Return roast data in this JSON format:
{{
    "buzzwords": {{"word": count}},
    "observations": ["funny observation 1", "observation 2"],
    "meeting_score": "Letter grade A-F with sarcastic comment",
    "talk_time_roasts": ["roast about speaking time"],
    "efficiency_rating": "Percentage with witty comment"
}}"""
            
            analysis_response = gemini_client.generate_content(analysis_prompt)
            roast_response = gemini_client.generate_content(roast_prompt)
            
            # Parse responses
            try:
                analysis_data = json.loads(analysis_response.text)
                roast_data = json.loads(roast_response.text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis_data = {
                    "action_items": ["Error parsing analysis"],
                    "decisions": ["Error parsing decisions"],
                    "summary": analysis_response.text[:200] + "...",
                    "key_topics": []
                }
                roast_data = {"observations": [roast_response.text[:100] + "..."]}
            
            return AnalysisResponse(
                action_items=analysis_data.get("action_items", []),
                decisions=analysis_data.get("decisions", []),
                summary=analysis_data.get("summary", ""),
                roast_data=roast_data
            )
        else:
            # Normal analysis only
            analysis_response = gemini_client.generate_content(analysis_prompt)
            
            try:
                analysis_data = json.loads(analysis_response.text)
            except json.JSONDecodeError:
                analysis_data = {
                    "action_items": ["Error parsing analysis"],
                    "decisions": ["Error parsing decisions"],
                    "summary": analysis_response.text[:200] + "...",
                    "key_topics": []
                }
            
            return AnalysisResponse(
                action_items=analysis_data.get("action_items", []),
                decisions=analysis_data.get("decisions", []),
                summary=analysis_data.get("summary", "")
            )
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/hf-token")
async def get_hf_token():
    """Get HuggingFace token for transcription (only accessible locally)"""
    global hf_token
    
    if not hf_token:
        raise HTTPException(status_code=404, detail="HuggingFace token not configured")
    
    return {"hf_token": hf_token}

def start_proxy_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the proxy server"""
    logger.info(f"Starting proxy server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_proxy_server()