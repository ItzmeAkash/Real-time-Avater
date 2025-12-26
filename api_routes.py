import os
import json
import glob
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv
import logging
from datetime import datetime
from collections import defaultdict

load_dotenv()

logger = logging.getLogger("api")
router = APIRouter()

# Get the base directory (where the script is located)
BASE_DIR = Path(__file__).parent.absolute()

# In-memory storage for real-time session transcripts
# Format: {room_name: [{"role": "...", "message": "...", "timestamp": "..."}, ...]}
active_session_transcripts: Dict[str, List[Dict[str, str]]] = defaultdict(list)


# Request models for POST requests
class TokenRequest(BaseModel):
    identity: Optional[str] = None
    name: Optional[str] = None
    room: Optional[str] = None


class TranscriptEntry(BaseModel):
    role: str
    message: str
    timestamp: Optional[str] = None


class EvaluationRequest(BaseModel):
    transcript: List[TranscriptEntry]


class ConversationEvaluationRequest(BaseModel):
    """Request model for conversation evaluation endpoint"""

    conversation: Optional[List[Dict[str, str]]] = None
    transcript: Optional[List[TranscriptEntry]] = None
    # Allow raw text format as well
    messages: Optional[List[Dict[str, str]]] = None


class RawConversationRequest(BaseModel):
    """Request model for raw conversation text evaluation"""

    conversation_text: str  # Raw conversation text like from transcript file


def _generate_token(identity: str, name: str, room: str):
    """Helper function to generate LiveKit token"""
    # Get API credentials from environment
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500, detail="LiveKit API credentials not configured"
        )

    # Import livekit api
    try:
        from livekit import api
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LiveKit package not installed. Please install livekit-api package.",
        )

    # Generate token
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )

    return {
        "token": token.to_jwt(),
        "url": os.getenv("LIVEKIT_URL", ""),
        "room": room,
        "identity": identity,
        "name": name,
    }


async def evaluate_conversation(transcript: List[Dict[str, str]]) -> str:
    """Evaluate the conversation using LLM based on the rubric"""

    # Build transcript text
    transcript_text = "\n".join([
        f"{entry['role']}: {entry['message']}" for entry in transcript
    ])

    rubric_prompt = """You are a certified Banking Sales Competency Evaluator. Your task is to evaluate a salesperson's performance in a role-play conversation with a customer applying for a personal loan. 

You must analyze ONLY the salesperson's messages—not the customer's.

Your scoring must follow the rubric exactly and align each behaviour to the appropriate competency level.

====================================================
RUBRIC - 11 COMPETENCIES
====================================================

1. Rapport Building & First Impression
Level 1: Abrupt, no warmth, ignores customer mood
Level 2: Polite but scripted, inconsistent warmth
Level 3: Warm, professional, adapts to emotional state
Level 4: Immediate trust, reduces tension quickly

2. Active Listening
Level 1: Misses details, interrupts, partial responses
Level 2: Listens but overlooks details, inaccurate paraphrasing
Level 3: Clear understanding, reflects accurately, addresses concerns
Level 4: Picks up spoken/unspoken concerns, validates effectively

3. Empathy & Emotional Intelligence
Level 1: Defensive, dismisses feelings, escalates frustration
Level 2: Generic empathy, doesn't reduce tension
Level 3: Identifies emotional state, specific empathy, helps calm
Level 4: Deep understanding, reduces stress, guides to rational discussion

4. Probing & Needs Exploration
Level 1: Few/irrelevant questions, misses core aspects
Level 2: Basic questions, doesn't explore deeper
Level 3: Structured questions, explores key areas
Level 4: Uncovers spoken/unspoken needs, identifies underlying concerns

5. Objection Handling
Level 1: Defensive, vague/incorrect info, increases doubt
Level 2: Answers but no structure, fails to reassure
Level 3: Clear structure (acknowledge, clarify, resolve), correct info
Level 4: Turns objections into trust-building, maintains credibility

6. Communication Clarity
Level 1: Confusing, overly technical, overwhelming
Level 2: Mostly understandable but unclear at times
Level 3: Simple language, clear structure, checks understanding
Level 4: Exceptional clarity, simple analogies, customer feels confident

7. Product Knowledge & Solution Fit
Level 1: Incorrect/incomplete info, unsure about features
Level 2: Basic features, generic info, standard statements
Level 3: Correct essential details, tailored explanation, highlights benefits
Level 4: Deep mastery, nuanced explanations, perfect fit demonstration

8. Sales Structuring & Pitch Quality
Level 1: Scattered, random order, pushy
Level 2: Weak structure, too much/little info
Level 3: Clear structure (need, features, benefits, value), confident
Level 4: Polished, conversational, weaves value seamlessly

9. Closing Skills
Level 1: Avoids closing, abrupt ending
Level 2: Attempts but lacks confidence/structure
Level 3: Clear summary, gentle trial closes, defines next step
Level 4: Natural close, reinforces value, customer feels confident

10. Ownership & Professionalism
Level 1: Defensive, casual, overpromises, loses composure
Level 2: Basic professionalism, inconsistent, partial ownership
Level 3: Calm, professional, takes responsibility, dependable
Level 4: Strong composure under pressure, embodies professionalism

11. Customer-First Mindset
Level 1: Focuses on sale, overlooks concerns, transactional
Level 2: Customer care inconsistent, misses opportunities
Level 3: Prioritizes understanding, encourages informed decisions
Level 4: Puts customer wellbeing above sale, builds trust through honesty

====================================================
WEIGHTING LOGIC
====================================================
- Behavioural Competencies (1-6) = 40%
- Technical Competencies (7-9) = 40%
- Attitude & Mindset (10-11) = 20%

====================================================
SCORING RULES
====================================================
1. Score observable behaviour only
2. Politeness ≠ Empathy
3. Volume ≠ Clarity
4. Emotion shift matters (Level 4 = customer's intensity reduces)
5. Probing must be meaningful
6. Objection handling: Acknowledge → Clarify → Resolve
7. Product knowledge must be factually correct
8. Closing must be natural and guided

====================================================
OUTPUT FORMAT (MANDATORY)
====================================================

### 1. Competency Scores
For each of the 11 competencies:
- Competency Name
- Level Assigned (1-4)
- 2-3 sentence justification with evidence

### 2. Weighted Final Score (0-100)
Show calculation:
- Behavioural subtotal (6 items)
- Technical subtotal (3 items)
- Attitude subtotal (2 items)
- Final weighted total

### 3. Strengths Summary
3-5 clear strengths in coaching language

### 4. Development Areas
3-5 improvement recommendations linked to behaviours

### 5. Actionable Coaching Advice (Short)
2-3 habits or practice tips

====================================================
CONVERSATION TRANSCRIPT
====================================================

{transcript}

====================================================
EVALUATION
====================================================

Evaluate the salesperson's performance now:"""

    evaluation_prompt = rubric_prompt.format(transcript=transcript_text)

    # Use Groq LLM for evaluation with langchain_groq (better rate limiting handling)
    try:
        import asyncio
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

        # Initialize ChatGroq with retry and rate limiting handling
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",  # Using a more stable model, or use "mixtral-8x7b-32768" or "llama-3.1-70b-versatile"
            temperature=0.3,
            max_retries=3,  # Automatic retries for rate limits
            timeout=120.0,
        )

        # Run the synchronous invoke in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: llm.invoke([HumanMessage(content=evaluation_prompt)])
        )
        return response.content

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        error_msg = str(e)
        # Handle rate limit errors more gracefully
        if "429" in error_msg or "rate limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment and try again.",
            )
        raise HTTPException(
            status_code=500, detail=f"Error generating evaluation: {error_msg}"
        )


@router.get("/getToken")
def getToken_get(
    identity: Optional[str] = Query(default="user", description="User identity"),
    name: Optional[str] = Query(default="User", description="User name"),
    room: Optional[str] = Query(default="my-room", description="Room name"),
):
    """Get LiveKit token via GET request"""
    try:
        return _generate_token(identity, name, room)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/getToken")
def getToken_post(request_body: Optional[TokenRequest] = None):
    """Get LiveKit token via POST request"""
    try:
        # Get parameters from JSON body or use defaults
        identity = (
            request_body.identity
            if request_body and request_body.identity is not None
            else "user"
        )
        name = (
            request_body.name
            if request_body and request_body.name is not None
            else "User"
        )
        room = (
            request_body.room
            if request_body and request_body.room is not None
            else "my-room"
        )

        return _generate_token(identity, name, room)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate(request_body: EvaluationRequest):
    """Evaluate a conversation transcript and return evaluation report"""
    try:
        # Convert Pydantic models to dict format
        transcript_dict = [
            {
                "role": entry.role,
                "message": entry.message,
                "timestamp": entry.timestamp or "",
            }
            for entry in request_body.transcript
        ]

        if not transcript_dict:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")

        # Generate evaluation
        evaluation_report = await evaluate_conversation(transcript_dict)

        return {
            "status": "success",
            "evaluation": evaluation_report,
            "transcript_count": len(transcript_dict),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcript/{room_name}")
def get_transcript(room_name: str):
    """Get transcript for a specific room"""
    try:
        # Search for evaluation JSON files matching the room name (they contain transcripts)
        pattern = str(BASE_DIR / f"evaluation_{room_name}_*.json")
        matching_files = glob.glob(pattern)

        if not matching_files:
            # Also check for transcript files
            transcript_pattern = str(BASE_DIR / "conversation_transcript_*.txt")
            all_transcript_files = glob.glob(transcript_pattern)

            # Try to find transcript files that might be related to this room
            # Since we don't have room name in transcript files, return the most recent one
            if all_transcript_files:
                latest_transcript = max(all_transcript_files, key=os.path.getctime)
                with open(latest_transcript, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Parse the transcript content
                    transcript = []
                    for line in content.split("\n"):
                        if line.strip() and not line.startswith("==="):
                            # Parse format: [timestamp] Role: message
                            if "] " in line:
                                parts = line.split("] ", 1)
                                if len(parts) == 2:
                                    role_part = parts[1].split(": ", 1)
                                    if len(role_part) == 2:
                                        transcript.append({
                                            "role": role_part[0],
                                            "message": role_part[1],
                                            "timestamp": parts[0].replace("[", ""),
                                        })
                    return {
                        "status": "success",
                        "room": room_name,
                        "transcript": transcript,
                        "source": "transcript_file",
                    }

            raise HTTPException(
                status_code=404, detail=f"No transcript found for room: {room_name}"
            )

        # Get the most recent evaluation file (which contains transcript)
        latest_file = max(matching_files, key=os.path.getctime)

        with open(latest_file, "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)

        return {
            "status": "success",
            "room": room_name,
            "transcript": evaluation_data.get("transcript", []),
            "timestamp": evaluation_data.get("timestamp", ""),
            "source": "evaluation_file",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/{room_name}")
def get_evaluation(room_name: str):
    """Get evaluation result for a specific room"""
    try:
        # Search for evaluation JSON files matching the room name using absolute path
        pattern = str(BASE_DIR / f"evaluation_{room_name}_*.json")
        matching_files = glob.glob(pattern)

        if not matching_files:
            raise HTTPException(
                status_code=404, detail=f"No evaluation found for room: {room_name}"
            )

        # Get the most recent evaluation file
        latest_file = max(matching_files, key=os.path.getctime)

        with open(latest_file, "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)

        return {
            "status": "success",
            "room": room_name,
            "evaluation": evaluation_data.get("evaluation", ""),
            "transcript": evaluation_data.get("transcript", []),
            "timestamp": evaluation_data.get("timestamp", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/{room_name}")
async def evaluate_room(room_name: str):
    """Get transcript for a room and evaluate it"""
    try:
        # First, get the transcript for the room
        transcript_response = get_transcript(room_name)

        if transcript_response["status"] != "success":
            raise HTTPException(
                status_code=404,
                detail=f"Could not retrieve transcript for room: {room_name}",
            )

        transcript = transcript_response.get("transcript", [])

        if not transcript:
            raise HTTPException(
                status_code=400, detail=f"Transcript is empty for room: {room_name}"
            )

        # Convert to dict format if needed
        transcript_dict = [
            {
                "role": entry.get("role", ""),
                "message": entry.get("message", ""),
                "timestamp": entry.get("timestamp", ""),
            }
            for entry in transcript
        ]

        # Generate evaluation
        evaluation_report = await evaluate_conversation(transcript_dict)

        # Save the evaluation result
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_data = {
            "room": room_name,
            "timestamp": timestamp,
            "evaluation": evaluation_report,
            "transcript": transcript,
        }
        evaluation_json_file = BASE_DIR / f"evaluation_{room_name}_{timestamp}.json"
        with open(evaluation_json_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation saved to: {evaluation_json_file}")

        return {
            "status": "success",
            "room": room_name,
            "evaluation": evaluation_report,
            "transcript_count": len(transcript_dict),
            "timestamp": timestamp,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating room: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_conversation_text(conversation_text: str) -> List[Dict[str, str]]:
    """
    Parse raw conversation text (like from transcript file) into structured format.
    Handles format: [timestamp] Role: message
    """
    transcript = []
    for line in conversation_text.split("\n"):
        line = line.strip()
        # Skip empty lines and header lines
        if not line or line.startswith("==="):
            continue
        # Parse format: [timestamp] Role: message
        if "] " in line and ": " in line:
            try:
                # Split timestamp and rest
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    timestamp = parts[0].replace("[", "")
                    role_message = parts[1]
                    # Split role and message
                    if ": " in role_message:
                        role_parts = role_message.split(": ", 1)
                        if len(role_parts) == 2:
                            role = role_parts[0].strip()
                            message = role_parts[1].strip()
                            if role and message:
                                transcript.append({
                                    "role": role,
                                    "message": message,
                                    "timestamp": timestamp,
                                })
            except Exception as e:
                logger.warning(f"Error parsing line: {line} - {e}")
                continue
    return transcript


@router.post("/evaluate-raw-conversation")
async def evaluate_raw_conversation(request_body: RawConversationRequest):
    """
    Evaluate a raw conversation text passed in the payload.
    Accepts the conversation text directly (like from transcript file) and parses it automatically.

    Example payload:
    {
        "conversation_text": "[2025-12-24T20:00:19] Salesperson: Hi. This is Silal. How may I help you?\n[2025-12-24T20:00:30] Customer: Look, I've already wasted 10 days..."
    }
    """
    try:
        if (
            not request_body.conversation_text
            or not request_body.conversation_text.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="conversation_text cannot be empty. Please provide the conversation text.",
            )

        logger.info(f"Received raw conversation text evaluation request")
        logger.debug(
            f"Conversation text length: {len(request_body.conversation_text)} characters"
        )

        # Parse the raw conversation text
        transcript_dict = parse_conversation_text(request_body.conversation_text)

        if not transcript_dict:
            raise HTTPException(
                status_code=400,
                detail="Could not parse conversation text. Please ensure the format is: [timestamp] Role: message",
            )

        logger.info(f"Parsed {len(transcript_dict)} messages from conversation text")

        # Generate evaluation
        evaluation_report = await evaluate_conversation(transcript_dict)

        return {
            "status": "success",
            "evaluation": evaluation_report,
            "transcript": transcript_dict,
            "transcript_count": len(transcript_dict),
            "message": "Evaluation completed successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate-raw-conversation endpoint: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error evaluating conversation: {str(e)}"
        )


@router.post("/evaluate-conversation")
async def evaluate_conversation_endpoint(request_body: ConversationEvaluationRequest):
    """
    Evaluate a conversation transcript passed in the request payload.
    Accepts multiple input formats:
    - conversation: List of dicts with 'role' and 'message' keys
    - transcript: List of TranscriptEntry objects
    - messages: List of dicts with 'role' and 'message' keys (alias for conversation)
    """
    try:
        transcript_dict = []

        # Try to get transcript from different possible fields
        if request_body.conversation:
            # Format: [{"role": "Salesperson", "message": "Hello"}, ...]
            transcript_dict = [
                {
                    "role": entry.get("role", ""),
                    "message": entry.get("message", ""),
                    "timestamp": entry.get("timestamp", ""),
                }
                for entry in request_body.conversation
            ]
        elif request_body.messages:
            # Format: [{"role": "Salesperson", "message": "Hello"}, ...]
            transcript_dict = [
                {
                    "role": entry.get("role", ""),
                    "message": entry.get("message", ""),
                    "timestamp": entry.get("timestamp", ""),
                }
                for entry in request_body.messages
            ]
        elif request_body.transcript:
            # Format: List of TranscriptEntry objects
            transcript_dict = [
                {
                    "role": entry.role,
                    "message": entry.message,
                    "timestamp": entry.timestamp or "",
                }
                for entry in request_body.transcript
            ]
        else:
            raise HTTPException(
                status_code=400,
                detail="No conversation data provided. Please provide 'conversation', 'messages', or 'transcript' field.",
            )

        if not transcript_dict:
            raise HTTPException(
                status_code=400,
                detail="Transcript cannot be empty. Please provide at least one conversation entry.",
            )

        logger.info(
            f"Received conversation evaluation request with {len(transcript_dict)} messages"
        )

        # Generate evaluation
        evaluation_report = await evaluate_conversation(transcript_dict)

        return {
            "status": "success",
            "evaluation": evaluation_report,
            "transcript": transcript_dict,
            "transcript_count": len(transcript_dict),
            "message": "Evaluation completed successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate-conversation endpoint: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error evaluating conversation: {str(e)}"
        )


@router.get("/process-evaluation")
async def process_evaluation():
    """
    Get the most recent transcript.txt file, parse it, generate evaluation, return result, and clean up old files.
    No room name needed - just uses the most recent conversation_transcript_*.txt file.
    """
    try:
        # Search for all transcript text files
        transcript_pattern = str(BASE_DIR / "conversation_transcript_*.txt")
        logger.info(f"Looking for transcript files in: {BASE_DIR}")
        logger.info(f"Search pattern: {transcript_pattern}")
        matching_files = glob.glob(transcript_pattern)
        logger.info(f"Found {len(matching_files)} matching transcript files")

        if not matching_files:
            # Log directory contents for debugging
            try:
                all_files = list(BASE_DIR.glob("*"))
                logger.warning(f"BASE_DIR contents: {[f.name for f in all_files]}")
            except Exception as e:
                logger.warning(f"Could not list BASE_DIR contents: {e}")
            raise HTTPException(
                status_code=404,
                detail="No transcript file found. Please wait for the conversation to complete.",
            )

        # Get the most recent transcript file
        latest_file = max(matching_files, key=os.path.getctime)
        logger.info(f"Processing transcript file: {latest_file}")

        # Parse transcript from text file
        transcript = []
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()
            for line in content.split("\n"):
                line = line.strip()
                # Skip empty lines and header lines
                if not line or line.startswith("==="):
                    continue
                # Parse format: [timestamp] Role: message
                if "] " in line and ": " in line:
                    try:
                        # Split timestamp and rest
                        parts = line.split("] ", 1)
                        if len(parts) == 2:
                            timestamp = parts[0].replace("[", "")
                            role_message = parts[1]
                            # Split role and message
                            if ": " in role_message:
                                role_parts = role_message.split(": ", 1)
                                if len(role_parts) == 2:
                                    role = role_parts[0].strip()
                                    message = role_parts[1].strip()
                                    if role and message:
                                        transcript.append({
                                            "role": role,
                                            "message": message,
                                            "timestamp": timestamp,
                                        })
                    except Exception as e:
                        logger.warning(f"Error parsing line: {line} - {e}")
                        continue

        if not transcript:
            raise HTTPException(
                status_code=400, detail="Transcript is empty or could not be parsed."
            )

        logger.info(f"Parsed {len(transcript)} messages from transcript file")

        # Convert to dict format for evaluation
        transcript_dict = [
            {
                "role": entry.get("role", ""),
                "message": entry.get("message", ""),
                "timestamp": entry.get("timestamp", ""),
            }
            for entry in transcript
        ]

        # Generate evaluation
        logger.info(f"Generating evaluation for {len(transcript_dict)} messages")
        evaluation_report = await evaluate_conversation(transcript_dict)

        # Clean up old transcript files (keep only the latest 3)
        try:
            if len(matching_files) > 3:
                # Sort by creation time and keep only the 3 most recent
                sorted_files = sorted(
                    matching_files, key=os.path.getctime, reverse=True
                )
                for old_file in sorted_files[3:]:
                    try:
                        os.remove(old_file)
                        logger.info(f"Deleted old transcript file: {old_file}")
                    except Exception as e:
                        logger.warning(
                            f"Could not delete old transcript file {old_file}: {e}"
                        )
        except Exception as e:
            logger.warning(f"Error cleaning up old transcript files: {e}")

        # Return both transcript and evaluation
        return {
            "status": "success",
            "evaluation": evaluation_report,
            "transcript": transcript,
            "transcript_count": len(transcript_dict),
            "source_file": os.path.basename(latest_file),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-transcript/{room_name}")
async def update_transcript(room_name: str, entry: TranscriptEntry):
    """
    Update transcript in real-time for an active session.
    Called by the agent process to store transcripts in memory.
    """
    try:
        entry_dict = {
            "role": entry.role,
            "message": entry.message,
            "timestamp": entry.timestamp or datetime.now().isoformat(),
        }
        active_session_transcripts[room_name].append(entry_dict)
        logger.info(
            f"Updated transcript for room '{room_name}': {entry.role} - {entry.message[:50]}..."
        )
        return {
            "status": "success",
            "room": room_name,
            "message": "Transcript updated",
            "total_messages": len(active_session_transcripts[room_name]),
        }
    except Exception as e:
        logger.error(f"Error updating transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-transcript-batch/{room_name}")
async def update_transcript_batch(room_name: str, transcript: List[TranscriptEntry]):
    """
    Update transcript in bulk for a room (useful when initializing or syncing).
    Can also be called with empty array to initialize room in memory.
    """
    try:
        active_session_transcripts[room_name] = [
            {
                "role": entry.role,
                "message": entry.message,
                "timestamp": entry.timestamp or datetime.now().isoformat(),
            }
            for entry in transcript
        ]
        if len(transcript) > 0:
            logger.info(
                f"Updated transcript batch for room '{room_name}': {len(transcript)} messages"
            )
        else:
            logger.info(
                f"Initialized room '{room_name}' in in-memory storage (empty transcript)"
            )
        return {
            "status": "success",
            "room": room_name,
            "message": "Transcript batch updated"
            if len(transcript) > 0
            else "Room initialized",
            "total_messages": len(active_session_transcripts[room_name]),
        }
    except Exception as e:
        logger.error(f"Error updating transcript batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-transcript/{room_name}")
async def clear_transcript(room_name: str):
    """
    Clear transcript from in-memory storage (called when session ends).
    """
    try:
        if room_name in active_session_transcripts:
            count = len(active_session_transcripts[room_name])
            del active_session_transcripts[room_name]
            logger.info(f"Cleared transcript for room '{room_name}' ({count} messages)")
            return {
                "status": "success",
                "room": room_name,
                "message": f"Cleared {count} messages from memory",
            }
        else:
            return {
                "status": "success",
                "room": room_name,
                "message": "No transcript found to clear",
            }
    except Exception as e:
        logger.error(f"Error clearing transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation-history/{room_name}")
async def get_conversation_history(room_name: str):
    """
    Get complete conversation history for a specific room.
    First checks in-memory storage (real-time), then LiveKit API, then saved files.
    Returns empty array if room exists in memory but has no messages yet (active session).
    """
    try:
        # First, check in-memory storage for active sessions (real-time)
        if room_name in active_session_transcripts:
            transcript = active_session_transcripts[room_name]
            if transcript:
                logger.info(
                    f"Returning {len(transcript)} conversation entries for room: {room_name} from in-memory storage (real-time)"
                )
            else:
                logger.info(
                    f"Room '{room_name}' found in memory but transcript is empty (active session, no messages yet)"
                )
            return {
                "status": "success",
                "room": room_name,
                "transcript": transcript,  # Complete conversation array (may be empty for new sessions)
                "total_messages": len(transcript),
                "source": "in_memory_realtime",
            }

        # Second, try to get transcript from LiveKit Cloud API
        result = await get_transcript_from_livekit_api(room_name)

        if result and result.get("transcript"):
            logger.info(
                f"Returning {len(result['transcript'])} conversation entries for room: {room_name} from LiveKit API"
            )
            return {
                "status": "success",
                "room": room_name,
                "transcript": result["transcript"],  # Complete conversation array
                "total_messages": len(result["transcript"]),
                "source": result.get("source", "livekit_api"),
            }

        # Third, try saved files
        logger.info(
            f"In-memory and LiveKit API don't have transcript for room {room_name}, trying saved files..."
        )
        result = await get_transcript_from_livekit_history(room_name)

        if result and result.get("transcript"):
            logger.info(
                f"Returning {len(result['transcript'])} conversation entries for room: {room_name} from saved files"
            )
            return {
                "status": "success",
                "room": room_name,
                "transcript": result["transcript"],  # Complete conversation array
                "total_messages": len(result["transcript"]),
                "source": result.get("source", "saved_files"),
            }

        # Final fallback to other saved files
        logger.info(
            f"Saved transcript files not found for room {room_name}, trying other fallback sources..."
        )
        result = await get_transcript_fallback(room_name)

        # Ensure we return the complete transcript
        if result and result.get("transcript"):
            logger.info(
                f"Returning {len(result['transcript'])} conversation entries for room: {room_name} from fallback source"
            )
            return {
                "status": "success",
                "room": room_name,
                "transcript": result["transcript"],  # Complete conversation array
                "total_messages": len(result["transcript"]),
                "source": result.get("source", "unknown"),
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation history found for room: {room_name}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}",
        )


async def get_transcript_from_livekit_api(room_name: str):
    """
    Try to get transcript from LiveKit Cloud API directly.

    IMPORTANT LIMITATION: LiveKit's REST API does NOT provide transcript/conversation history.
    This is a limitation of LiveKit's API design. Transcripts are only available:
    1. During active sessions via session.history (in agent code)
    2. From saved files after session ends (saved in main.py from session.history)

    This function verifies the room exists but cannot retrieve transcripts via REST API.
    """
    try:
        # Get LiveKit API credentials
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        livekit_url = os.getenv("LIVEKIT_URL", "")

        if not api_key or not api_secret or not livekit_url:
            logger.info(
                f"LiveKit API credentials not configured. "
                f"Cannot query LiveKit Cloud for room {room_name}. "
                f"Will use saved transcript files instead."
            )
            return None

        # Import LiveKit API
        try:
            from livekit import api as livekit_api
        except ImportError:
            logger.info(
                f"LiveKit API package not available. "
                f"Cannot query LiveKit Cloud for room {room_name}. "
                f"Will use saved transcript files instead."
            )
            return None

        # Create LiveKit API client
        try:
            livekit_api_client = livekit_api.LiveKitAPI(
                url=livekit_url,
                api_key=api_key,
                api_secret=api_secret,
            )

            # Try to verify room exists via LiveKit API
            # NOTE: LiveKit REST API has NO endpoint for transcripts/history
            # We can only check if room exists, but cannot get conversation data
            try:
                room_service = livekit_api.RoomService(livekit_api_client)
                # LiveKit API is typically synchronous
                room_info = room_service.list_rooms(names=[room_name])

                if (
                    room_info
                    and hasattr(room_info, "rooms")
                    and len(room_info.rooms) > 0
                ):
                    room = room_info.rooms[0]
                    logger.info(
                        f"✓ Room '{room_name}' verified via LiveKit Cloud API. "
                        f"Room has {getattr(room, 'num_participants', 0)} participants. "
                        f"However, LiveKit REST API does NOT provide transcript access. "
                        f"This is a limitation of LiveKit's API - transcripts are only available "
                        f"from session.history during active sessions or from saved files. "
                        f"Falling back to saved transcript files..."
                    )
                else:
                    logger.info(
                        f"Room '{room_name}' not found in LiveKit Cloud (may have ended). "
                        f"Will check saved transcript files."
                    )
            except Exception as e:
                logger.info(
                    f"Could not verify room via LiveKit API: {e}. "
                    f"This is OK - room may have ended. Will check saved transcript files."
                )
        except Exception as e:
            logger.warning(f"Error creating LiveKit API client: {e}")
            # Continue to fallback

        # LiveKit REST API does NOT provide transcript access - this is by design
        # Transcripts must come from saved files (saved from session.history in main.py)
        logger.info(
            f"LiveKit REST API cannot provide transcripts (API limitation). "
            f"Transcripts are saved to files in main.py from session.history. "
            f"Checking saved files for room '{room_name}'..."
        )
        return None

    except Exception as e:
        logger.warning(f"Error querying LiveKit API: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        logger.info(f"Will fall back to saved transcript files for room '{room_name}'")
        return None


async def get_transcript_from_livekit_history(room_name: str):
    """
    Get transcript from LiveKit session history (saved as transcript JSON files).
    These files are saved directly from session.history in main.py and are the most accurate source.
    This is the PRIMARY source since LiveKit REST API doesn't provide transcripts.
    """
    try:
        # Search for transcript JSON files saved from LiveKit session.history
        # These are saved in main.py with format: transcript_{room_name}_{timestamp}.json
        transcript_json_pattern = str(BASE_DIR / f"transcript_{room_name}_*.json")
        logger.info(
            f"Searching for transcript files matching pattern: {transcript_json_pattern}"
        )
        transcript_json_files = glob.glob(transcript_json_pattern)
        logger.info(
            f"Found {len(transcript_json_files)} transcript JSON file(s) for room '{room_name}': {transcript_json_files}"
        )

        if transcript_json_files:
            # Get the most recent transcript file (by modification time, not creation time)
            # This ensures we get the latest version if the file was updated
            latest_json = max(transcript_json_files, key=os.path.getmtime)
            logger.info(
                f"✓ Loading transcript from LiveKit session history file: {latest_json}"
            )

            with open(latest_json, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            # Handle different JSON structures
            if isinstance(transcript_data, list):
                transcript = transcript_data
                logger.info(f"Transcript data is a list with {len(transcript)} entries")
            elif isinstance(transcript_data, dict):
                # Check if this is the format saved in main.py: {"room": ..., "timestamp": ..., "transcript": ...}
                transcript = transcript_data.get("transcript", [])
                logger.info(
                    f"Transcript data is a dict. Found 'transcript' key with {len(transcript)} entries. "
                    f"Room in file: {transcript_data.get('room')}, Timestamp: {transcript_data.get('timestamp')}"
                )
                # Verify the room name matches
                if transcript_data.get("room") != room_name:
                    logger.warning(
                        f"Room name mismatch: expected '{room_name}', got '{transcript_data.get('room')}'. "
                        f"Using transcript anyway."
                    )
            else:
                transcript = []
                logger.warning(
                    f"Unexpected transcript data format: {type(transcript_data)}"
                )

            if transcript:
                logger.info(
                    f"✓ Successfully loaded {len(transcript)} conversation entries from LiveKit session history "
                    f"(saved from session.history in main.py)"
                )
                return {
                    "status": "success",
                    "room": room_name,
                    "transcript": transcript,  # Complete conversation array
                    "source": "livekit_session_history",
                }
            else:
                logger.warning(
                    f"Transcript file found but transcript array is empty for room '{room_name}'"
                )

        # No transcript JSON files found
        logger.info(
            f"No transcript JSON files found for room '{room_name}' matching pattern: transcript_{room_name}_*.json"
        )
        return None

    except Exception as e:
        logger.error(f"Error getting transcript from LiveKit history: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


async def get_transcript_fallback(room_name: str):
    """Get complete transcript from saved files - returns entire conversation (fallback when LiveKit history not available)"""
    try:
        # First, check for evaluation JSON files matching the room name
        pattern = str(BASE_DIR / f"evaluation_{room_name}_*.json")
        matching_files = glob.glob(pattern)

        if matching_files:
            latest_file = max(matching_files, key=os.path.getctime)
            logger.info(f"Loading transcript from evaluation file: {latest_file}")
            with open(latest_file, "r", encoding="utf-8") as f:
                evaluation_data = json.load(f)

            transcript = evaluation_data.get("transcript", [])
            logger.info(
                f"Found {len(transcript)} conversation entries in evaluation file"
            )

            return {
                "status": "success",
                "room": room_name,
                "transcript": transcript,  # Complete conversation array
                "source": "evaluation_file",
            }

        # Fallback: check for text transcript files
        transcript_pattern = str(BASE_DIR / "conversation_transcript_*.txt")
        all_transcript_files = glob.glob(transcript_pattern)

        if all_transcript_files:
            # Get the most recent transcript file
            latest_transcript = max(all_transcript_files, key=os.path.getctime)
            logger.info(f"Loading transcript from text file: {latest_transcript}")

            with open(latest_transcript, "r", encoding="utf-8") as f:
                content = f.read()

            transcript = []
            for line in content.split("\n"):
                if line.strip() and not line.startswith("==="):
                    # Parse format: [timestamp] Role: message
                    if "] " in line:
                        parts = line.split("] ", 1)
                        if len(parts) == 2:
                            role_part = parts[1].split(": ", 1)
                            if len(role_part) == 2:
                                transcript.append({
                                    "role": role_part[0].strip(),
                                    "message": role_part[1].strip(),
                                    "timestamp": parts[0].replace("[", "").strip(),
                                })

            logger.info(f"Parsed {len(transcript)} conversation entries from text file")

            return {
                "status": "success",
                "room": room_name,
                "transcript": transcript,  # Complete conversation array
                "source": "transcript_file",
            }

        raise HTTPException(
            status_code=404,
            detail=f"No conversation history found for room: {room_name}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in transcript fallback: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}",
        )


@router.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "livekit_configured": bool(
            os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET")
        ),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }
