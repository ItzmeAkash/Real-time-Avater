import os
import json
import glob
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger("api")
router = APIRouter()

# Get the base directory (where the script is located)
BASE_DIR = Path(__file__).parent.absolute()


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
    import httpx

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

    # Use Groq LLM for evaluation
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-120b",
                    "messages": [{"role": "user", "content": evaluation_prompt}],
                    "temperature": 0.3,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during evaluation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error calling evaluation API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating evaluation: {str(e)}"
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
