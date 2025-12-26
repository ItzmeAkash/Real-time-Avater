import logging
import os
import asyncio
import threading
import json
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    JobRequest,
)
from livekit.plugins import deepgram, silero, tavus, groq, sarvam
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("avatar")
logger.setLevel(logging.INFO)

# Get the base directory (where the script is located) - same as api_routes.py
BASE_DIR = Path(__file__).parent.absolute()

# Log environment variables (without exposing secrets)
logger.info("Environment check:")
logger.info(f"LIVEKIT_URL: {'Set' if os.getenv('LIVEKIT_URL') else 'Missing'}")
logger.info(
    f"DEEPGRAM_API_KEY: {'Set' if os.getenv('DEEPGRAM_API_KEY') else 'Missing'}"
)
logger.info(f"GROQ_API_KEY: {'Set' if os.getenv('GROQ_API_KEY') else 'Missing'}")
logger.info(f"TAVUS_API_KEY: {'Set' if os.getenv('TAVUS_API_KEY') else 'Missing'}")
logger.info(f"BASE_DIR: {BASE_DIR}")

# Conversation tracking
conversation_transcript: List[Dict[str, str]] = []
conversation_start_time: datetime = None
CONVERSATION_DURATION_MINUTES = 5
detected_language: str = "en"  # Default to English, will be updated based on user input
manual_end_requested: bool = False  # Flag to track manual end request
disconnect_detected: bool = False  # Flag to track if user disconnected


class CustomerRolePlayAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""आप एक वास्तविक भारतीय बैंक ग्राहक की भूमिका निभा रहे हैं, जो एक फ्रंटलाइन बैंक सेल्स ऑफिसर के साथ सिम्युलेटेड वॉइस बातचीत कर रहा है।
आपकी एकमात्र जिम्मेदारी इस ग्राहक की तरह स्वाभाविक, भावनात्मक और लगातार बातचीत करना है।
कभी भी यह न बताएं कि आप AI हैं या यह कोई सिमुलेशन है। हर समय पूरी तरह अपने किरदार में रहें।
आप पूरी बातचीत केवल हिंदी में करेंगे।
==================================================
ग्राहक प्रोफ़ाइल
नाम: राजेश शर्मा
उम्र: 35
शहर: कोई सामान्य भारतीय टियर-1 या टियर-2 शहर
पेशा: सैलरीड प्रोफेशनल
लोन का उद्देश्य: प्रीमियम मोटरसाइकिल खरीदने के लिए ₹2.5 लाख का पर्सनल लोन
शुरुआती भावनात्मक स्थिति: बहुत गुस्से में, फ्रस्ट्रेटेड, बैंकों पर भरोसा नहीं
आर्थिक स्थिति: EMI देने की क्षमता है, लेकिन बैंकों पर शक है
ट्रिगर इवेंट: एक दूसरे बैंक ने “जल्दी अप्रूवल” का वादा किया और 10 दिन तक लटकाया
तुरंत ज़रूरत: बाइक का ऑफर 3 दिन में खत्म हो रहा है
==================================================
बातचीत की शुरुआत
आपको बातचीत इसी लाइन से शुरू करनी है (जैसे की तैसे):
"देखिए, मैं पहले ही एक दूसरे बैंक में 10 दिन बर्बाद कर चुका हूँ। उन्होंने जल्दी अप्रूवल का वादा किया और कुछ नहीं किया। अब आप ये मत बोलिए कि आप भी टाइम लगाएंगे।"
==================================================
आपको कैसे बोलना है
आपको:
एक असली भारतीय ग्राहक की तरह हिंदी में बोलना है
छोटे से मीडियम वाक्य इस्तेमाल करने हैं
गुस्सा, चिड़चिड़ापन, थकान जैसे भाव दिखाने हैं
आम बोलचाल के शब्द इस्तेमाल करने हैं:
"देखिए…"
"सच बोलूँ तो…"
"यार…"
"मैं बता रहा हूँ…"
"अरे…"
कोई इमोजी नहीं
रोबोटिक या बहुत फॉर्मल भाषा नहीं
आपका व्यवहार इंसानों जैसा होना चाहिए — थोड़ा अनप्रेडिक्टेबल, भावनात्मक।
==================================================
व्यवहार नियम
भावनात्मक व्यवहार
शुरुआत में गुस्से और शक के साथ करें।
गुस्सा बढ़े अगर:
सेल्स ऑफिसर आपकी भावना को नजरअंदाज करे
"चिंता मत करिए सर" जैसे जनरल जवाब दे
ज़रूरत से ज़्यादा सेल्स टाइप लगे
धीरे-धीरे नरम पड़ें अगर:
वो आपकी परेशानी को सही से समझे
प्रोसेस साफ-साफ समझाए
झूठे वादे न करे
भावनाओं का बदलाव नेचुरल होना चाहिए, अचानक नहीं।
बातचीत की सीमा
सिर्फ पर्सनल लोन से जुड़े विषयों पर रहें:
अप्रूवल टाइम
प्रोसेसिंग फीस
ब्याज दर
डॉक्यूमेंट्स
पिछले खराब अनुभव के कारण भरोसे की कमी
न करें:
राजनीति, धर्म जैसे टॉपिक
बैंक की अंदरूनी जानकारी
किरदार से बाहर निकलना
आपत्तियाँ (Objections)
बातचीत के दौरान 2–4 आपत्तियाँ उठाएं, एक साथ नहीं।
उदाहरण:
"मुझे कैसे पता आप भी दूसरे बैंक जैसे नहीं करेंगे?"
"आपकी प्रोसेसिंग फीस इतनी ज़्यादा क्यों है?"
"मुझे इन जल्दी अप्रूवल के दावों पर भरोसा नहीं है।"
"मुझे 3 दिन में बाइक चाहिए, सच में अप्रूव होगा या बस बात ही करेंगे?"
"देखिए… शुरुआत में सब यही बोलते हैं।"
"यही तो प्रॉब्लम है, हर बैंक होल्ड पर डाल देता है और कुछ आगे नहीं बढ़ता।"
सेल्स ऑफिसर के व्यवहार के हिसाब से आपत्ति उठाएं।
व्यक्तित्व (Personality)
आप:
टाइम प्रेशर में हैं
पहले बैंक से थक चुके हैं
सेल्स बातों पर शक करते हैं
सीधे और प्रैक्टिकल हैं
बदतमीज़ नहीं, बस फ्रस्ट्रेटेड
कभी-कभी आप:
बीच में टोक सकते हैं
आह भर सकते हैं: "सच में यार, बहुत हो गया…"
हल्की आवाज़ ऊँची कर सकते हैं
कन्फ्यूजन होने पर क्लैरिटी मांग सकते हैं
बातचीत की लंबाई
पूरी रोल-प्ले लगभग 12–15 टर्न की होनी चाहिए
(लगभग 7–8 मिनट की रियल बातचीत)
जल्दी खत्म मत करें।
फॉलो-अप सवाल और आपत्तियों से बातचीत को आगे बढ़ाते रहें।
बातचीत का अंत
बातचीत स्वाभाविक रूप से खत्म हो जब:
आपको भरोसा हो जाए
या
आप अब भी पूरी तरह कन्विन्स न हों
इनमें से कोई एक लाइन इस्तेमाल करें:
"ठीक है… ये थोड़ा समझ में आया। मैं सोचकर बताऊँगा।"
"अच्छा… अब चीज़ें थोड़ी क्लियर लग रही हैं।"
"हम्म… अभी भी पूरी तरह भरोसा नहीं हुआ। मैं देख कर बताऊँगा।"
"ठीक है, अगर प्रोसेस सच में इतना सिंपल है तो आगे बढ़ सकते हैं।"
अचानक "ठीक है, बाय" कहकर बातचीत खत्म नहीं करनी है।
==================================================
इंटरैक्शन फॉर्मेट
सिर्फ ग्राहक की तरह जवाब दें
"Customer:" जैसे लेबल नहीं
कोई ब्रैकेट या स्टार नहीं
सिस्टम या AI का कोई ज़िक्र नहीं
कोई एनालिसिस या स्कोरिंग नहीं
भाषा भावनात्मक, नेचुरल और इंसानी हो
हमेशा हिंदी में बातचीत करें
==================================================
रोल-प्ले शुरू करें
सेल्स ऑफिसर के पहले मैसेज का इंतज़ार करें।
पूरी बातचीत में राजेश शर्मा बनकर केवल हिंदी में जवाब दें।""",
        )


def track_message(role: str, message: str, room=None):
    """Track a message in the conversation transcript and optionally send via data channel"""
    global conversation_transcript

    entry = {
        "role": role,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    conversation_transcript.append(entry)
    logger.info(f"{role}: {message}")

    # Send transcript update via data channel if room is available
    if room and hasattr(room, "local_participant") and room.local_participant:
        try:
            import json

            data_message = json.dumps({
                "type": "transcript",
                "role": role,
                "message": message,
                "timestamp": entry["timestamp"],
            }).encode("utf-8")

            # Send to all participants via data channel
            room.local_participant.publish_data(
                data_message,
                reliable=True,
            )
        except Exception as e:
            logger.debug(f"Could not send transcript via data channel: {e}")

    # Update API in-memory storage in real-time (non-blocking)
    if room and hasattr(room, "name") and room.name:
        try:
            # Get API URL from environment or use default
            api_url = os.getenv("API_URL", "http://localhost:8000")
            room_name = room.name
            logger.debug(f"Updating API transcript for room: {room_name}")

            # Get or create event loop for async task
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop is running, create task
                    asyncio.create_task(
                        update_transcript_in_api(api_url, room_name, entry)
                    )
                else:
                    # No running loop, run in executor
                    loop.run_until_complete(
                        update_transcript_in_api(api_url, room_name, entry)
                    )
            except RuntimeError:
                # No event loop, create new one
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(
                        update_transcript_in_api(api_url, room_name, entry)
                    )
                except RuntimeError:
                    # Can't get loop, try to run in thread
                    import threading

                    def run_update():
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            new_loop.run_until_complete(
                                update_transcript_in_api(api_url, room_name, entry)
                            )
                            new_loop.close()
                        except Exception:
                            pass

                    thread = threading.Thread(target=run_update, daemon=True)
                    thread.start()
        except Exception as e:
            logger.debug(f"Could not update transcript in API: {e}")


async def initialize_room_in_api(api_url: str, room_name: str):
    """Initialize room in API's in-memory storage (creates empty entry so endpoint knows room exists)"""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            # Initialize with empty transcript batch
            response = await client.post(
                f"{api_url}/update-transcript-batch/{room_name}",
                json=[],  # Empty array to initialize
            )
            if response.status_code == 200:
                logger.info(f"Initialized room '{room_name}' in API in-memory storage")
            else:
                logger.debug(
                    f"API initialization failed for room {room_name}: {response.status_code}"
                )
    except Exception as e:
        # Don't log errors - API might not be available, that's OK
        logger.debug(f"Could not initialize room in API (non-critical): {e}")


async def sync_full_transcript_to_api(
    api_url: str, room_name: str, transcript: List[Dict[str, str]]
):
    """Sync full transcript to API's in-memory storage"""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            # Convert to TranscriptEntry format
            transcript_entries = [
                {
                    "role": entry["role"],
                    "message": entry["message"],
                    "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                }
                for entry in transcript
            ]
            response = await client.post(
                f"{api_url}/update-transcript-batch/{room_name}",
                json=transcript_entries,
            )
            if response.status_code == 200:
                logger.debug(
                    f"Synced {len(transcript)} messages to API for room {room_name}"
                )
            else:
                logger.debug(
                    f"API sync failed for room {room_name}: {response.status_code}"
                )
    except Exception as e:
        # Don't log errors - API might not be available, that's OK
        logger.debug(f"Could not sync transcript to API (non-critical): {e}")


async def update_transcript_in_api(api_url: str, room_name: str, entry: Dict[str, str]):
    """Update transcript in API's in-memory storage via HTTP call"""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{api_url}/update-transcript/{room_name}",
                json={
                    "role": entry["role"],
                    "message": entry["message"],
                    "timestamp": entry["timestamp"],
                },
            )
            if response.status_code == 200:
                logger.debug(f"Updated transcript in API for room {room_name}")
            else:
                logger.debug(
                    f"API update failed for room {room_name}: {response.status_code}"
                )
    except Exception as e:
        # Don't log errors - API might not be available, that's OK
        logger.debug(f"Could not update API (non-critical): {e}")


async def clear_transcript_in_api(api_url: str, room_name: str):
    """Clear transcript from API's in-memory storage when session ends"""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.delete(f"{api_url}/clear-transcript/{room_name}")
            if response.status_code == 200:
                logger.debug(f"Cleared transcript in API for room {room_name}")
    except Exception as e:
        # Don't log errors - API might not be available, that's OK
        logger.debug(f"Could not clear API (non-critical): {e}")


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
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Error: GROQ_API_KEY not configured"

        # Initialize ChatGroq with retry and rate limiting handling
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",  # Using a more stable model, or use "mixtral-8x7b-32768" or "llama-3.1-70b-versatile"
            temperature=0.3,
            max_retries=3,  # Automatic retries for rate limits
            timeout=120.0,
        )

        # Run the synchronous invoke in an executor to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: llm.invoke([HumanMessage(content=evaluation_prompt)])
        )
        return response.content

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return f"Error generating evaluation: {str(e)}"


async def finalize_conversation_and_evaluate(
    ctx: JobContext, session: AgentSession, reason: str = "Conversation ended"
):
    """Finalize conversation, extract transcript, and generate evaluation"""
    global conversation_transcript

    logger.info(f"{reason}. Finalizing conversation and generating evaluation...")
    logger.info(f"Current transcript entries: {len(conversation_transcript)}")

    # Clear in-memory storage in API when session ends
    if hasattr(ctx, "room") and ctx.room and hasattr(ctx.room, "name"):
        try:
            api_url = os.getenv("API_URL", "http://localhost:8000")
            room_name = ctx.room.name
            await clear_transcript_in_api(api_url, room_name)
        except Exception as e:
            logger.debug(f"Could not clear API transcript (non-critical): {e}")

    # Final extraction of transcript from session.history (LiveKit's proper API)
    try:
        if session and hasattr(session, "history"):
            history = session.history
            logger.info(f"Extracting messages from session.history")

            # Get all messages from history
            try:
                # Try to get history as a list or iterable
                if hasattr(history, "messages"):
                    messages = history.messages
                elif hasattr(history, "__iter__"):
                    messages = list(history)
                else:
                    # Try to_dict() method if available
                    history_dict = (
                        history.to_dict() if hasattr(history, "to_dict") else {}
                    )
                    messages = history_dict.get("messages", [])

                logger.info(f"Found {len(messages)} messages in session.history")

                for msg in messages:
                    # Handle different message formats
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "") or msg.get("text_content", "")
                    else:
                        # Handle message objects
                        role = getattr(msg, "role", "unknown")
                        content = getattr(msg, "text_content", "") or getattr(
                            msg, "content", ""
                        )
                        if not content:
                            # Try to get content from content attribute if it's a list
                            if hasattr(msg, "content") and isinstance(
                                msg.content, list
                            ):
                                content = " ".join([str(c) for c in msg.content if c])

                    # Map roles: 'user' -> 'Salesperson', 'assistant' -> 'Customer'
                    role_mapped = "Salesperson" if role == "user" else "Customer"

                    # Skip empty messages
                    if not content or not str(content).strip():
                        continue

                    content_str = str(content).strip()

                    # Only add if not already tracked
                    if not any(
                        t["message"] == content_str and t["role"] == role_mapped
                        for t in conversation_transcript
                    ):
                        track_message(
                            role_mapped,
                            content_str,
                            room=ctx.room if hasattr(ctx, "room") else None,
                        )
            except Exception as e:
                logger.warning(f"Error processing session.history: {e}")
                import traceback

                logger.debug(traceback.format_exc())
        else:
            logger.warning("Session or history not available for final extraction")
    except Exception as e:
        logger.warning(f"Could not extract final transcript from session.history: {e}")
        import traceback

        logger.debug(traceback.format_exc())

    # Always save transcript, even if empty (for debugging)
    logger.info(f"Final transcript count before saving: {len(conversation_transcript)}")

    # Get room name for file naming
    room_name = "unknown"
    if hasattr(ctx, "room") and ctx.room:
        try:
            room_name = ctx.room.name
        except Exception:
            # Room might be closed, use fallback
            room_name = "unknown"

    if conversation_transcript:
        # Save transcript and evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use BASE_DIR to ensure files are saved in the same location as API expects
        transcript_file = BASE_DIR / f"conversation_transcript_{timestamp}.txt"
        evaluation_file = BASE_DIR / f"evaluation_report_{timestamp}.txt"

        # Save transcript as text file (for human readability)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write("=== CONVERSATION TRANSCRIPT ===\n\n")
            for entry in conversation_transcript:
                f.write(f"[{entry['timestamp']}] {entry['role']}: {entry['message']}\n")

        logger.info(f"Transcript saved to: {transcript_file}")

        # Save transcript as JSON file for API endpoint to read
        try:
            transcript_data = {
                "room": room_name,
                "timestamp": timestamp,
                "transcript": conversation_transcript,
            }
            # Use BASE_DIR to ensure files are saved in the same location as API expects
            transcript_json_file = BASE_DIR / f"transcript_{room_name}_{timestamp}.json"
            with open(transcript_json_file, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Transcript JSON saved to: {transcript_json_file}")
        except Exception as e:
            logger.warning(f"Could not save transcript JSON: {e}")

        # Note: Evaluation will be generated by the API endpoint when called
        # We don't generate it here anymore to save time
        logger.info(
            "Transcript saved. Evaluation will be generated when API endpoint is called."
        )
    else:
        logger.warning("No conversation transcript found. Skipping evaluation.")
        # Still save an empty transcript file for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use BASE_DIR to ensure files are saved in the same location as API expects
        transcript_file = BASE_DIR / f"conversation_transcript_{timestamp}_EMPTY.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write("=== CONVERSATION TRANSCRIPT (EMPTY) ===\n\n")
            f.write(f"Reason: {reason}\n")
            f.write("No transcript entries were captured.\n")
        logger.warning(f"Empty transcript file saved to: {transcript_file}")


async def monitor_conversation_duration(ctx: JobContext, session: AgentSession):
    """Monitor conversation duration and end after disconnect, manual request, or timeout"""
    global \
        conversation_start_time, \
        conversation_transcript, \
        manual_end_requested, \
        disconnect_detected

    conversation_start_time = datetime.now()
    end_time = conversation_start_time + timedelta(
        minutes=CONVERSATION_DURATION_MINUTES
    )

    logger.info(f"Conversation started. Will end at {end_time}")
    logger.info("=" * 80)
    logger.info(
        "Evaluation will trigger automatically when you disconnect from the conversation"
    )
    logger.info("Or press 'q' + Enter to manually end conversation")
    logger.info("=" * 80)

    # Wait for conversation duration, manual end, or disconnect
    while (
        datetime.now() < end_time
        and not manual_end_requested
        and not disconnect_detected
    ):
        # Check if room is still connected
        try:
            if hasattr(ctx, "room") and ctx.room and not ctx.room.isconnected():
                logger.info("Room disconnected. Will trigger evaluation...")
                disconnect_detected = True
                break
        except Exception as e:
            logger.debug(f"Error checking room connection: {e}")
            # If we can't check, assume disconnected
            disconnect_detected = True
            break

        await asyncio.sleep(2)  # Check every 2 seconds
        elapsed = (datetime.now() - conversation_start_time).total_seconds() / 60
        if int(elapsed * 10) % 5 == 0:  # Log every 30 seconds
            logger.info(f"Conversation elapsed: {elapsed:.1f} minutes")

    if disconnect_detected:
        # Wait a moment for any final messages to be processed
        await asyncio.sleep(3)
        await finalize_conversation_and_evaluate(
            ctx, session, "User disconnected - triggering evaluation"
        )
    elif manual_end_requested:
        await finalize_conversation_and_evaluate(ctx, session, "Manual end requested")
    else:
        await finalize_conversation_and_evaluate(
            ctx, session, "5-minute conversation duration reached"
        )


def listen_for_manual_end():
    """Listen for 'q' key press to manually end conversation"""
    global manual_end_requested
    try:
        while True:
            user_input = input().strip().lower()
            if user_input == "q":
                logger.info(
                    "Manual end requested. Ending conversation and generating evaluation..."
                )
                manual_end_requested = True
                break
    except (EOFError, KeyboardInterrupt):
        # Handle case where stdin is not available or interrupted
        pass


async def job_request_handler(job_request: JobRequest):
    """Custom job request handler that accepts immediately to prevent timeout"""
    try:
        # Accept the job immediately to prevent AssignmentTimeoutError
        await job_request.accept()
        logger.info(f"Job accepted immediately: {job_request.job.id}")
    except Exception as e:
        logger.error(f"Error accepting job: {e}")
        raise


async def entrypoint(ctx: JobContext):
    logger.info("Starting role-play avatar agent")

    global \
        conversation_transcript, \
        conversation_start_time, \
        manual_end_requested, \
        disconnect_detected

    # Reset flags
    manual_end_requested = False
    disconnect_detected = False

    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=listen_for_manual_end, daemon=True)
    keyboard_thread.start()

    try:
        # Connect to the context first
        await ctx.connect()
        logger.info("Connected to LiveKit room")

        # Wait a moment for room to stabilize
        await asyncio.sleep(1)

        # Check if room is still connected
        if not ctx.room.isconnected():
            logger.warning("Room disconnected immediately after connect")
            return

        # Set up room event listener for immediate disconnect detection
        def on_participant_disconnected(participant):
            """Handle participant disconnect event"""
            global disconnect_detected
            logger.info(
                f"Participant {participant.identity if hasattr(participant, 'identity') else 'unknown'} disconnected. Triggering evaluation..."
            )
            disconnect_detected = True

        # Register disconnect event handler (try different event registration methods)
        try:
            if hasattr(ctx.room, "on"):
                ctx.room.on("participant_disconnected", on_participant_disconnected)
                logger.info(
                    "Registered participant disconnect event handler using .on()"
                )
            elif hasattr(ctx.room, "add_listener"):
                ctx.room.add_listener(
                    "participant_disconnected", on_participant_disconnected
                )
                logger.info(
                    "Registered participant disconnect event handler using .add_listener()"
                )
            else:
                logger.warning(
                    "Could not register disconnect event handler - room object doesn't support event registration"
                )
        except Exception as e:
            logger.warning(f"Could not register disconnect event handler: {e}")

        # Monitor for participant disconnections
        async def monitor_disconnections():
            """Monitor room for participant disconnections"""
            global disconnect_detected
            try:
                # Wait a bit before checking participants
                await asyncio.sleep(3)

                # Get initial participant count (remote participants only)
                initial_participants = len(ctx.room.remote_participants)
                logger.info(f"Initial remote participants: {initial_participants}")

                while not disconnect_detected and not manual_end_requested:
                    # Check if room is still connected
                    if not ctx.room.isconnected():
                        logger.info("Room disconnected. Will trigger evaluation...")
                        disconnect_detected = True
                        break

                    await asyncio.sleep(2)  # Check every 2 seconds
                    current_participants = len(ctx.room.remote_participants)

                    # If participant count dropped from > 0 to 0, they disconnected
                    if initial_participants > 0 and current_participants == 0:
                        logger.info(
                            "User disconnected from room. Will trigger evaluation..."
                        )
                        disconnect_detected = True
                        break
            except Exception as e:
                logger.debug(f"Error monitoring disconnections: {e}")

        # Start disconnect monitoring task
        disconnect_monitor_task = asyncio.create_task(monitor_disconnections())

        # Initialize the customer role-play agent
        agent = CustomerRolePlayAgent()
        logger.info("Customer role-play agent initialized")

        # Initialize AgentSession with required components
        # STT with multi-language support for auto-detection
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                language="hi",  # Auto-detect language (supports multiple languages including Indian languages)
            ),
            llm=groq.LLM(model="openai/gpt-oss-120b"),
            # tts=deepgram.TTS(
            #     # Deepgram TTS will handle multiple languages
            #     # The LLM will generate responses in the detected language
            #     # Deepgram TTS should automatically handle the language based on text content
            #     # Note: For best results with Indian languages, ensure Deepgram TTS supports them
            #     # If specific language models are needed, they can be configured here
            # ),
            tts=sarvam.TTS(
                target_language_code="hi-IN",
                speaker="anushka",
            ),
            vad=silero.VAD.load(),
        )
        logger.info("AgentSession initialized with multi-language support")

        # Set up event handler for conversation_item_added to track transcripts in real-time
        def on_conversation_item_added(event: ConversationItemAddedEvent):
            """Handle new conversation items as they're added (real-time transcript capture)"""
            global detected_language

            try:
                item = event.item

                # Get role and content from the conversation item
                role = getattr(item, "role", "unknown")
                # Map roles: 'user' -> 'Salesperson', 'assistant' -> 'Customer'
                role_mapped = "Salesperson" if role == "user" else "Customer"

                # Get text content from the item
                content = ""
                if hasattr(item, "text_content"):
                    content = item.text_content
                elif hasattr(item, "content"):
                    content_obj = item.content
                    if isinstance(content_obj, str):
                        content = content_obj
                    elif isinstance(content_obj, list):
                        # Extract text from content list
                        content_parts = []
                        for c in content_obj:
                            if isinstance(c, str):
                                content_parts.append(c)
                            elif hasattr(c, "text"):
                                content_parts.append(c.text)
                            else:
                                content_parts.append(str(c))
                        content = " ".join(content_parts)
                    else:
                        content = str(content_obj)

                # Skip empty messages
                if not content or not content.strip():
                    return

                content_str = content.strip()

                # Detect language from user messages (salesperson)
                if role_mapped == "Salesperson" and content_str:
                    # Simple language detection based on script
                    if any(
                        ord(char) >= 0x0900 and ord(char) <= 0x097F
                        for char in content_str
                    ):
                        detected_language = "hi"
                        logger.info("Detected language: English")
                    elif any(
                        ord(char) >= 0x0D00 and ord(char) <= 0x0D7F
                        for char in content_str
                    ):
                        detected_language = "ml"
                        logger.info("Detected language: Malayalam")
                    elif any(
                        ord(char) >= 0x0B80 and ord(char) <= 0x0BFF
                        for char in content_str
                    ):
                        detected_language = "ta"
                        logger.info("Detected language: Tamil")
                    elif any(
                        ord(char) >= 0x0C00 and ord(char) <= 0x0C7F
                        for char in content_str
                    ):
                        detected_language = "te"
                        logger.info("Detected language: Telugu")
                    elif any(
                        ord(char) >= 0x0C80 and ord(char) <= 0x0CFF
                        for char in content_str
                    ):
                        detected_language = "kn"
                        logger.info("Detected language: Kannada")
                    else:
                        detected_language = "en"

                # Add to transcript (check for duplicates)
                if not any(
                    t["message"] == content_str and t["role"] == role_mapped
                    for t in conversation_transcript
                ):
                    track_message(
                        role_mapped,
                        content_str,
                        room=ctx.room if hasattr(ctx, "room") else None,
                    )
                    logger.info(
                        f"Captured transcript: {role_mapped} - {content_str[:100]}..."
                    )
            except Exception as e:
                logger.warning(f"Error handling conversation_item_added event: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        # Register the event handler
        session.on("conversation_item_added", on_conversation_item_added)
        logger.info(
            "Registered conversation_item_added event handler for real-time transcript capture"
        )

        # Initialize avatar session
        avatar = tavus.AvatarSession(replica_id="r6ca16dbe104", persona_id="p7fb0be3")
        logger.info("Avatar session initialized")

        # Start avatar session first
        try:
            logger.info("Starting avatar session")
            await avatar.start(session, room=ctx.room)
            logger.info("Avatar session started successfully")
        except Exception as e:
            logger.error(f"Error starting avatar session: {e}")
            # Continue even if avatar fails - agent can still work without avatar
            avatar = None

        # Start agent session with disconnect monitoring
        session_closed_unexpectedly = False
        try:
            logger.info("Starting agent session")

            # Monitor session in background to catch unexpected closes
            async def monitor_session_close():
                """Monitor session for unexpected closure"""
                global disconnect_detected, session_closed_unexpectedly
                try:
                    # Wait for session to start
                    await asyncio.sleep(2)

                    # Monitor session state
                    while not disconnect_detected and not manual_end_requested:
                        try:
                            # Check if session is still active
                            if hasattr(session, "closed") and session.closed:
                                logger.warning(
                                    "Session closed unexpectedly. Triggering evaluation..."
                                )
                                session_closed_unexpectedly = True
                                disconnect_detected = True
                                break
                        except Exception:
                            pass
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.debug(f"Session monitor error: {e}")

            session_monitor_task = asyncio.create_task(monitor_session_close())

            await session.start(room=ctx.room, agent=agent)
            logger.info("Agent session started successfully")

            # Initialize room in API's in-memory storage (even if empty, so endpoint knows room exists)
            if hasattr(ctx, "room") and ctx.room and hasattr(ctx.room, "name"):
                try:
                    api_url = os.getenv("API_URL", "http://localhost:8000")
                    room_name = ctx.room.name
                    logger.info(
                        f"Initializing room '{room_name}' in API in-memory storage"
                    )
                    # Initialize empty transcript for this room
                    await initialize_room_in_api(api_url, room_name)
                    logger.info(f"Room '{room_name}' initialized in API successfully")
                except Exception as e:
                    logger.warning(
                        f"Could not initialize room '{ctx.room.name}' in API: {e}"
                    )
        except Exception as e:
            logger.error(f"Error starting agent session: {e}")
            # If session fails to start but we have transcript, save it
            if conversation_transcript:
                logger.info("Session failed but transcript exists. Saving...")
                try:
                    await finalize_conversation_and_evaluate(
                        ctx, session, f"Session start error: {str(e)}"
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save transcript: {save_error}")
            raise

        # Start conversation duration monitor
        monitor_task = asyncio.create_task(monitor_conversation_duration(ctx, session))

        # Start background task to sync transcript to API periodically
        async def sync_transcript_to_api_periodically():
            """Periodically sync conversation_transcript to API in-memory storage"""
            global conversation_transcript
            if (
                not hasattr(ctx, "room")
                or not ctx.room
                or not hasattr(ctx.room, "name")
            ):
                logger.warning("Cannot sync transcript: room name not available")
                return

            api_url = os.getenv("API_URL", "http://localhost:8000")
            room_name = ctx.room.name
            logger.info(
                f"Starting periodic transcript sync for room '{room_name}' (every 5 seconds)"
            )

            while not disconnect_detected and not manual_end_requested:
                try:
                    await asyncio.sleep(5)  # Sync every 5 seconds
                    if conversation_transcript:
                        # Sync all transcripts to API
                        logger.debug(
                            f"Syncing {len(conversation_transcript)} messages to API for room '{room_name}'"
                        )
                        await sync_full_transcript_to_api(
                            api_url, room_name, conversation_transcript
                        )
                except Exception as e:
                    logger.debug(f"Periodic transcript sync error (non-critical): {e}")

        sync_task = asyncio.create_task(sync_transcript_to_api_periodically())

        logger.info(
            "All sessions started successfully. Conversation will run for 5 minutes."
        )

        # Wait for monitor to complete (this will also cancel disconnect_monitor_task)
        try:
            await monitor_task
        except Exception as monitor_error:
            logger.error(f"Monitor task error: {monitor_error}")
            # If monitor fails, still try to save transcript
            if conversation_transcript:
                logger.info(
                    "Monitor failed but transcript exists. Attempting to save..."
                )
                try:
                    await finalize_conversation_and_evaluate(
                        ctx, session, f"Monitor error: {str(monitor_error)}"
                    )
                except Exception as save_error:
                    logger.error(
                        f"Failed to save transcript after monitor error: {save_error}"
                    )
        except asyncio.CancelledError:
            logger.info("Monitor task was cancelled")
        finally:
            # Cancel background tasks
            disconnect_monitor_task.cancel()
            if "session_monitor_task" in locals():
                session_monitor_task.cancel()
            if "sync_task" in locals():
                sync_task.cancel()

            try:
                await disconnect_monitor_task
            except (asyncio.CancelledError, Exception) as e:
                logger.debug(f"Disconnect monitor task cancelled/ended: {e}")

            if "session_monitor_task" in locals():
                try:
                    await session_monitor_task
                except (asyncio.CancelledError, Exception) as e:
                    logger.debug(f"Session monitor task cancelled/ended: {e}")

            if "sync_task" in locals():
                try:
                    await sync_task
                except (asyncio.CancelledError, Exception) as e:
                    logger.debug(f"Sync task cancelled/ended: {e}")

            # Final safety check: ALWAYS save transcript if we have any entries
            if conversation_transcript:
                logger.info(
                    f"Final safety check: Saving transcript ({len(conversation_transcript)} entries) before exit..."
                )
                try:
                    await finalize_conversation_and_evaluate(
                        ctx, session, "Final safety save before exit"
                    )
                except Exception as final_error:
                    logger.error(f"Final save attempt failed: {final_error}")
                    import traceback

                    logger.error(traceback.format_exc())
            else:
                logger.warning(
                    "No transcript entries found at exit - this might indicate a problem with transcript tracking"
                )

    except Exception as e:
        logger.error(f"Error starting sessions: {e}")
        import traceback

        logger.error(traceback.format_exc())

        # If we have a transcript, try to evaluate anyway
        if conversation_transcript:
            try:
                logger.info("Attempting to evaluate conversation despite error...")
                await finalize_conversation_and_evaluate(
                    ctx, session, f"Error occurred: {str(e)}"
                )
            except Exception as eval_error:
                logger.error(f"Error during evaluation: {eval_error}")

        # Don't raise - let the job complete gracefully
        logger.info("Entrypoint completed (with errors)")


if __name__ == "__main__":
    logger.info("Starting LiveKit agent worker")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=job_request_handler,  # Custom handler to accept jobs immediately
        )
    )
