import logging
import os
import asyncio
import threading
import json
from datetime import datetime, timedelta
from typing import List, Dict
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
)
from livekit.plugins import deepgram, silero, tavus, groq, sarvam
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("avatar")
logger.setLevel(logging.INFO)

# Log environment variables (without exposing secrets)
logger.info("Environment check:")
logger.info(f"LIVEKIT_URL: {'Set' if os.getenv('LIVEKIT_URL') else 'Missing'}")
logger.info(
    f"DEEPGRAM_API_KEY: {'Set' if os.getenv('DEEPGRAM_API_KEY') else 'Missing'}"
)
logger.info(f"GROQ_API_KEY: {'Set' if os.getenv('GROQ_API_KEY') else 'Missing'}")
logger.info(f"TAVUS_API_KEY: {'Set' if os.getenv('TAVUS_API_KEY') else 'Missing'}")

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

    # Use Groq LLM for evaluation
    try:
        # Use direct API call to Groq
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
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
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return f"Error generating evaluation: {str(e)}"


async def finalize_conversation_and_evaluate(
    ctx: JobContext, session: AgentSession, reason: str = "Conversation ended"
):
    """Finalize conversation, extract transcript, and generate evaluation"""
    global conversation_transcript

    logger.info(f"{reason}. Finalizing conversation and generating evaluation...")

    # Final extraction of transcript from session
    try:
        if hasattr(session, "chat_ctx") and session.chat_ctx:
            messages = session.chat_ctx.messages
            for msg in messages:
                role = "Salesperson" if msg.role == "user" else "Customer"
                content = msg.content if hasattr(msg, "content") else str(msg)
                # Only add if not already tracked
                if not any(
                    t["message"] == content and t["role"] == role
                    for t in conversation_transcript
                ):
                    track_message(
                        role, content, room=ctx.room if hasattr(ctx, "room") else None
                    )
    except Exception as e:
        logger.warning(f"Could not extract final transcript: {e}")

    # Generate evaluation
    if conversation_transcript:
        logger.info("Generating evaluation report...")
        evaluation_report = await evaluate_conversation(conversation_transcript)

        # Save transcript and evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = f"conversation_transcript_{timestamp}.txt"
        evaluation_file = f"evaluation_report_{timestamp}.txt"

        # Save transcript
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write("=== CONVERSATION TRANSCRIPT ===\n\n")
            for entry in conversation_transcript:
                f.write(f"[{entry['timestamp']}] {entry['role']}: {entry['message']}\n")

        # Save evaluation
        with open(evaluation_file, "w", encoding="utf-8") as f:
            f.write("=== EVALUATION REPORT ===\n\n")
            f.write(evaluation_report)

        logger.info(f"Transcript saved to: {transcript_file}")
        logger.info(f"Evaluation saved to: {evaluation_file}")

        # Save evaluation result in JSON format for API retrieval (keyed by room name)
        try:
            room_name = "unknown"
            if hasattr(ctx, "room") and ctx.room:
                try:
                    room_name = ctx.room.name
                except Exception:
                    # Room might be closed, use fallback
                    room_name = "unknown"
            evaluation_data = {
                "room": room_name,
                "timestamp": timestamp,
                "evaluation": evaluation_report,
                "transcript": conversation_transcript,
            }
            evaluation_json_file = f"evaluation_{room_name}_{timestamp}.json"
            with open(evaluation_json_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation JSON saved to: {evaluation_json_file}")
        except Exception as e:
            logger.warning(f"Could not save evaluation JSON: {e}")

        # Also log the evaluation
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION REPORT")
        logger.info("=" * 80)
        logger.info(evaluation_report)
        logger.info("=" * 80)
    else:
        logger.warning("No conversation transcript found. Skipping evaluation.")


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

        # Track messages and detect language from session's chat context
        # We'll extract messages from the session after it's started
        async def extract_transcript_periodically():
            """Periodically extract transcript from session and detect language"""
            global detected_language
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                try:
                    # Access session's chat context if available
                    if hasattr(session, "chat_ctx") and session.chat_ctx:
                        messages = session.chat_ctx.messages
                        # Update transcript with new messages
                        for msg in messages:
                            role = "Salesperson" if msg.role == "user" else "Customer"
                            content = (
                                msg.content if hasattr(msg, "content") else str(msg)
                            )

                            # Detect language from user messages (salesperson)
                            if role == "Salesperson" and content:
                                # Simple language detection based on script
                                if any(
                                    ord(char) >= 0x0900 and ord(char) <= 0x097F
                                    for char in content
                                ):
                                    # Devanagari script (Hindi, Marathi, etc.)
                                    detected_language = "hi"
                                    logger.info("Detected language: Hindi")
                                elif any(
                                    ord(char) >= 0x0D00 and ord(char) <= 0x0D7F
                                    for char in content
                                ):
                                    # Malayalam script
                                    detected_language = "ml"
                                    logger.info("Detected language: Malayalam")
                                elif any(
                                    ord(char) >= 0x0B80 and ord(char) <= 0x0BFF
                                    for char in content
                                ):
                                    # Tamil script
                                    detected_language = "ta"
                                    logger.info("Detected language: Tamil")
                                elif any(
                                    ord(char) >= 0x0C00 and ord(char) <= 0x0C7F
                                    for char in content
                                ):
                                    # Telugu script
                                    detected_language = "te"
                                    logger.info("Detected language: Telugu")
                                elif any(
                                    ord(char) >= 0x0C80 and ord(char) <= 0x0CFF
                                    for char in content
                                ):
                                    # Kannada script
                                    detected_language = "kn"
                                    logger.info("Detected language: Kannada")
                                else:
                                    # Default to English
                                    detected_language = "en"

                            # Only add if not already tracked
                            if not any(
                                t["message"] == content and t["role"] == role
                                for t in conversation_transcript
                            ):
                                track_message(
                                    role,
                                    content,
                                    room=ctx.room if hasattr(ctx, "room") else None,
                                )
                except Exception as e:
                    logger.debug(f"Error extracting transcript: {e}")

        # Start transcript extraction task
        transcript_task = asyncio.create_task(extract_transcript_periodically())

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

        # Start agent session
        try:
            logger.info("Starting agent session")
            await session.start(room=ctx.room, agent=agent)
            logger.info("Agent session started successfully")
        except Exception as e:
            logger.error(f"Error starting agent session: {e}")
            raise

        # Start conversation duration monitor
        monitor_task = asyncio.create_task(monitor_conversation_duration(ctx, session))

        logger.info(
            "All sessions started successfully. Conversation will run for 5 minutes."
        )

        # Wait for monitor to complete (this will also cancel transcript_task and disconnect_monitor_task)
        try:
            await monitor_task
        finally:
            transcript_task.cancel()
            disconnect_monitor_task.cancel()
            try:
                await transcript_task
            except asyncio.CancelledError:
                pass
            try:
                await disconnect_monitor_task
            except asyncio.CancelledError:
                pass

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
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
