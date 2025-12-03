import logging
import os
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import deepgram, silero, tavus, groq
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


class AvatarAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Hi! You are Insura, a friendly insurance assistant created by CloudSubset. Greet people warmly, keep the tone conversational, and focus on explaining insurance concepts clearly.

Your primary role is to answer questions about insurance—especially medical and motor insurance—by breaking down terms, coverage options, exclusions, and claims steps in simple language. Keep responses practical, empathetic, and proactive about offering helpful tips or next steps.

When the user asks about "about us", "about CloudSubset", "company overview", or similar, reply with exactly the following text, verbatim, and do not add anything else:

CloudSubset is an insurance-focused innovator dedicated to making medical and motor coverage easier to understand and access. We blend deep industry expertise with human-friendly technology so every customer can pick the right protection, manage claims with confidence, and stay supported at every step. Whether you need comprehensive health cover or tailored motor plans, CloudSubset keeps insurance simple, transparent, and personal.""",
        )


async def entrypoint(ctx: JobContext):
    logger.info("Starting agent and avatar session")

    try:
        # Connect to the context first
        await ctx.connect()
        logger.info("Connected to LiveKit room")

        # Initialize the agent
        agent = AvatarAgent()
        logger.info("Agent initialized")

        # Initialize AgentSession with required components
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=deepgram.TTS(),
            vad=silero.VAD.load(),
            # Uncomment and configure if turn_detection is needed
            # turn_detection=MultilingualModel(),
        )
        logger.info("AgentSession initialized")

        # Initialize avatar session
        avatar = tavus.AvatarSession(replica_id="r6ca16dbe104", persona_id="p7fb0be3")
        logger.info("Avatar session initialized")

        # Start avatar session first
        logger.info("Starting avatar session")
        await avatar.start(session, room=ctx.room)

        # Start agent session
        logger.info("Starting agent session")
        await session.start(room=ctx.room, agent=agent)

        logger.info("All sessions started successfully")

    except Exception as e:
        logger.error(f"Error starting sessions: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
