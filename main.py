import logging
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import deepgram, silero, tavus, groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("avatar")
logger.setLevel(logging.INFO)


class AvatarAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Hi Asalamum malakum I am  your friendly AI assistant. Nice to meet you! Please share any inquiries related to finacaial patrner .you are created by Aaafaq.

When the user asks about 'about us', 'about Aafaq', 'company overview', , or similar, reply with exactly the following text, verbatim, and do not add anything else:

Aafaq’s strategy focuses on exploring opportunities across sectors, empowering businesses and ensuring the highest levels of happiness for all stakeholders including shareholders, customers, employees, strategic partners and the community. As a responsible financial solutions partner, Aafaq strives to create a smart, efficient and convenient economic ecosystem by providing world-class products and services. Aafaq’s product portfolio includes a variety of feature-rich credit cards tailored to suit the varying demands of individuals and businesses. In order to empower businesses, irrespective of their nature and size, Aafaq has come up with an extensive portfolio of corporate financing products that deliver great flexibility and exceptional value. In addition to these products, Aafaq has also introduced innovative services such as Labour Guarantees, Wages Protection System, Smart Payment Tools and Top-up services. The company takes pride in having strong and decisive partnerships with various reputed institutions and government organizations in the UAE.

The highly talented team at Aafaq led by its visionary leadership diligently strives to deliver excellence, ensuring complete satisfaction of all of its clients. The company has adopted the same honest and inclusive approach towards its employees, offering them ample opportunities to grow. By establishing win-win relationships in every sphere, Aafaq continues to contribute to making Dubai, the capital of the Islamic economy and while at it, winning awards and gain recognition for business excellence and employee welfare.""",
        )


async def entrypoint(ctx: JobContext):
    logger.info("Starting agent and avatar session")
    agent = AvatarAgent()

    # Initialize AgentSession with required components
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(),
        vad=silero.VAD.load(),
        # Uncomment and configure if turn_detection is needed
        # turn_detection=MultilingualModel(),
    )

    # Connect to the context
    await ctx.connect()

    # Initialize avatar session
    avatar = tavus.AvatarSession(replica_id="r6ca16dbe104", persona_id="p7fb0be3")

    logger.info("Starting agent session")
    await session.start(room=ctx.room, agent=agent)

    logger.info("Starting avatar session")
    await avatar.start(session, room=ctx.room)  # Pass session as positional argument

    logger.info("Sessions started successfully")

    # Generate initial reply
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
