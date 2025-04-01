import logging
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

load_dotenv(dotenv_path=".env.local")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You are a voice assistant. Use short, concise responses."
    )

    logger.info(f"attempting to connect to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"successfully connected to room {ctx.room.name}")

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],           # Required VAD
        stt=deepgram.STT(model="base"),       # Lightweight STT model
        llm=openai.LLM(model="gpt-4o-mini"),   # OpenAI LLM
        tts=deepgram.TTS(),                     # Deepgram TTS
        chat_ctx=initial_ctx                    # Initial LLM context
    )

    agent.start(ctx.room, participant)
    await agent.say("Hello, I’m here to help!", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
