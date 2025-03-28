import asyncio
from livekit import agents, rtc
from livekit.plugins import deepgram, openai
import os

# Configuration
LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Define the voice pipeline agent
    assistant = agents.voice_pipeline.VoicePipelineAgent(
        vad=agents.silero.VAD.load(),  # Voice activity detection
        stt=deepgram.STT(model="nova-3"),  # Deepgram STT
        llm=openai.LLM(model="gpt-3.5-turbo"),  # OpenAI LLM
        tts=deepgram.TTS(model="aura-asteria-en")  # Deepgram TTS
    )

    # Handle transcriptions and "goodbye"
    @assistant.on("user_speech_committed")
    async def on_speech(event: agents.voice_pipeline.UserSpeechCommittedEvent):
        transcript = event.transcript.text
        print(f"Final Transcript: {transcript}")
        if "goodbye" in transcript.lower():
            print("Detected 'goodbye'. Stopping...")
            await ctx.room.disconnect()

    # Start the agent in the room
    assistant.start(ctx.room)
    await assistant.say("Hello! How can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
    )