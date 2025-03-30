import asyncio
from livekit import agents, rtc
from livekit.plugins import deepgram, openai
import os

# Configuration (set via environment variables or hardcode for simplicity)
LIVEKIT_PORT = os.environ["LIVEKIT_PORT"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Voice pipeline agent
    assistant = agents.voice_pipeline.VoicePipelineAgent(
        vad=agents.silero.VAD.load(),
        stt=deepgram.STT(model="base", api_key=DEEPGRAM_API_KEY),
        llm=openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
        tts=deepgram.TTS(model="aura-asteria-en", api_key=DEEPGRAM_API_KEY)
    )

    @assistant.on("user_speech_committed")
    async def on_speech(event: agents.voice_pipeline.UserSpeechCommittedEvent):
        transcript = event.transcript.text
        print(f"Transcript: {transcript}")
        if "goodbye" in transcript.lower():
            print("Detected 'goodbye'. Disconnecting...")
            await ctx.room.disconnect()

    assistant.start(ctx.room)
    await assistant.say("Hello! How can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            url=f"ws://0.0.0.0:{LIVEKIT_PORT}",
            api_key=os.environ["API_KEY"], 
            api_secret=os.environ["API_SECRET"]
        )
    )