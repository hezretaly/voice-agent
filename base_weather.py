import logging
import requests
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv(dotenv_path=".env.local")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Jenna. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(instructions="greet the user and ask about their day")

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
        latitude: str,
        longitude: str,
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """

        logger.info(f"Looking up weather for {location}")

        response = requests.post(os.environ.get("N8N_WEBHOOK_URL"), json={"location": location}, timeout=5)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "response": "Some error happened"
            }
        
    # agent tool
    @function_tool
    async def agent_request(
        self,
        context: RunContext,
        query: str,
    ):
        """Called when the user asks to send email or slack message or telegram message.
        Ensure the user specifies what the message or email to be sent.
        This agent is responsible for sending emails, slack and telegram messages so make the querry clear enough for agent to know what to send and whoom to send it.

        Args:
            query: The querry for the agent to process.
        """

        logger.info(f"Qerrying agent for {query}")

        response = requests.post(os.environ.get("N8N_WEBHOOK_URL_SEYD"), json={"query": query}, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "response": "Some error happened"
            }



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "user_id",
    }
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-2"),
        # tts=deepgram.TTS(model="aura-asteria-en"),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))