import logging
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
import os
import aiohttp

load_dotenv(dotenv_path=".env.local")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("voice-agent")

OPENROUTER_API_KEY=os.environ["OPENROUTER_API_KEY"]

# llm with n8n request
class ChatResponse:
    def __init__(self, message):
        self.message = message

async def fetch_data(url, params=None):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to fetch data: {response.status}"}
        except Exception as e:
            return {"error": f"Web request error: {str(e)}"}

class WebEnhancedOpenAILLM(openai.LLM):
    def __init__(self, api_endpoint="https://runwayngihts.app.n8n.cloud/webhook-test/4f7c69f6-fe17-4927-adcf-6117f916dfd8", *args, **kwargs):
        super().__init__(base_url="https://openrouter.ai/api/v1/", model="google/gemini-2.0-flash-exp:free", api_key=OPENROUTER_API_KEY, **kwargs)
        self.web_api_endpoint = api_endpoint

    async def chat(self, chat_ctx: llm.ChatContext, fnc_ctx=None) -> ChatResponse:
        user_message = chat_ctx.messages[-1].text if chat_ctx.messages else ""
        
        # Fetch external data
        external_data = await fetch_data("https://runwayngihts.app.n8n.cloud/webhook-test/4f7c69f6-fe17-4927-adcf-6117f916dfd8", {"query": user_message})
        
        # Modify chat context with fetched data
        enhanced_ctx = chat_ctx.copy()
        enhanced_ctx.messages.append(
            llm.ChatMessage(
                role="system",
                text=f"Additional info from web: {external_data.get('result', 'No data found')}"
            )
        )
        
        # Generate response using the enhanced context
        openai_response = await super().chat(enhanced_ctx, fnc_ctx=fnc_ctx)
        
        # Optionally modify the response further
        final_text = f"{openai_response.message.text} (Source: Web data)"
        return ChatResponse(
            message=llm.ChatMessage(role="assistant", text=final_text)
        )


# class WebEnhancedOpenAILLM(openai.LLM):
#     def __init__(self, api_endpoint="https://runwayngihts.app.n8n.cloud/webhook-test/4f7c69f6-fe17-4927-adcf-6117f916dfd8", api_key=None, *args, **kwargs):
#         # Pass all original arguments to the parent OpenAI LLM class
#         super().__init__(*args, **kwargs)
#         self.web_api_endpoint = api_endpoint
#         self.web_api_key = api_key

#     async def chat(self, chat_ctx: llm.ChatContext) -> llm.ChatResponse:
#         # Extract the latest user message
#         user_message = chat_ctx.messages[-1].text if chat_ctx.messages else ""

#         # Step 1: Make a web request to fetch external data
#         external_data = await self._fetch_web_data(user_message)

#         # Step 2: Modify the chat context with the fetched data
#         enhanced_ctx = chat_ctx.copy()
#         enhanced_ctx.messages.append(
#             llm.ChatMessage(
#                 role="system",
#                 text=f"Additional info from web: {external_data}"
#             )
#         )

#         # Step 3: Call the parent OpenAI LLM's chat method with the enhanced context
#         openai_response = await super().chat(enhanced_ctx)

#         # Step 4: Optionally modify the response further
#         final_text = f"{openai_response.message.text} (Source: Web data)"
#         return llm.ChatResponse(
#             message=llm.ChatMessage(role="assistant", text=final_text)
#         )

#     async def _fetch_web_data(self, query: str) -> str:
#         """Intermediary function to make an async web request."""
#         headers = {"Authorization": f"Bearer {self.web_api_key}"} if self.web_api_key else {}
#         params = {"query": query}

#         async with aiohttp.ClientSession() as session:
#             try:
#                 async with session.get(self.web_api_endpoint, headers=headers, params=params) as resp:
#                     if resp.status == 200:
#                         data = await resp.json()
#                         return data.get("result", "No data found")
#                     else:
#                         return f"Web request failed: HTTP {resp.status}"
#             except Exception as e:
#                 return f"Web request error: {str(e)}"
            
# Initialize the enhanced LLM
web_enhanced_llm = WebEnhancedOpenAILLM()


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
        # llm=openai.LLM(
        #     base_url="https://openrouter.ai/api/v1/",
        #     model="google/gemini-2.0-flash-exp:free",
        #     api_key=OPENROUTER_API_KEY,
        # ),   # Openrouter.ai LLM
        llm=web_enhanced_llm,
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

