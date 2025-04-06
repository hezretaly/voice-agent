import asyncio
import time
import uuid
import aiohttp
import os
import json
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    llm as llm_base,
    VoicePipelineAgent
)
from livekit.agents.llm import LLMStream, LLMStreamChunk
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
    turn_detector,
)

# Setup logging and load environment variables
logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
load_dotenv(dotenv_path=".env.local")

# --- Tool Functions ---

async def search_weather(location: str):
    """Fetches current weather information using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OpenWeatherMap API key not found")
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch weather data: {response.status}")
            data = await response.json()
            try:
                weather_desc = data['weather'][0]['description']
                temp = data['main']['temp']
                result = f"The weather in {location} is {weather_desc} with a temperature of {temp} degrees Celsius."
                print(f"[{time.strftime('%H:%M:%S')} TOOL DONE]: search_weather for {location}")
                return result
            except KeyError:
                raise Exception("Invalid weather data received")

async def internet_search(query: str):
    """Performs an internet search using SerpStack API."""
    api_key = os.getenv("SERPSTACK_API_KEY")
    if not api_key:
        raise ValueError("SerpStack API key not found")
    
    url = f"http://api.serpstack.com/search?access_key={api_key}&query={query}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to perform search: {response.status}")
            data = await response.json()
            try:
                results = data['organic_results']
                if results:
                    top_result = results[0]
                    title = top_result['title']
                    snippet = top_result.get('snippet', '')
                    result = f"The top search result for '{query}' is '{title}': {snippet}"
                else:
                    result = f"No search results found for '{query}'."
                print(f"[{time.strftime('%H:%M:%S')} TOOL DONE]: internet_search for '{query}'")
                return result
            except KeyError:
                raise Exception("Invalid search data received")

async def send_to_n8n(message: str):
    """Sends a message to an n8n workflow via a webhook."""
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("n8n webhook URL not found")
    
    payload = {"message": message}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Failed to send message to n8n: {response.status}")
            print(f"[{time.strftime('%H:%M:%S')} TOOL DONE]: send_to_n8n with message '{message}'")
            return "Message sent to n8n successfully."

# Mapping function names to the actual async functions
available_functions = {
    "search_weather": search_weather,
    "internet_search": internet_search,
    "send_to_n8n": send_to_n8n,
}

# --- Define the Function Schema for the LLM ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internet_search",
            "description": "Search the internet for information based on a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_n8n",
            "description": "Send a message to an n8n workflow",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send",
                    },
                },
                "required": ["message"],
            },
        },
    }
]

# --- Custom LLM Wrapper ---

class FunctionCallingLLM(llm_base.LLM):
    """
    Wraps an LLM to add function calling, task management, and TTS feedback.
    """
    def __init__(self,
                 base_llm: llm_base.LLM,
                 agent: VoicePipelineAgent,
                 tools: Optional[list] = None,
                 available_functions: Optional[Dict[str, callable]] = None):
        super().__init__()
        self._base_llm = base_llm
        self._agent = agent
        self._tools = tools or []
        self._available_functions = available_functions or {}
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._active_task_id: Optional[str] = None

    async def _execute_function_and_update_state(self, task_id: str, function_name: str, arguments: dict):
        update_task = None
        try:
            func = self._available_functions.get(function_name)
            if not func:
                raise ValueError(f"Function '{function_name}' is not available.")
            
            update_task = asyncio.create_task(self._send_periodic_updates(task_id))
            self._tasks[task_id]["task_handle"] = update_task
            
            logger.info(f"Executing tool function: {function_name}")
            result = await func(**arguments)
            logger.info(f"Tool function {function_name} completed")
            
            if task_id in self._tasks and self._tasks[task_id]["status"] == "processing":
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["result"] = result
                await self._agent.say(str(result), allow_interruptions=True)
        except Exception as e:
            error_message = f"Sorry, I encountered an error when trying to {function_name.replace('_', ' ')}."
            logger.error(f"Task {task_id} failed - Function: {function_name}, Error: {e}", exc_info=True)
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["result"] = str(e)
                await self._agent.say(error_message, allow_interruptions=True)
        finally:
            if update_task and not update_task.done():
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
            if self._active_task_id == task_id:
                self._active_task_id = None

    async def _send_periodic_updates(self, task_id: str):
        update_interval_seconds = 7
        try:
            while True:
                await asyncio.sleep(update_interval_seconds)
                task_info = self._tasks.get(task_id)
                if task_info and task_info["status"] == "processing" and self._active_task_id == task_id:
                    await self._agent.say("Just letting you know, I'm still working on that...", allow_interruptions=True)
                else:
                    break
        except asyncio.CancelledError:
            logger.info(f"Periodic updates cancelled for task {task_id}.")

    async def chat(self, history: llm_base.ChatContext, **kwargs) -> llm_base.LLMResponse:
        if self._active_task_id and self._tasks.get(self._active_task_id, {}).get("status") == "processing":
            response = llm_base.LLMResponse()
            response.choices.append(llm_base.LLMChoice(
                message=llm_base.ChatMessage(role="assistant", content="I'm still working on your previous request. Please wait a moment.")
            ))
            return response

        if self._tools:
            kwargs['tools'] = self._tools
            kwargs['tool_choice'] = "auto"

        llm_response = await self._base_llm.chat(history, **kwargs)
        tool_calls = llm_response.choices[0].message.tool_calls if llm_response.choices and llm_response.choices[0].message.tool_calls else None

        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                response = llm_base.LLMResponse()
                response.choices.append(llm_base.LLMChoice(
                    message=llm_base.ChatMessage(role="assistant", content=f"Sorry, I couldn't understand the details needed for {function_name}.")
                ))
                return response

            if function_name in self._available_functions:
                task_id = f"task_{function_name}_{uuid.uuid4()}"
                self._tasks[task_id] = {
                    "status": "processing",
                    "result": None,
                    "function_name": function_name,
                    "task_handle": None
                }
                self._active_task_id = task_id

                ack_message = f"Okay, I will {function_name.replace('_', ' ')} for you."
                response = llm_base.LLMResponse()
                response.choices.append(llm_base.LLMChoice(
                    message=llm_base.ChatMessage(role="assistant", content=ack_message)
                ))

                asyncio.create_task(self._execute_function_and_update_state(task_id, function_name, arguments))
                return response
            else:
                response = llm_base.LLMResponse()
                response.choices.append(llm_base.LLMChoice(
                    message=llm_base.ChatMessage(role="assistant", content=f"Sorry, I don't have the capability to {function_name.replace('_', ' ')}.")
                ))
                return response
        return llm_response

    async def stream(self, history: llm_base.ChatContext, **kwargs) -> LLMStream:
        if self._active_task_id and self._tasks.get(self._active_task_id, {}).get("status") == "processing":
            async def _stream_wrapper():
                yield LLMStreamChunk(choice_delta=llm_base.LLMChoiceDelta(content="I'm still working on your previous request. Please wait a moment."))
            return LLMStream(stream=_stream_wrapper())

        llm_response = await self.chat(history, **kwargs)
        async def _response_to_stream(resp: llm_base.LLMResponse):
            if resp.choices and resp.choices[0].message.content:
                yield LLMStreamChunk(choice_delta=llm_base.LLMChoiceDelta(content=resp.choices[0].message.content))
                await asyncio.sleep(0)
        return LLMStream(stream=_response_to_stream(llm_response))

# --- Entrypoint and Worker Setup ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm_base.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "When you need to call a function/tool, respond ONLY with the initial acknowledgement "
            "(e.g., 'Okay, looking up the weather...'). The actual result will be spoken later by the system."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    base_llm = openai.LLM(model="gpt-4o-mini")
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model='nova-2'),
        tts=deepgram.TTS(model='aura-asteria-en'),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=2.0,
        noise_cancellation=noise_cancellation.BVC(),
        chat_ctx=initial_ctx.copy(),
    )

    function_llm = FunctionCallingLLM(
        base_llm=base_llm,
        agent=agent,
        tools=TOOLS,
        available_functions=available_functions
    )
    agent.llm = function_llm

    usage_collector = metrics.UsageCollector()
    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )