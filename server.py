"""
Thiva Voice Agent Server
Multilingual voice agent using Twilio ConversationRelay, Claude, and Azure Speech.
Supports Sinhala, English, and Tamil via DTMF language selection.
Integrates with eZee PMS for room availability and booking.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import date

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from anthropic import Anthropic

from knowledge_base import retrieve_context, initialize_kb
from tools import get_tools, execute_tool

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize knowledge base on startup."""
    docs_dir = os.environ.get("KB_DOCS_DIRECTORY", "knowledge_docs")
    initialize_kb(docs_dir)
    yield

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Thiva Voice Agent", lifespan=lifespan)

# Environment variables
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# Language configurations keyed by DTMF digit
LANGUAGE_CONFIGS = {
    "1": {
        "name": "Sinhala",
        "voice": "si-LK-ThiliniNeural",
        "lang_code": "si-LK",
        "greeting": "සිංහල භාෂාව තෝරා ගන්නා ලදී.",
    },
    "2": {
        "name": "English",
        "voice": "en-US-JennyNeural",
        "lang_code": "en-US",
        "greeting": "English language selected.",
    },
    "3": {
        "name": "Tamil",
        "voice": "ta-IN-PallaviNeural",
        "lang_code": "ta-IN",
        "greeting": "தமிழ் மொழி தேர்ந்தெடுக்கப்பட்டது.",
    },
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a warm, helpful hotel booking agent. "
    "You speak ONLY in {language}. "
    "Today's date is {today}. "
    "You help guests check room availability and make reservations. "
    "When a guest wants to book, collect: dates, name, and phone number. "
    "Always check availability before creating a booking. "
    "Confirm all details with the guest before finalizing a booking. "
    "Be concise - keep responses under 2 sentences for voice. "
    "Use the provided knowledge base context to answer questions accurately."
)


def build_twiml(body: str) -> Response:
    """Wrap TwiML body in a proper XML response."""
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>'
    return Response(content=xml, media_type="application/xml")


# --- HTTP Endpoints ---


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    """Twilio webhook for incoming calls. Greets caller and collects language choice."""
    gather_body = (
        '<Say language="si-LK">ආයුබෝවන්! සිංහල සඳහා එක බොත්තම ඔබන්න.</Say>'
        '<Say language="en-US">Press 2 for English.</Say>'
        '<Say language="ta-IN">தமிழுக்கு மூன்றை அழுத்தவும்.</Say>'
    )
    twiml = (
        f'<Gather numDigits="1" action="/voice/handle-language" method="POST" timeout="5">'
        f"{gather_body}"
        f"</Gather>"
        # Default to Sinhala on timeout
        '<Redirect method="POST">/voice/handle-language?Digits=1</Redirect>'
    )
    return build_twiml(twiml)


@app.post("/voice/handle-language")
async def handle_language(request: Request):
    """Handles DTMF result and connects caller to ConversationRelay WebSocket."""
    form = await request.form()
    digit = form.get("Digits", "1")

    config = LANGUAGE_CONFIGS.get(digit, LANGUAGE_CONFIGS["1"])
    logger.info(f"Language selected: {config['name']} (digit={digit})")

    # Determine WebSocket URL from the request host
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = "wss" if request.headers.get("x-forwarded-proto", "http") == "https" else "ws"
    ws_url = f"{scheme}://{host}/ws/conversation"

    twiml = (
        "<Connect>"
        f'<ConversationRelay url="{ws_url}" '
        f'ttsProvider="azure" '
        f'voice="{config["voice"]}" '
        f'language="{config["lang_code"]}" '
        f'azureSpeechKey="{AZURE_SPEECH_KEY}" '
        f'azureSpeechRegion="{AZURE_SPEECH_REGION}" '
        f'welcomeGreeting="{config["greeting"]}"'
        ">"
        f'<Parameter name="selected_language" value="{config["name"]}" />'
        f'<Parameter name="lang_code" value="{config["lang_code"]}" />'
        "</ConversationRelay>"
        "</Connect>"
    )
    return build_twiml(twiml)


# --- WebSocket Endpoint ---


@app.websocket("/ws/conversation")
async def ws_conversation(websocket: WebSocket):
    """ConversationRelay WebSocket handler for bidirectional text conversation."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    conversation_history: list[dict] = []
    system_prompt = ""
    language_name = "Sinhala"
    tools = get_tools()

    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type")

            # Setup message: extract language config from ConversationRelay
            if msg_type == "setup":
                params = message.get("customParameters", {})
                language_name = params.get("selected_language", "Sinhala")
                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    language=language_name, today=date.today().isoformat()
                )
                logger.info(f"Session initialized: language={language_name}")
                continue

            # User speech transcription
            if msg_type == "prompt":
                user_text = message.get("voicePrompt", "").strip()
                if not user_text:
                    continue

                logger.info(f"User ({language_name}): {user_text}")

                # Retrieve relevant knowledge base context
                kb_context = retrieve_context(user_text)

                # Build messages with KB context
                augmented_system = system_prompt
                if kb_context:
                    augmented_system += (
                        f"\n\n<knowledge_base>\n{kb_context}\n</knowledge_base>"
                    )

                conversation_history.append({"role": "user", "content": user_text})

                # Call Claude in a loop to handle tool use
                full_response = await _run_claude_with_tools(
                    client,
                    augmented_system,
                    conversation_history,
                    tools,
                    websocket,
                    language_name,
                )

                logger.info(f"Agent ({language_name}): {full_response[:100]}...")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(
            f"Session ended: {len(conversation_history) // 2} exchanges"
        )


async def _run_claude_with_tools(
    client: Anthropic,
    system: str,
    conversation_history: list[dict],
    tools: list[dict],
    websocket: WebSocket,
    language_name: str,
) -> str:
    """
    Run Claude with tool use support. Loops until Claude produces a final text
    response (no more tool calls). Streams the final text to the WebSocket.
    """
    max_tool_rounds = 5  # Safety limit

    for _ in range(max_tool_rounds):
        # Build API kwargs — only include tools if available
        api_kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": system,
            "messages": conversation_history,
        }
        if tools:
            api_kwargs["tools"] = tools

        # Non-streaming call when tools might be invoked
        response = client.messages.create(**api_kwargs)

        # Check if Claude wants to use tools
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if tool_use_blocks:
            # Send any interim text to the caller (e.g. "Let me check that for you.")
            interim_text = " ".join(b.text for b in text_blocks)
            if interim_text.strip():
                for word in interim_text.split():
                    await websocket.send_text(
                        json.dumps({"type": "text", "token": word + " "})
                    )

            # Add assistant message with all content blocks to history
            conversation_history.append({
                "role": "assistant",
                "content": [b.model_dump() for b in response.content],
            })

            # Execute each tool and collect results
            tool_results = []
            for tool_block in tool_use_blocks:
                logger.info(
                    f"Tool call: {tool_block.name}({tool_block.input})"
                )
                result_str = await execute_tool(
                    tool_block.name, tool_block.input
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_str,
                })

            # Add tool results to history and loop back
            conversation_history.append({
                "role": "user",
                "content": tool_results,
            })
            continue

        # No tool calls — stream the final text response
        # Re-do as streaming call for smooth token delivery
        full_response = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=conversation_history,
        ) as stream:
            for token in stream.text_stream:
                full_response += token
                await websocket.send_text(
                    json.dumps({"type": "text", "token": token})
                )

        # Signal end of response
        await websocket.send_text(
            json.dumps({"type": "text", "token": "", "last": True})
        )

        conversation_history.append(
            {"role": "assistant", "content": full_response}
        )
        return full_response

    # Fallback if max tool rounds exceeded
    fallback = "I'm sorry, I had trouble processing that. Could you try again?"
    await websocket.send_text(
        json.dumps({"type": "text", "token": fallback, "last": True})
    )
    conversation_history.append({"role": "assistant", "content": fallback})
    return fallback


# --- Health Check ---


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "thiva-voice-agent",
        "languages": ["si-LK", "en-US", "ta-IN"],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
