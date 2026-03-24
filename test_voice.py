"""
Local voice test for the Thiva agent.
Type your messages and hear the agent's spoken response via Azure TTS.
Same Claude + knowledge-base + eZee tools pipeline as the live server.

Usage:
    python test_voice.py
"""

import os
import sys
import re
import json
import asyncio
import threading
import queue
from datetime import date

from dotenv import load_dotenv

load_dotenv()

# --- Check dependencies early ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")

if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in .env")
    sys.exit(1)
if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    print("ERROR: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
    sys.exit(1)

import azure.cognitiveservices.speech as speechsdk
from anthropic import Anthropic
from knowledge_base import retrieve_context, initialize_kb
from tools import get_tools, execute_tool

# Language configs (same as server.py)
LANGUAGES = {
    "1": {"name": "Sinhala", "voice": "si-LK-SameeraNeural", "lang_code": "si-LK"},
    "2": {"name": "English", "voice": "en-US-JennyNeural", "lang_code": "en-US"},
    "3": {"name": "Tamil", "voice": "ta-IN-PallaviNeural", "lang_code": "ta-IN"},
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


def create_synthesizer(voice: str, region: str, key: str):
    """Create a reusable Azure TTS synthesizer."""
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    return speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )


# Sentence-ending punctuation pattern (handles ., !, ?, and their unicode equivalents)
SENTENCE_END = re.compile(r'[.?]\s*')


def tts_worker(tts_queue: queue.Queue, synthesizer):
    """Background thread that speaks sentences from the queue in order."""
    while True:
        chunk = tts_queue.get()
        if chunk is None:  # Poison pill — we're done
            break
        result = synthesizer.speak_text_async(chunk).get()
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            print(f"\n  [TTS Error: {details.reason} - {details.error_details}]")
        tts_queue.task_done()


def run_tool_loop(client, augmented_system, conversation_history, tools):
    """
    Run Claude with tool use. Handles tool calls synchronously, then
    streams the final text response. Returns the final text.
    """
    max_tool_rounds = 5

    for _ in range(max_tool_rounds):
        api_kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": augmented_system,
            "messages": conversation_history,
        }
        if tools:
            api_kwargs["tools"] = tools

        response = client.messages.create(**api_kwargs)

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if tool_use_blocks:
            # Print any interim text
            interim = " ".join(b.text for b in text_blocks)
            if interim.strip():
                print(interim, end=" ", flush=True)

            # Add assistant message to history
            conversation_history.append({
                "role": "assistant",
                "content": [b.model_dump() for b in response.content],
            })

            # Execute tools (async → run in event loop)
            tool_results = []
            for tb in tool_use_blocks:
                print(f"\n  [Calling {tb.name}...]", flush=True)
                result_str = asyncio.get_event_loop().run_until_complete(
                    execute_tool(tb.name, tb.input)
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": result_str,
                })

            conversation_history.append({
                "role": "user",
                "content": tool_results,
            })
            continue

        # No tool calls — return text for streaming
        return None  # Signal to caller: stream the final response

    return None


def main():
    # Initialize knowledge base
    docs_dir = os.environ.get("KB_DOCS_DIRECTORY", "knowledge_docs")
    initialize_kb(docs_dir)

    # Language selection
    print("\n" + "=" * 50)
    print("  THIVA VOICE AGENT - Local Test")
    print("=" * 50)
    print("\nSelect language:")
    print("  1 - Sinhala")
    print("  2 - English")
    print("  3 - Tamil")

    choice = input("\nEnter 1, 2, or 3: ").strip()
    if choice not in LANGUAGES:
        print("Invalid choice, defaulting to Sinhala.")
        choice = "1"

    config = LANGUAGES[choice]
    language = config["name"]
    voice = config["voice"]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        language=language, today=date.today().isoformat()
    )
    tools = get_tools()

    if tools:
        print(f"\nLanguage: {language} | Voice: {voice} | eZee: connected")
    else:
        print(f"\nLanguage: {language} | Voice: {voice} | eZee: not configured")
    print("Type your messages below. Type 'quit' to exit.\n")
    print("-" * 50)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    conversation_history = []

    # Create a reusable synthesizer (avoids re-creating it every turn)
    synthesizer = create_synthesizer(voice, AZURE_SPEECH_REGION, AZURE_SPEECH_KEY)

    while True:
        user_input = input(f"\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Retrieve KB context
        kb_context = retrieve_context(user_input)
        augmented_system = system_prompt
        if kb_context:
            augmented_system += f"\n\n<knowledge_base>\n{kb_context}\n</knowledge_base>"

        conversation_history.append({"role": "user", "content": user_input})

        print("Agent: ", end="", flush=True)

        # Run tool loop (handles tool calls, adds to history)
        run_tool_loop(client, augmented_system, conversation_history, tools)

        # Set up background TTS queue — speaks sentences as they arrive
        tts_queue = queue.Queue()
        tts_thread = threading.Thread(
            target=tts_worker, args=(tts_queue, synthesizer), daemon=True
        )
        tts_thread.start()

        # Stream the final text response
        api_kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": augmented_system,
            "messages": conversation_history,
        }
        if tools:
            api_kwargs["tools"] = tools

        sentence_buffer = ""
        with client.messages.stream(**api_kwargs) as stream:
            full_response = ""
            for token in stream.text_stream:
                full_response += token
                sentence_buffer += token
                print(token, end="", flush=True)

                # Check if buffer contains a complete sentence
                if SENTENCE_END.search(sentence_buffer):
                    tts_queue.put(sentence_buffer.strip())
                    sentence_buffer = ""

        # Send any remaining text that didn't end with punctuation
        if sentence_buffer.strip():
            tts_queue.put(sentence_buffer.strip())

        print()  # newline after response
        conversation_history.append({"role": "assistant", "content": full_response})

        # Wait for all sentences to finish speaking, then clean up
        tts_queue.put(None)  # Signal worker to exit
        tts_thread.join()


if __name__ == "__main__":
    main()
