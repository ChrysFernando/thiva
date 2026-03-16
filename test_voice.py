"""
Local voice test for the Thiva agent.
Type your messages and hear the agent's spoken response via Azure TTS.
Same Claude + knowledge-base pipeline as the live server.

Usage:
    python test_voice.py
"""

import os
import sys

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

# Language configs (same as server.py)
LANGUAGES = {
    "1": {"name": "Sinhala", "voice": "si-LK-ThiliniNeural", "lang_code": "si-LK"},
    "2": {"name": "English", "voice": "en-US-JennyNeural", "lang_code": "en-US"},
    "3": {"name": "Tamil", "voice": "ta-IN-PallaviNeural", "lang_code": "ta-IN"},
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a warm, helpful customer service agent for Winrich. "
    "You speak ONLY in {language}. "
    "You help customers with product inquiries, orders, complaints, and general questions. "
    "Be concise - keep responses under 2 sentences for voice. "
    "Use the provided knowledge base context to answer questions accurately."
)


def speak(text: str, voice: str, region: str, key: str):
    """Speak text using Azure TTS."""
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice

    # Use default speaker output
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        pass  # Success
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"  [TTS Error: {details.reason} - {details.error_details}]")


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

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)

    print(f"\nLanguage: {language} | Voice: {voice}")
    print("Type your messages below. Type 'quit' to exit.\n")
    print("-" * 50)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    conversation_history = []

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

        # Get Claude response
        print("Agent: ", end="", flush=True)
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=augmented_system,
            messages=conversation_history,
        ) as stream:
            full_response = ""
            for token in stream.text_stream:
                full_response += token
                print(token, end="", flush=True)

        print()  # newline after response
        conversation_history.append({"role": "assistant", "content": full_response})

        # Speak the response
        print("  [Speaking...]")
        speak(full_response, voice, AZURE_SPEECH_REGION, AZURE_SPEECH_KEY)


if __name__ == "__main__":
    main()
