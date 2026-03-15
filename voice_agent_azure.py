import os
import sys
import time
import azure.cognitiveservices.speech as speechsdk
from google import genai
import socket

# Force IPv4 to prevent IPv6 DNS timeouts
old_getaddrinfo = socket.getaddrinfo


def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]


socket.getaddrinfo = new_getaddrinfo


def main():
    # 1. API Keys & Setup
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    azure_speech_key = os.environ.get("AZURE_SPEECH_KEY")
    azure_region = os.environ.get("AZURE_REGION")  # e.g., "eastus"

    if not all([gemini_api_key, azure_speech_key, azure_region]):
        print("Error: Missing environment variables for Gemini or Azure.")
        sys.exit(1)

    print("Initializing Gemini Client & Azure Speech...")

    # Use standard Gemini for text reasoning
    gemini_client = genai.Client(api_key=gemini_api_key)
    chat = gemini_client.chats.create(
        model="gemini-2.5-flash",
        config={
            "system_instruction": "You are a warm, helpful customer service agent for Winrich. Speak primarily in natural Sinhala, but understand both Sinhala and English. Give concise responses."
        },
    )

    # 2. Configure Azure Speech
    speech_config = speechsdk.SpeechConfig(
        subscription=azure_speech_key, region=azure_region
    )

    # Set the Speech-to-Text language (listening)
    speech_config.speech_recognition_language = "si-LK"

    # Set the Text-to-Speech voice (speaking - Sameera is male, Thilini is female)
    speech_config.speech_synthesis_voice_name = "si-LK-SameeraNeural"

    # Azure automatically uses your default microphone and speakers
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    print("\n" + "=" * 50)
    print("Inbound Receptionist Active.")
    print("Start speaking in Sinhala. Say 'Stop' or 'Exit' to end.")
    print("=" * 50 + "\n")

    try:
        while True:
            print("\nListening...")
            result = speech_recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                user_text = result.text
                print(f"Customer: {user_text}")

                if "stop" in user_text.lower() or "exit" in user_text.lower():
                    print("Shutting down receptionist.")
                    break

                # --- GEMINI TIMER ---
                print("Agent is thinking... (Pinging Gemini API)")
                gemini_start = time.time()

                response = chat.send_message(user_text)
                agent_reply = response.text

                gemini_end = time.time()
                print(
                    f"--> Gemini replied in {gemini_end - gemini_start:.2f} seconds."
                )
                print(f"Agent Text: {agent_reply}")

                # --- AZURE TIMER ---
                print("Generating and playing audio... (Pinging Azure API)")
                azure_start = time.time()

                speech_synthesizer.speak_text_async(agent_reply).get()

                azure_end = time.time()
                print(
                    f"--> Azure finished playing in {azure_end - azure_start:.2f} seconds."
                )

            elif result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized. Try again.")

    except KeyboardInterrupt:
        print("\nExited manually.")


if __name__ == "__main__":
    main()
