import asyncio
import os
import sys
import pyaudio
from google import genai
from google.genai import types

# Audio settings for the Live API
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512


async def audio_playback_task(session, p):
    """Receive audio from Gemini and play it back through speakers."""
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    try:
        async for response in session.receive():
            if response.server_content is not None:
                model_turn = response.server_content.model_turn
                if model_turn:
                    for part in model_turn.parts:
                        # If the part is audio data, play it
                        if part.inline_data:
                            stream.write(part.inline_data.data)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()


async def audio_capture_task(session, p):
    """Capture audio from the microphone and stream it to Gemini."""
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    try:
        while True:
            # Read a chunk of audio from the microphone
            data = stream.read(CHUNK, exception_on_overflow=False)

            # Send audio data to the Live API
            await session.send_realtime_input(
                audio={"data": data, "mime_type": "audio/pcm;rate=16000"}
            )

            # Small sleep to yield control back to the event loop
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error capturing audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please run: export GEMINI_API_KEY='your_google_ai_studio_api_key'")
        sys.exit(1)

    print("Initializing Gemini Client...")
    client = genai.Client(api_key=api_key)

    # The Multimodal Live API requires the gemini-2.0-flash model
    model = "gemini-2.5-flash-native-audio-preview-12-2025"

    # Force the response to be AUDIO (it will speak back to us)
    config = {"response_modalities": ["AUDIO"]}

    p = pyaudio.PyAudio()

    print(f"Connecting to {model}...")
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            print("\n" + "=" * 50)
            print("Connected! Start speaking in Sinhala or English.")
            print("Press Ctrl+C to stop.")
            print("=" * 50 + "\n")

            # 1. Send an initial system prompt to configure the agent's persona
            await session.send_client_content(
                turns={
                    "parts": [
                        {
                            "text": "You are a warm, helpful customer service agent for Winrich. Speak primarily in natural Sinhala, but understand both Sinhala and English. Give concise responses."
                        }
                    ]
                },
                turn_complete=True,
            )

            # 2. Start streaming microphone audio up and streaming speaker audio down
            capture_task = asyncio.create_task(audio_capture_task(session, p))
            playback_task = asyncio.create_task(audio_playback_task(session, p))

            try:
                await asyncio.gather(capture_task, playback_task)
            except asyncio.CancelledError:
                pass
            finally:
                capture_task.cancel()
                playback_task.cancel()
                await asyncio.gather(
                    capture_task, playback_task, return_exceptions=True
                )

    except KeyboardInterrupt:
        print("\nDisconnecting...")
    except Exception as e:
        print(f"\nConnection failed: {e}")
    finally:
        p.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")
