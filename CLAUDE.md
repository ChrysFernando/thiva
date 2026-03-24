# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Thiva is a multilingual AI voice agent for hotel booking. It handles phone calls via Twilio, uses Claude for conversation with tool use (eZee PMS integration for availability & reservations), Azure Speech for TTS, and a ChromaDB-based RAG knowledge base.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the production server (Twilio + WebSocket)
python server.py

# Run local voice test (CLI input + Azure TTS output)
python test_voice.py

# Alternative voice agents (experimental)
python voice_agent_live.py     # Gemini Live API with native audio
python voice_agent_azure.py    # Gemini + Azure Speech
```

## Environment Setup

Copy `.env.example` to `.env` and fill in API keys. Required:
- `ANTHROPIC_API_KEY` — Claude API
- `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` — Azure Speech Services
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` — Twilio (production only)
- `EZEE_HOTEL_CODE`, `EZEE_AUTH_CODE`, `EZEE_API_KEY` — eZee PMS (for booking features)

## Architecture

**Production call flow** (`server.py`):
Incoming call → `/voice/incoming` (IVR greeting) → DTMF digit selects language → `/voice/handle-language` connects Twilio ConversationRelay → WebSocket `/ws/conversation` → Claude with tool use (eZee API calls for availability/booking) → streams final response token-by-token → ConversationRelay handles Azure TTS playback.

**Local test flow** (`test_voice.py`):
CLI language selection → typed input → KB retrieval → Claude tool use loop (eZee API) → streaming final response → sentence-by-sentence Azure TTS on background thread.

**eZee PMS integration** (`ezee_api.py` + `tools.py`):
Claude has two tools: `check_availability` (room search by date range) and `create_booking` (reserve rooms). Tools are defined in `tools.py`, API calls in `ezee_api.py`. Tools are only enabled when eZee credentials are configured. The tool loop in server.py supports up to 5 rounds of tool calls before streaming the final text response.

**Knowledge base** (`knowledge_base.py`):
Files in `knowledge_docs/` are chunked (500 chars, 50 overlap) → embedded with `all-MiniLM-L6-v2` → stored in ChromaDB (`./chroma_db/`). On each query, top 3 chunks are retrieved and injected into the system prompt as `<knowledge_base>` context. Documents are ingested idempotently (tracked by content hash).

## Key Design Decisions

- **Three languages**: Sinhala (`si-LK`), English (`en-US`), Tamil (`ta-IN`) — each with a dedicated Azure Neural voice
- **Claude model**: `claude-sonnet-4-20250514` with `max_tokens=1024` (increased for tool use responses)
- **Claude tool use**: `check_availability` and `create_booking` tools call eZee PMS API; tools auto-disable if eZee not configured
- **System prompt enforces language**: "You speak ONLY in {language}" prevents code-switching
- **Streaming TTS in test_voice.py**: Sentences are queued to a background TTS thread as they complete during Claude streaming, so speech starts before the full response is generated
- **ConversationRelay handles TTS in production**: server.py sends tokens to Twilio which manages Azure TTS internally
- **KB module degrades gracefully**: Works without chromadb/sentence-transformers installed (returns empty context)
