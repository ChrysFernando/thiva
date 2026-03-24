# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Thiva is a multilingual voice customer service agent for Winrich. It handles phone calls via Twilio, uses Claude for conversation, Azure Speech for TTS, and a ChromaDB-based RAG knowledge base.

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

## Architecture

**Production call flow** (`server.py`):
Incoming call → `/voice/incoming` (IVR greeting) → DTMF digit selects language → `/voice/handle-language` connects Twilio ConversationRelay → WebSocket `/ws/conversation` streams Claude responses token-by-token → ConversationRelay handles Azure TTS playback.

**Local test flow** (`test_voice.py`):
CLI language selection → typed input → KB retrieval → Claude streaming → sentence-by-sentence Azure TTS on background thread (splits on `.` and `?` for low-latency playback).

**Knowledge base** (`knowledge_base.py`):
Files in `knowledge_docs/` are chunked (500 chars, 50 overlap) → embedded with `all-MiniLM-L6-v2` → stored in ChromaDB (`./chroma_db/`). On each query, top 3 chunks are retrieved and injected into the system prompt as `<knowledge_base>` context. Documents are ingested idempotently (tracked by content hash).

## Key Design Decisions

- **Three languages**: Sinhala (`si-LK`), English (`en-US`), Tamil (`ta-IN`) — each with a dedicated Azure Neural voice
- **Claude model**: `claude-sonnet-4-20250514` with `max_tokens=256` (kept short for voice)
- **System prompt enforces language**: "You speak ONLY in {language}" prevents code-switching
- **Streaming TTS in test_voice.py**: Sentences are queued to a background TTS thread as they complete during Claude streaming, so speech starts before the full response is generated
- **ConversationRelay handles TTS in production**: server.py sends tokens to Twilio which manages Azure TTS internally
- **KB module degrades gracefully**: Works without chromadb/sentence-transformers installed (returns empty context)
