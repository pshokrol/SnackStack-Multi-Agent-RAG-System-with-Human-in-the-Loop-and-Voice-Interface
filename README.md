# SnackStack: Multi-Agent RAG System with Human-in-the-Loop and Voice Interface

SnackStack is a multi-agent conversational AI system for food ordering, built with LangGraph. It features an Orchestrator for query routing, a Menu Agent with ChromaDB RAG for semantic food search, and an Order Agent with Human-in-the-Loop (HITL) for order tracking. Responses are unified by a Synthesizer. Bonus: voice I/O via OpenAI Whisper STT and TTS.

---

## Architecture

```
User Query
    ↓
Orchestrator  (structured-output routing)
    ↓              ↓
Menu Agent    Order Agent
(RAG/ChromaDB) (HITL + regex)
    ↓              ↓
       Synthesizer
            ↓
       Final Answer
```

---

## Features

- **Orchestrator**: Routes queries to the correct agent using GPT-4o structured output
- **Menu Agent**: Semantic food search powered by ChromaDB vector store and RAG
- **Order Agent**: Order tracking with Human-in-the-Loop (HITL) interrupt when no identifier is provided
- **Synthesizer**: Merges agent outputs into a single coherent, friendly reply
- **Voice I/O**: Full voice support via OpenAI Whisper (STT) and TTS
- **Three interaction modes**: Text chat, full voice (mic + speaker), or type in / speak out

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB |
| Speech-to-Text | OpenAI Whisper |
| Text-to-Speech | OpenAI TTS |
| Audio I/O | sounddevice, soundfile |
| Environment | Python 3.11, Conda |

---

## Project Structure

```
├── run.py                  # Main entry point — all agents, graph, and voice loop
├── .gitignore              # Excludes .env from version control
├── .env                    # API keys (NOT uploaded to GitHub)
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/pshokrol/SnackStack-Multi-Agent-RAG-System-with-Human-in-the-Loop-and-Voice-Interface.git
cd SnackStack-Multi-Agent-RAG-System-with-Human-in-the-Loop-and-Voice-Interface
```

### 2. Create and activate a conda environment
```bash
conda create -n snackstack python=3.11
conda activate snackstack
```

### 3. Install dependencies
```bash
pip install langgraph langchain langchain-openai langchain-community chromadb openai sounddevice soundfile numpy python-dotenv pydantic
```

### 4. Set up your API key
Create a `.env` file in the project folder:
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. Run the assistant
```bash
python run.py
```

You will be prompted to select a mode:
```
1 - Text chat
2 - Full voice (mic + speaker)
3 - Type in, speak out
```

---

## Example Queries

| Query | Agent Routed |
|---|---|
| `hi there!` | Menu Agent |
| `Show me vegan dishes` | Menu Agent |
| `Indian food under $300` | Menu Agent |
| `Track order ORD-201` | Order Agent |
| `Where is my order?` | Order Agent (HITL triggered) |
| `Show me pizza and track ORD-203` | Both Agents |

---

## Voice Modes

| Mode | Input | Output |
|---|---|---|
| Mode 2 — Full Voice | Microphone (Whisper STT) | Speaker (TTS) |
| Mode 3 — Voice Out | Keyboard | Speaker (TTS) |

---

## Notes

- ChromaDB runs in-memory and rebuilds each session; this is normal
- The `.env` file is excluded from version control via `.gitignore`
- Voice mode requires a working microphone and speakers
- All LLM inference runs via OpenAI API (no GPU required)

---

## License

MIT License
