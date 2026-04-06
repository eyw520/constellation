# Setup

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) for dependency management
- Working microphone and speakers
- API keys for:
  - OpenAI (for LLM and TTS)
  - Anthropic (optional, for Claude models)
  - Deepgram (for ASR)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd constellation
```

2. Install dependencies:
```bash
poetry install
```

3. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # Optional
DEEPGRAM_API_KEY=...
```

## Running the Agent

Run the demo agent:

```bash
poetry run constellation run agents/demo_agent.yaml
```

With verbose logging:

```bash
poetry run constellation run agents/demo_agent.yaml -v
```

The agent will:
1. Initialize audio input/output
2. Connect to ASR service
3. Begin listening for speech
4. Process transcripts through the LLM
5. Speak responses via TTS
6. Handle interruptions when you speak while it's responding

Press `Ctrl+C` to stop.
