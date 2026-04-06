# Constellation

A voice agent harness for building conversational Voice Agents with real-time audio processing.

## Architecture

Constellation provides a complete voice application stack:

```
┌─────────────────────────────────────────────────────────────┐
│                      VoiceSession                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐ │
│  │   ASR   │──▶│   VAD   │──▶│   LLM   │──▶│     TTS     │ │
│  │Deepgram │   │ WebRTC  │   │ Service │   │   OpenAI    │ │
│  └─────────┘   └─────────┘   └─────────┘   └─────────────┘ │
│       ▲                           │                         │
│       │                           ▼                         │
│  ┌─────────┐              ┌─────────────┐                  │
│  │   Mic   │              │   Engines   │                  │
│  │  Input  │              │ Sync/Async  │                  │
│  └─────────┘              └─────────────┘                  │
│       │                           │                         │
│       ▼                           ▼                         │
│  ┌─────────┐              ┌─────────────┐                  │
│  │ Speaker │              │    Tools    │                  │
│  │ Output  │              │Registry/MCP │                  │
│  └─────────┘              └─────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **VoiceSession**: Orchestrates the complete voice interaction loop
- **ASR (Deepgram)**: Real-time speech-to-text transcription
- **VAD (WebRTC)**: Voice activity detection for interruption handling
- **LLM Service**: Multi-provider support (OpenAI, Anthropic) with streaming
- **TTS (OpenAI)**: Text-to-speech synthesis with streaming output
- **Engines**: Sync/Async processing pipelines with TaskTag outputs
- **Tools**: Registry-based tool system with MCP server support

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

## Agent Configuration

Agents are configured via YAML files. Here's the structure:

```yaml
# System prompt for the LLM
prompt: |
  You are a helpful voice assistant. Be conversational and concise.

# Conversation initiation settings
initiation:
  enabled: true
  greeting: "Hello! How can I help you today?"

# LLM configuration
llm:
  model: gpt-4.1-2025-04-14  # or claude-sonnet-4-20250514, etc.
  temperature: 0.7
  max_tokens: 1024

# Tools available to the agent
tools:
  - type: builtin
    name: get_current_time
  - type: builtin
    name: end_conversation

# Processing engines
engines:
  - type: sync
    name: intent_classifier
    system_prompt: |
      Classify the user's intent.
    user_prompt_template: |
      Conversation:
      {turns}
    output_choices:
      - general
      - farewell
    task_tag_mapping:
      general: null
      farewell:
        type: result
        task_name: farewell
        result: "User is ending the conversation."
    model: gpt-4.1-2025-04-14
    num_turns: 3

# MCP server connections (optional)
mcp_servers:
  - name: filesystem
    transport:
      type: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    timeout_seconds: 30
```

### Supported Models

**OpenAI:**
- `gpt-4o-2024-11-20`
- `gpt-4.1-2025-04-14`
- `gpt-4.1-mini-2025-04-14`
- `gpt-4.1-nano-2025-04-14`

**Anthropic:**
- `claude-sonnet-4-20250514`
- `claude-opus-4-20250514`

### Builtin Tools

| Tool | Description |
|------|-------------|
| `get_current_time` | Returns the current date and time |
| `end_conversation` | Ends the conversation gracefully |

### Engine Types

**Sync Engines**: Run before each LLM turn, can inject context or trigger actions

```yaml
- type: sync
  name: classifier
  system_prompt: "Classification prompt"
  user_prompt_template: "Template with {turns}"
  output_choices: [choice1, choice2]
  task_tag_mapping:
    choice1: null  # No action
    choice2:
      type: result
      task_name: my_task
      result: "Injected context"
  model: gpt-4.1-2025-04-14
  num_turns: 5  # History window
  gate:  # Optional gate condition
    prompt: "Should this engine run?"
    model: gpt-4.1-mini-2025-04-14
    num_turns: 2
```

**Async Engines**: Run in background, don't block the conversation

```yaml
- type: async
  name: logger
  system_prompt: "Summarize the conversation"
  user_prompt_template: "{turns}"
  model: gpt-4.1-mini-2025-04-14
  num_turns: 10
```

### MCP Integration

Connect to MCP servers for additional tool capabilities:

```yaml
mcp_servers:
  - name: my_server
    transport:
      type: stdio
      command: node
      args: ["path/to/server.js"]
      env:
        MY_VAR: value
    timeout_seconds: 30

tools:
  - type: my_custom_tool
    description: "Tool from MCP server"
    handler:
      type: mcp
      server: my_server
      tool: actual_tool_name
      input_mapping:
        mcp_param: local_param
      output_mapping:
        local_field: mcp_field
```

## Project Structure

```
constellation/
├── src/constellation/
│   ├── core/           # Agent, Session, Context, Turn
│   ├── engines/        # Sync/Async engines, Executor
│   ├── services/
│   │   ├── llm/        # LLM service, Chat, Types
│   │   ├── asr/        # Deepgram ASR
│   │   ├── tts/        # OpenAI TTS
│   │   ├── vad/        # Voice activity detection
│   │   └── mcp/        # MCP client, transport, manager
│   ├── tools/          # Registry, Factory, Builtins
│   ├── audio/          # Microphone input, Speaker output
│   ├── models/         # Pydantic config models
│   ├── cli.py          # CLI entry point
│   ├── loader.py       # YAML config loading
│   ├── settings.py     # Environment settings
│   └── logger.py       # Logging configuration
├── agents/             # Agent configuration files
├── logging.conf        # Optional logging configuration
├── pyproject.toml
└── README.md
```
