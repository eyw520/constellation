# Constellation

A multi-agent voice harness that orchestrates specialized LLM agents for real-time conversational AI.

## Why Constellation

Most voice agent frameworks use a single LLM for everything: understanding intent, generating responses, deciding when to call tools. This creates a monolithic system where the primary LLM must handle classification, routing, and generation in a single pass.

Constellation takes a different approach: **hierarchical multi-agent orchestration**. Specialized classification agents preprocess each user message before the primary LLM responds. This enables:

- **Intent extraction** without polluting the main conversation context
- **Cost optimization** via gate mechanisms that skip irrelevant classifiers
- **Parallel background processing** for analytics, logging, or enrichment
- **Config-driven behavior** with no code changes for new agent patterns

## Multi-Agent Architecture

Each user message flows through three agent tiers:

```
                              User Message
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        SYNC ENGINE LAYER                                 │
│  Classification agents that run BEFORE the primary LLM responds          │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐                         │
│  │    SyncEngine 1    │  │    SyncEngine 2    │  ...                    │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │                         │
│  │  │   Gate LLM   │──┼──│  │   Gate LLM   │  │  Fast/cheap pre-filter  │
│  │  └──────┬───────┘  │  │  └──────┬───────┘  │                         │
│  │         │ pass     │  │         │ pass     │                         │
│  │  ┌──────▼───────┐  │  │  ┌──────▼───────┐  │                         │
│  │  │  Class LLM   │  │  │  │  Class LLM   │  │  Structured output      │
│  │  └──────┬───────┘  │  │  └──────┬───────┘  │                         │
│  │         │          │  │         │          │                         │
│  │    [TaskTag]       │  │    [TaskTag]       │                         │
│  └─────────┼──────────┘  └─────────┼──────────┘                         │
│            └───────────┬───────────┘                                     │
│                        ▼                                                 │
│              Synthetic Tool Exchange                                     │
│         (inject results into conversation)                               │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        PRIMARY LLM LAYER                                 │
│  Main conversation agent with full context + tool access                 │
│                                                                          │
│  Context (static + dynamic) + Turn History + Synthetic Results           │
│                         │                                                │
│                         ▼                                                │
│                   Primary LLM ──▶ Stream Response ──▶ TTS ──▶ Speaker    │
│                         │                                                │
│                         ▼                                                │
│                   Tool Calls ──▶ Execute ──▶ Loop                        │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ (parallel, non-blocking)
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       ASYNC ENGINE LAYER                                 │
│  Background agents that don't block response generation                  │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐                         │
│  │   AsyncEngine 1    │  │   AsyncEngine 2    │  ...                    │
│  │   (Sentiment)      │  │   (Summarization)  │                         │
│  └────────────────────┘  └────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────────┘
```

A single user utterance may invoke 4+ LLM calls: gate evaluations, classifications, primary response, and background processing—all coordinated automatically.

## Core Concepts

### Engines

Engines are specialized LLM agents that process messages independently from the primary conversation LLM.

**Sync Engines** run sequentially before the primary LLM responds. They classify user intent and output `TaskTags` that influence the conversation:

```yaml
engines:
  - type: sync
    name: farewell_detector
    system_prompt: "Classify if the user is ending the conversation."
    user_prompt_template: "Conversation:\n{turns}"
    output_choices: [continue, farewell]
    task_tag_mapping:
      continue: null
      farewell:
        type: result
        task_name: farewell_detected
        result: "User is ending the conversation. Wrap up politely."
```

**Async Engines** run in parallel threads after the primary response begins. They don't block the conversation:

```yaml
engines:
  - type: async
    name: sentiment_logger
    system_prompt: "Analyze the emotional tone of this exchange."
    user_prompt_template: "{turns}"
```

### TaskTags

TaskTags are the outputs of sync engines. They determine what action to take based on classification:

| Type | Effect |
|------|--------|
| `result` | Injects information into conversation via synthetic tool exchange |
| `invocation` | Executes a tool automatically before the primary LLM responds |
| `disable` | Turns off this engine for the remainder of the conversation |

```yaml
task_tag_mapping:
  # No action for this classification
  general: null

  # Inject context into conversation
  scheduling_intent:
    type: result
    task_name: scheduling
    result: "User wants to schedule something. Check calendar availability."

  # Auto-execute a tool
  fetch_data:
    type: invocation
    tool_name: get_user_profile
    tool_input: {}

  # Disable this engine after triggering
  greeting_complete:
    type: disable
```

### Synthetic Tool Exchange

When a sync engine returns a `TaskTagResult`, the system creates a synthetic tool call/result pair and injects it into the conversation history. The primary LLM sees this as if it had requested the information itself:

```
Engine classifies "farewell"
    → System creates fake tool call: {name: "engine_task_result", input: {}}
    → System creates fake tool result: {content: "User is ending the conversation..."}
    → Both injected into turn history
    → Primary LLM naturally incorporates the hint
```

This avoids complex prompt engineering—classification results flow through the existing tool paradigm.

### Gate Mechanism

Gates are optional pre-filters that decide whether a sync engine should run at all. Use cheaper/faster models for gates to reduce cost and latency:

```yaml
engines:
  - type: sync
    name: technical_support
    gate:
      prompt: "Is this message about a technical issue or product problem?"
      model: gpt-4.1-nano-2025-04-14  # Fast gate
      num_turns: 2
    # Full classification only runs if gate passes
    system_prompt: "Classify the technical issue type..."
    model: gpt-4.1-2025-04-14
    output_choices: [billing, bug, feature_request, other]
```

### Interrupt Handling

Constellation supports real-time interruption. When the user speaks while the agent is responding:

1. VAD detects speech onset
2. LLM inference is canceled mid-stream
3. TTS audio queue is cleared
4. Speaker output stops immediately
5. System processes the new transcript

This enables natural turn-taking without waiting for the agent to finish.

## Configuration Reference

Agents are fully configured via YAML:

```yaml
prompt: |
  You are a helpful voice assistant.

initiation:
  enabled: true
  greeting: "Hello! How can I help you?"

llm:
  model: gpt-4.1-2025-04-14
  temperature: 0.7
  max_tokens: 1024

tools:
  - type: builtin
    name: get_current_time
  - type: builtin
    name: end_conversation

engines:
  - type: sync
    name: intent_classifier
    # ... engine config
  - type: async
    name: background_logger
    # ... engine config

mcp_servers:
  - name: external_service
    transport:
      type: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4o-2024-11-20`, `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`, `gpt-4.1-nano-2025-04-14` |
| Anthropic | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |

### MCP Integration

Connect external tool servers via Model Context Protocol:

```yaml
mcp_servers:
  - name: my_server
    transport:
      type: stdio
      command: node
      args: ["path/to/server.js"]
    timeout_seconds: 30

tools:
  - type: my_custom_tool
    handler:
      type: mcp
      server: my_server
      tool: actual_tool_name
      input_mapping:
        mcp_param: local_param
```

## Getting Started

See [SETUP.md](SETUP.md) for installation and running instructions.

## Project Structure

```
constellation/
├── src/constellation/
│   ├── core/           # Agent, VoiceSession, ContextManager
│   ├── engines/        # SyncEngine, AsyncEngine, EngineExecutor
│   ├── services/
│   │   ├── llm/        # LLMService, ChatLLM (multi-provider)
│   │   ├── asr/        # DeepgramASR
│   │   ├── tts/        # OpenAITTS
│   │   ├── vad/        # WebRTCVAD
│   │   └── mcp/        # MCPClient, MCPServerManager
│   ├── tools/          # ToolRegistry, ToolFactory
│   ├── audio/          # MicrophoneInput, SpeakerOutput
│   └── models/         # Pydantic configuration models
├── agents/             # Agent YAML configurations
└── SETUP.md
```
