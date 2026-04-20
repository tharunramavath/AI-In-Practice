# Agentic AI Basics

Welcome to the foundational module on Agentic AI. This folder introduces the core concepts of AI agents and their applications.

---

## Course Contents

### Lesson 1: What is Agentic AI?

#### What Are AI Agents?

- An AI agent is a system that can perceive inputs, reason about them, take actions using tools, and iterate until a goal is achieved.
- An AI agent operates in a closed loop of goal -> planning -> action -> observation -> refinement, rather than producing a single output.
- An AI agent uses a language model such as GPT-4 or Claude as its reasoning engine, but extends it with tools and memory.
- An AI agent is designed to handle multi-step, dynamic, and uncertain tasks where the next step depends on previous results.
- An AI agent differs from a simple chatbot because it can take actions in external systems such as APIs, browsers, or databases.

#### AI Workflows vs AI Agents

| Aspect | AI Workflow | AI Agent |
|--------|-------------|----------|
| Flow | Fixed sequence of steps | Dynamic, runtime decisions |
| Suitable for | Deterministic tasks | Non-deterministic tasks |
| Structure | Input -> Model -> Output | Goal -> Plan -> Act -> Observe -> Repeat |

- An AI workflow is a predefined sequence of steps where the flow is fixed and does not change based on intermediate outcomes.
- An AI agent is a dynamic system that decides the next step at runtime based on observations and feedback.
- An AI workflow is suitable for deterministic tasks where the same process works every time, such as document summarization pipelines.
- An AI agent is suitable for non-deterministic tasks where decisions must adapt dynamically, such as job searching or travel booking.

#### Types of AI Agents

| Type | Description | Examples |
|------|-------------|----------|
| **Reactive Agents** | Respond only to current input without memory | Nest Learning Thermostat, Tesla Obstacle Avoidance, Outlook Spam Filters |
| **Memory-Based Agents** | Store past interactions and improve responses | ChatGPT, Gemini, Mem0, Zep |
| **Goal-Based Agents** | Plan sequences of actions to achieve objectives | OpenAI Operator, Google Maps Navigation, Lindy AI |
| **Utility-Based Agents** | Optimize measurable objectives (cost, time, accuracy) | Uber Dispatch, Google Ads Smart Bidding, Fraud Detection |
| **Learning Agents** | Improve performance over time using feedback | Tesla FSD, YouTube Algorithm, AlphaZero |
| **Autonomous Agents** | Operate independently with minimal human input | Devin AI, AutoGPT, BabyAGI, Boston Dynamics Atlas |
| **Multi-Agent Systems** | Multiple specialized agents collaborating | CrewAI, Microsoft AutoGen, LangGraph, ChatDev |
| **Tool-Using Agents** | Interact with APIs, databases, and external tools | Claude Computer Use, Gemini Extensions, SmolAgents |
| **Voice AI Agents** | Understand and respond via speech in real time | Gemini Live, GPT-4o Voice, Bland AI, ElevenLabs |

#### When (and When Not) to Use AI Agents

**Use AI Agents When:**
- A task requires multiple steps, dynamic decision-making, and interaction with external tools
- The problem involves uncertainty, iteration, and changing conditions during execution
- The goal is high-level and cannot be solved with a single model call

**Avoid AI Agents When:**
- Simple, single-step tasks where a direct model response is sufficient
- Latency, cost, or system complexity must be minimized
- No adaptive behavior is required

---

### Lesson 2: Tool Calling & MCP


#### Why Agents Need Tools

- AI agents need tools because LLMs can generate text but cannot directly interact with external systems such as APIs, databases, or websites.
- Tools enable agents to perform real actions and access live data, which is essential for solving practical, multi-step problems.
- *Real-world example:* In a job application agent, tools are used to scrape job listings, filter roles, and submit applications automatically.

#### Tool Calling for LLMs

- Tool calling allows an LLM to select and invoke external functions by generating structured outputs with required parameters.
- The system executes the tool and feeds the result back to the model, enabling iterative reasoning and decision-making.
- *Real-world example:* A banking assistant agent calls an API to fetch recent transactions and then formats them into a user-friendly response.

#### When to Call Tools Manually

- Manual tool calling is used when you need strict control over execution to ensure correctness, safety, and reliability.
- It is preferred when tool usage conditions are fixed and predictable rather than dependent on model judgment.
- *Real-world example:* In a payment system, the backend enforces that the payment API is only called after explicit user confirmation.

#### What is MCP (Model Context Protocol)?

- MCP is a protocol that standardizes how AI models connect with tools, data sources, and external systems through a unified interface.
- It enables agents to use multiple tools dynamically without requiring custom integration for each one.
- *Real-world example:* A data assistant agent can seamlessly choose between querying a database, reading a file, or calling an API using a common MCP interface.

| Aspect | Tools | MCP | API |
|--------|-------|-----|-----|
| **Core Purpose** | Functions/actions an agent can execute | Standardized protocol for model-tool interaction | Interfaces for software communication |
| **Level of Abstraction** | Agent level (executable capabilities) | System level (coordination layer) | Infrastructure level (endpoints) |
| **Real-World Example** | "search_jobs", "apply_job" tools | Uniform access to job tools, databases, APIs | LinkedIn API for job listings |

---

### Lesson 3: Agentic RAG & Multi-Agent Systems


#### Why RAG?

- RAG is used to provide LLMs with external, up-to-date, and domain-specific knowledge instead of relying only on their training data.
- RAG improves accuracy and reduces hallucinations by grounding responses in retrieved documents or database results.
- *Real-world example:* A company chatbot retrieves internal policy documents before answering.

#### Agentic RAG

- Agentic RAG extends RAG by allowing the agent to decide when, what, and how to retrieve information dynamically based on the task.
- The agent can iterate multiple retrieval steps, refine queries, and combine results before generating the final answer.
- *Real-world example:* A research assistant agent searches multiple sources, refines queries, and validates information before producing a structured report.

#### What are Orchestrator Agents?

- Orchestrator agents are responsible for managing and coordinating multiple tools, models, or sub-agents to achieve a goal.
- They break down a complex task into subtasks and assign them to specialized agents or components.
- *Real-world example:* In a job automation system, the orchestrator assigns tasks such as job scraping, resume customization, and application submission to different modules.

#### Multi-Agent Systems Fundamentals

- Multi-agent systems consist of multiple specialized agents collaborating to solve a complex problem more efficiently than a single agent.
- Each agent has a specific role, such as retrieval, reasoning, or execution, and communicates with others to complete tasks.
- *Real-world example:* A customer support system may use one agent for query understanding, another for knowledge retrieval, and another for response generation.

---

#### Risks of Agentic AI

- Agentic systems can produce incorrect or unsafe actions if the model makes wrong decisions or uses tools improperly.
- These systems can introduce security and privacy risks when interacting with sensitive data or external APIs.
- *Real-world example:* An autonomous agent could execute an incorrect financial transaction or misuse an API without proper validation safeguards.
