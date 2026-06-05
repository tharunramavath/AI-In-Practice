# Intro to AI Engineering

Welcome to the foundational module of AI in Practice. This folder introduces the core concepts of AI and their applications.

---

### Lesson 0: Introduction to AI Engineering


#### 0.1 What is AI Engineering?

AI Engineering is the discipline of designing, building, deploying, and operating software applications that are powered by artificial intelligence models — primarily Large Language Models (LLMs). Unlike traditional software engineering, which operates on deterministic, rule-based logic, AI Engineering embraces a new paradigm where outputs are probabilistic, contextual, and emergent.

An AI Engineer sits at the crossroads of three domains:

- **Software Engineering:** writing clean, maintainable code, managing APIs, building pipelines
- **AI/ML Knowledge:** understanding model behavior, prompt design, evaluation, fine-tuning
- **Product Thinking:** framing problems correctly, measuring user value, iterating rapidly

- Think of AI Engineering as 'infrastructure for intelligence' — the plumbing, guardrails, and orchestration that make AI models useful in real products.

In traditional software, you write deterministic functions: input A always produces output B. In AI Engineering, you are working with a stochastic function — the same prompt can produce different outputs depending on temperature, model version, and context. This fundamentally changes how you write, test, and deploy software.

| Aspect | Traditional Software | AI Engineering |
|--------|----------------------|----------------|
| Logic | Explicit rules | Learned from data / prompted |
| Output | Deterministic | Probabilistic |
| Testing | Unit/integration tests | Eval suites, LLM-as-judge |
| Debugging | Stack traces, logs | Prompt inspection, token analysis |
| Versioning | Code versions | Model + prompt versions |
| Failure mode | Errors/exceptions | Hallucinations, bias, drift |

#### 0.2 The AI Engineering Lifecycle

Every production AI application moves through a repeatable lifecycle. Understanding each phase helps you avoid common pitfalls and ship faster.

1. **Problem Framing:** Define what problem you are solving, who the users are, and whether AI is even the right tool. This is the most underrated step.
2. **Data & Context Design:** Decide what information the model needs. What goes in the system prompt? What comes from a database? What comes from the user?
3. **Prototype:** Build the smallest possible version — usually a single API call with a system prompt. Validate that the approach works at all.
4. **Prompt Engineering:** Systematically refine prompts. Write eval sets. Measure output quality. Iterate until performance meets your threshold.
5. **Integration:** Wrap the LLM calls in proper application code — error handling, retries, logging, streaming, structured output parsing.
6. **Evaluation:** Build automated evals. Use LLM-as-judge, human review, or rule-based checks to continuously measure quality.
7. **Deployment:** Ship to production with monitoring, cost tracking, and fallback strategies.
8. **Iteration:** Analyze failure cases, collect user feedback, update prompts and models, redeploy.

- The lifecycle is NOT linear. You will often jump between steps 3–6 dozens of times before shipping. Embrace this loop.

#### 0.3 AI Application Architecture

A well-architected AI application is not just a wrapper around an API. It has several distinct layers:

**The Core Layers**

- **Presentation Layer:** Chat UI, REST API, CLI, or automated pipeline — whatever the user or downstream system interacts with
- **Orchestration Layer:** The logic that decides when to call the model, with what context, and how to handle the response. This is where most AI engineering work happens.
- **Model Layer:** The LLM itself — accessed via API (OpenAI, Anthropic, Google) or locally deployed (Ollama, vLLM)
- **Memory & Context Layer:** Short-term: conversation history. Long-term: vector databases, structured databases, file systems
- **Tool/Action Layer:** Functions the model can call — search, code execution, database queries, API calls
- **Evaluation & Observability Layer:** Logging, tracing, LLM-as-judge evals, cost tracking, latency monitoring

**Request Flow**

A typical AI request flows like this:

```
User Input
    ↓
Input Validation & Safety Filtering
    ↓
Context Retrieval  (RAG / memory lookup)
    ↓
Prompt Assembly   (system + context + user message)
    ↓
LLM API Call      (with retry + timeout)
    ↓
Response Parsing  (structured output / plain text)
    ↓
Output Validation (guardrails / hallucination check)
    ↓
Logging & Tracing
    ↓
Return to User
```

#### 0.4 AI Engineering vs. ML Engineering

These two roles are often confused but serve different purposes:

| Dimension | AI Engineer | ML Engineer |
|-----------|-------------|-------------|
| Primary focus | Building applications using pre-trained models | Training, fine-tuning, and serving ML models |
| Key skill | Prompt engineering, API integration, system design | Model architecture, distributed training, MLOps |
| Models used | GPT-4, Claude, Gemini (via API) | Custom PyTorch/TensorFlow models |
| When to use AI | Prototyping, language tasks, reasoning | When you need domain-specific accuracy at scale |
| Time to first output | Hours to days | Weeks to months |
| Infrastructure | API calls, serverless | GPUs, Kubernetes, model registries |
| Evaluation | Prompt evals, user feedback | Model metrics: loss, accuracy, F1 |

- AI Engineers need ML intuition — understanding why a model fails, what fine-tuning can fix, and when retrieval beats prompting — even if they never train a model themselves.

#### 0.5 AI Product Thinking

Building with AI is fundamentally a product problem first, an engineering problem second. Many AI projects fail not because the model was wrong, but because the product was misconceived.

**The Three Questions Before You Write a Single Line of Code**

1. **What is the job-to-be-done?** What task is the user trying to complete? Be specific. 'Help me write emails' is not specific. 'Help a sales rep write personalized cold emails in under 30 seconds' is.
2. **What does 'good' look like?** Before building, define your success criteria. Is it speed? Accuracy? User satisfaction? You need this to build evals.
3. **Is AI the right tool?** Could a simpler approach — a template, a search engine, a rule-based system — do the job better and cheaper? AI should be chosen deliberately, not by default.

**The AI Product Development Mindset**

- **Start with evals:** Write your test cases before you write your prompt. Know what 'passing' means.
- **Fail fast with stubs:** Use a simple LLM call first. Get user feedback. Then add complexity.
- **Treat prompts as code:** Version control them. Review them. Test them.
- **Plan for graceful degradation:** What happens when the model fails, hallucinates, or returns garbage?
- **Measure, don't guess:** Track latency, cost per request, user satisfaction, and task completion rate.

---

### Lesson 2: What is Agentic AI?

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

### Lesson 3: Tool Calling & MCP


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

#### What is an API?

- An API is a way for one software system to communicate with another software system.
- It exposes specific functionality through endpoints.

*Example — Job Portal API:*

- `GET /jobs`
- `GET /jobs/123`
- `POST /applications`

- The AI or application sends requests to these endpoints.
- The API returns responses.

*Think of an API as:*

- A service that provides functionality.
- A door through which software communicates.

#### What is MCP (Model Context Protocol)?

- MCP (Model Context Protocol) is a standard communication protocol between AI models and external systems.
- It defines a common way for tools, databases, files, and APIs to expose capabilities to AI models.

##### Simple Definition

- MCP is for AI tools what HTTP is for websites.
- HTTP standardized communication between browsers and websites.
- MCP standardizes communication between AI models and tools.

##### What Problem Exists Without MCP?

Suppose:

- **Job System** → `search_jobs(query)`
- **Resume System** → `get_file(path)`
- **Application System** → `submit_application(job_id, resume)`
- All systems use different formats.
- Every new system requires custom integration.
- The AI platform must learn how to interact with each one separately.

##### How MCP Works

**Without MCP:**

```
AI
 ├── Custom Integration with Job API
 ├── Custom Integration with Resume System
 └── Custom Integration with Application System
```

**With MCP:**

```
AI
  │
 MCP
  │
 ├── Job System
 ├── Resume System
 └── Application System
```

- The AI talks only to MCP.
- MCP provides a common interface.
- New tools can be connected without changing the AI.

**Real Example**

User asks: "Find Machine Learning jobs and apply using my resume."

MCP Exposes:
- `search_jobs()`
- `read_resume()`
- `apply_job()`

AI Workflow:
1. `search_jobs()`
2. `read_resume()`
3. `apply_job()`

The AI does not need to know:
- Which API is used
- Which database stores jobs
- Where the resume file is located
- How authentication works

MCP hides those implementation details.

##### Difference Between API, Tool, and MCP

| Aspect | API | Tool | MCP |
|--------|-----|------|-----|
| **Purpose** | Expose software functionality | Action AI can execute | Standard way for AI to access tools |
| **Seen By** | Developers | AI Agent | AI Platform & Tool Providers |
| **Example** | `POST /applications` | `apply_job()` | Protocol that exposes `apply_job()` uniformly |
| **Level** | Infrastructure | Agent Capability | Coordination Layer |
| **Responsibility** | Provides service | Performs task | Standardizes communication |

##### One-Line Understanding

- **API** = The actual service.
- **Tool** = The capability the AI uses.
- **MCP** = The standard protocol that allows AI models to discover and use tools consistently.

---

### Lesson 4: Agentic RAG & Multi-Agent Systems


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
