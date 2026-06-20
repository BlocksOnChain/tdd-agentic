# Local Model Fine-Tuning Research Brief

**Purpose:** Research document for evaluating how to adapt local Qwen models to work more efficiently with the tools, skills, and agent roles in the **TDD Agentic** project (`tdd-agentic`).

**Author context:** M2 Max MacBook, 64 GB unified memory. Local inference models:
- **Qwen 3-coder-next** (~50 GB on disk — likely a large dense coder GGUF)
- **Qwen 3.6** (~16 GB — likely a quantized MoE or mid-size instruct model)

**Audience:** A follow-up research agent. This doc is intentionally dense and actionable.

---

## 1. Executive Summary

Fine-tuning is **one lever among several**. For this project, the highest ROI path is usually:

1. **Baseline without training** — wire local models via `OPENAI_BASE_URL`, tune prompts/tools, collect traces.
2. **Per-role specialization** — train **small LoRA adapters** (or separate small models) for the agents that fail most often locally: PM routing, Lead JSON planning, Dev TDD tool loops.
3. **Cloud train → local deploy** — use Hugging Face Jobs / Unsloth on NVIDIA GPUs for larger bases (including coder variants), merge LoRA, convert to GGUF, serve with `llama-server`.
4. **Keep the 50 GB coder for inference** — do not attempt full fine-tune of a 50 GB dense model on 64 GB RAM; distill its behavior into the 16 GB model or a 4–8B specialist.

The project's architecture already separates cognitive roles (Lead plans JSON only; Coordinator persists) specifically to make **narrow fine-tuning targets** easier.

---

## 2. What “Efficiency” Means in This Project

Efficiency is not just tokens/second. Success metrics for local models in `tdd-agentic`:

| Dimension | Definition | Where it shows up |
|-----------|------------|-------------------|
| **Tool selection accuracy** | Correct tool chosen among 6–12 tools | PM, Dev, Lead, Researcher |
| **Tool argument validity** | Pydantic schemas satisfied (UUIDs, enums, RITE test_cases) | All tool-using agents |
| **Structured output reliability** | Valid routing JSON / execution_plan JSON without prose | PM, Lead |
| **Workflow adherence** | TDD red→green, one subtask per turn, resume-from-DB | Dev, DevOps, QA |
| **Context discipline** | Uses `list_tickets` not redundant `get_ticket`; passes `ticket_id` to dev tools | PM, Dev |
| **Skill usage** | Calls `rag_query` for skill bodies instead of hallucinating | All roles with assigned skills |
| **Latency & cost** | Fewer tool-call rounds per subtask; fits `DEV_AGENT_MAX_STEPS=60` | Dev agents |

Existing internal analysis (`docs/tool-calling-and-prompts-analysis.md`, `docs/agent-efficiency-plan.md`) documents prompt/tool gaps independent of model choice. **Fine-tuning complements but does not replace** those improvements.

---

## 3. Project Architecture (Training-Relevant)

### 3.1 Agent roles and LLM slots

Configured via `.env` (`backend/config.py`, `backend/agents/llm.py`):

| Env var | Default (cloud) | Role |
|---------|-----------------|------|
| `PM_MODEL` | anthropic/claude-sonnet-4-6 | Supervisor routing + ticket orchestration |
| `RESEARCHER_MODEL` | openai/gpt-4o | Web search, docs, RAG ingest, skill creation |
| `LEAD_MODEL` | anthropic/claude-sonnet-4-6 | Execution plan JSON (no tools) |
| `COORDINATOR_MODEL` | falls back to `DEV_MODEL` | DB persistence from plan |
| `DEV_MODEL` | anthropic/claude-sonnet-4-6 | Default for all devs |
| `BACKEND_DEV_MODEL` / `FRONTEND_DEV_MODEL` | optional overrides | Per-domain dev |
| `GRADER_MODEL` | anthropic/claude-haiku-4-5 | CRAG relevance grading |

**Local wiring:** Set `OPENAI_BASE_URL=http://127.0.0.1:8080/v1` and use `openai/<model_id>` slugs for every role. All roles must use the `openai/` prefix — `anthropic/` slugs bypass the local server.

### 3.2 Tool inventory by agent

| Agent | Tool count | Tools |
|-------|------------|-------|
| **PM** | 8 | `list_tickets`, `get_ticket`, `create_ticket`, `update_ticket_status`, `add_question_to_ticket`, `rag_query`, `ask_human` |
| **Lead** | 0 | **No tools** — outputs `execution_plan` JSON only |
| **Coordinator** | 5 | `save_ticket`, `transition_ticket`, `save_execution_plan`, `complete_assignment`, `rag_query` |
| **Researcher** | 7 | `web_search`, `rag_query`, `rag_ingest_text`, `create_skill`, `fs_write`, `fs_read`, `fs_list` |
| **Dev / DevOps / QA** | 12 | `get_ticket`, `next_pending_subtask`, `next_pending_subtask_in_project`, `update_subtask_status`, `mark_todo_done`, `add_todo_to_subtask`, `fs_*`, `shell_run`, `run_tests`, `rag_query` |

Tool definitions live in:
- `backend/tools/ticket_tools.py` — ticket/subtask lifecycle
- `backend/tools/code_tools.py` — workspace FS + tests
- `backend/tools/rag_tools.py` — CRAG retrieval
- `backend/tools/persistence_tools.py` — coordinator writes
- `backend/tools/hitl_tools.py` — human interrupt
- `backend/tools/web_search_tools.py` — Tavily/Anthropic search
- `backend/agents/researcher/subgraph.py` — `create_skill`

Tools use LangChain `@tool` with Pydantic `args_schema` where needed. Descriptions include **USE WHEN / AVOID WHEN / RETURNS** patterns (partially implemented — see analysis doc).

### 3.3 Skills system

- Skills stored at `workspace/_skills/<name>/SKILL.md` + `registry.json`
- Created at runtime by Researcher via `create_skill`
- Injected into prompts via `backend/agents/skills/loader.py` (index only; full body via `rag_query`)
- Roles: `project_manager`, `researcher`, `lead`, `coordinator`, `backend_dev`, `frontend_dev`, `devops`, `qa`

**Fine-tuning implication:** Skills are dynamic per project. Training should teach **behavior** (when to `rag_query`, how to follow SKILL.md conventions), not memorize specific skill text.

### 3.4 Prompts and structured outputs

- Central prompts: `backend/agents/prompts.py` (~650 lines)
- PM outputs routing JSON parsed by regex in `supervisor.py` (fragile — known issue)
- Lead outputs `execution_plan` JSON only
- Dev prompts encode strict TDD + RITE test format + stop conditions
- `DEFAULT_STACK_POLICY`: React/Express/Prisma/Vitest unless project overrides

### 3.5 Existing local integration points

From `.env.example`:
```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
PM_MODEL=openai/qwen3-coder-next
DEV_MODEL=openai/qwen3-coder-next
EMBEDDING_PROVIDER=local   # sentence-transformers on Mac
WEB_SEARCH_PROVIDER=tavily   # required for local LLMs (no Anthropic search)
OPENAI_REQUEST_TIMEOUT=300   # local inference is slow
```

Langfuse tracing (`LANGFUSE_*`) can capture LLM + tool traces for dataset building.

---

## 4. Hardware Profile & Model Fit

### 4.1 M2 Max 64 GB constraints

| Activity | 50 GB Qwen 3-coder-next | 16 GB Qwen 3.6 |
|----------|-------------------------|----------------|
| **Inference (Q4–Q5 GGUF)** | Tight — may need reduced context (`-c 4096`), one model at a time | Comfortable |
| **LoRA SFT (4-bit base + adapters)** | **Impractical** for full dense ~32B+ | **Feasible** if MoE (~3B active) or ~7–14B dense |
| **Full fine-tune** | Not feasible | Not feasible |
| **Merge LoRA → GGUF** | N/A locally | Feasible via mlx-tune export |
| **Parallel roles** | Cannot load PM + Dev models simultaneously if both are 50 GB | Possible with smaller quant |

**Unified memory rule of thumb:** Training needs base weights + optimizer states + activations. Budget ~1.5–2× model size for 4-bit LoRA on MLX; dense 50 GB leaves no headroom.

### 4.2 Recommended model assignment (inference)

| Role | Recommended local model | Rationale |
|------|-------------------------|-----------|
| PM, Lead | **Qwen 3.6** (general instruct) | Routing + JSON planning; smaller/faster turns |
| Backend/Frontend Dev, DevOps, QA | **Qwen 3-coder-next** | Code + test loops |
| Coordinator | **Qwen 3.6** or smallest viable | Mechanical tool mapping |
| Researcher | **Qwen 3.6** + Tavily | Less code generation |
| Grader (CRAG) | **Qwen 3.6** Q4 or dedicated 4B | Short classification outputs |

Use **role-specific env overrides** (`BACKEND_DEV_MODEL`, etc.) rather than one model for everything.

---

## 5. Option Taxonomy (All Paths)

### Option A — No fine-tuning (baseline & data collection)

**What:** Prompt engineering, tool description enrichment, structured output APIs, phased tool binding, RAG/skills tuning.

**Pros:** Zero training cost; aligns with existing docs (`tool-calling-and-prompts-analysis.md`).

**Cons:** Local Qwen may still weak on tool JSON vs Claude.

**Actions:**
1. Run stack locally with Langfuse enabled.
2. Implement PM `with_structured_output` before training (cleaner traces).
3. Log `(system, tools, messages, assistant_tool_calls, tool_results)` per agent turn.

**When to choose:** Always do this first; it produces training data and may be sufficient.

---

### Option B — Prompt / skill distillation (soft training)

**What:** Compress `prompts.py` + tool docs into role-specific “playbooks” injected at runtime; Researcher `create_skill` authors project playbooks.

**Pros:** No GPU; instant iteration.

**Cons:** Burns context tokens; doesn't fix weak native tool-calling.

**When to choose:** Alongside any training path; low effort.

---

### Option C — LoRA SFT on Apple Silicon (MLX)

**Stack:** [mlx-tune](https://github.com/ARahim3/mlx-tune) (formerly unsloth-mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), Apple's MLX framework.

**Best base for your machine:** **Qwen 3.6** (likely `mlx-community/Qwen3.5-35B-A3B-4bit` or similar MoE) — ~20 GB for 4-bit + LoRA per mlx-tune docs.

**Training methods:**
- **SFT** — imitate successful tool-call trajectories (primary)
- **DPO** — preferred vs rejected tool sequences (fix wrong tool picks)
- **GRPO** — reward correct routing + valid JSON (advanced)

**Workflow:**
1. Fine-tune LoRA on MLX with ShareGPT / OpenAI-messages dataset.
2. `save_pretrained_merged()` or export GGUF.
3. Serve merged or adapter via `llama-server`.
4. Point `OPENAI_BASE_URL` at server.

**Pros:** Native Mac; unified memory; no cloud cost; Unsloth-compatible API.

**Cons:** Slower than CUDA; limited to bases that fit in ~64 GB; GGUF export can have edge cases.

**Feasibility for 50 GB coder:** Train a **separate smaller coder LoRA** (e.g. Qwen3-Coder-7B/14B 4-bit) rather than the 50 GB checkpoint.

---

### Option D — Cloud GPU training (HF Jobs / Unsloth CUDA) → GGUF local

**Stack:** Hugging Face Jobs + TRL (`SFTTrainer`, `DPOTrainer`) or Unsloth; see Hugging Face `huggingface-llm-trainer` skill.

**Best for:** Qwen3-Coder-Next full-size or 32B+ adapters; multi-epoch production datasets.

**Workflow:**
1. Push dataset to Hub (private).
2. SFT with LoRA (`r=16`, `target_modules` attention + MLP).
3. Merge adapters → push merged weights.
4. Convert to GGUF (`llama.cpp` or HF Space `gguf-my-repo`).
5. Download to Mac; serve with `llama-server --host 0.0.0.0`.

**Pros:** Trains models too large for Mac; faster iteration; Qwen official function-call examples on GitHub.

**Cons:** HF Pro/Team for Jobs; egress/storage; must validate tool-call template matches llama-server.

**Cost estimate:** Demo 100 examples on T4 ~$1–3; production 5k–20k examples on A10G ~$20–80 per run (order-of-magnitude — verify with `estimate_cost.py` in HF skill).

---

### Option E — Per-role adapter suite (recommended training architecture)

Instead of one fine-tuned model, train **separate LoRA adapters** on a shared base:

| Adapter name | Base | Training focus | Deploy |
|--------------|------|----------------|--------|
| `tdd-pm-routing` | Qwen 3.6 instruct | Routing JSON + ticket tool selection | PM_MODEL |
| `tdd-lead-planning` | Qwen 3.6 instruct | `execution_plan` + RITE test_cases | LEAD_MODEL |
| `tdd-dev-tdd` | Qwen Coder 7–14B | TDD tool loop: next_subtask → fs_write → run_tests | DEV_MODEL |
| `tdd-coordinator` | Qwen 3.6 instruct | Map execution_plan → persistence tools | COORDINATOR_MODEL |

**Serving options:**
- **Multiple llama-server instances** on different ports (8080 PM, 8081 Dev).
- **Single server, swap GGUF** between runs (simplest but no parallelism).
- **llama.cpp LoRA hot-load** if build supports `--lora` (check version).

---

### Option F — Distillation from cloud teachers

**What:** Run projects with Claude/GPT as teachers (current defaults); export Langfuse traces; train local student on assistant turns only.

**Pros:** Highest-quality labels for tool sequences; matches production behavior.

**Cons:** Teacher cost during collection; risk of copying cloud-specific phrasing.

**Implementation:**
1. Enable Langfuse + log tool calls in `backend/agents/llm_audit.py`.
2. Filter traces: successful subtask completions, valid routing parses.
3. Convert to training rows (see §6).

---

### Option G — Synthetic dataset generation

**What:** Generate `(scenario → tool_calls)` pairs from tool schemas + prompts without live runs.

**Pros:** Cheap; covers edge cases (invalid UUID, empty test_cases).

**Cons:** Distribution shift vs real projects; must mix with real traces.

**Sources in repo:**
- Pydantic schemas in `ticket_tools.py`, `persistence_tools.py`
- Test fixtures in `backend/tests/`
- `backend/workspace_seed/` example projects

---

### Option H — Embedding model fine-tune (RAG, not chat)

**What:** Fine-tune `sentence-transformers` bi-encoder for project-specific retrieval (skills, docs, code).

**When:** Local models follow tools well but **RAG misses** relevant skill/doc chunks.

**Stack:** `train-sentence-transformers` HF skill; runs on Mac CPU/MPS for small models.

**Integration:** `EMBEDDING_PROVIDER=local`, `LOCAL_EMBEDDING_MODEL=<your-finetuned-model>`.

---

### Option I — RL / GRPO for tool-use (experimental)

**What:** Reward function on: valid JSON, tool success, test pass, subtask done.

**Stack:** TRL GRPO (cloud) or mlx-tune GRPO (Mac, smaller scale).

**When:** SFT plateaus; wrong tools persist after 1k+ examples.

**Risk:** Reward hacking (mark done without tests) — must align with `runner.py` verification gates.

---

### Option J — Replace fine-tune with smaller specialist models

**What:** Use Qwen3-4B/7B-Coder per role instead of adapting one 50 GB model.

**Pros:** Faster inference; easier training; fits multi-server setup.

**Cons:** More operational complexity; may lose reasoning depth on PM/Lead.

---

## 6. Dataset Design (Project-Specific)

### 6.1 Training formats

**LangChain runtime format:** OpenAI-compatible `tools` array + `tool_calls` in assistant messages.

**Qwen function-call fine-tune:** Often **ReAct-style** text in assistant turns (Qwen docs: no separate `function` role). Verify template for your exact Qwen version:
- [Qwen function_call_finetune_examples.py](https://github.com/QwenLM/Qwen/blob/main/examples/function_call_finetune_examples.py)
- Match template to what **llama-server** expects for your GGUF (critical).

**Recommended interchange:** Store datasets as **OpenAI messages JSONL**; convert with a small script per target template.

### 6.2 Per-role dataset contents

#### PM (`tdd-pm-routing`)
- **Input:** System prompt (abbreviated) + `project_id` + condensed ticket list state
- **Output:** Either tool calls (`list_tickets` first) OR final routing JSON
- **Negative examples:** Prose around JSON; guessed UUIDs; `get_ticket` when `list_tickets` suffices
- **Target size:** 500–2,000 trajectories

#### Lead (`tdd-lead-planning`)
- **Input:** Ticket requirements (business + technical)
- **Output:** Single JSON blob `execution_plan` with RITE `test_cases`
- **Constraints to reinforce:** min subtask counts, `assigned_to` enum, infra at `order_index=0` for devops
- **Target size:** 300–1,000 tickets worth of plans

#### Dev (`tdd-dev-tdd`)
- **Input:** PM handoff + subtask with test_cases from tool result
- **Output sequences:** `next_pending_subtask_in_project` → `fs_write` (test) → `run_tests` (fail) → `fs_write` (impl) → `run_tests` (pass) → `update_subtask_status(done)`
- **Include:** `shell_run` diagnostic branches, `blocked` status examples
- **Target size:** 1,000–5,000 subtask trajectories (highest volume)

#### Coordinator (`tdd-coordinator`)
- **Input:** `execution_plan` in state + system prompt
- **Output:** `save_execution_plan(...)` tool call with parsed subtasks
- **Target size:** 200–500 examples (easiest role — mechanical)

#### Researcher (`tdd-researcher`)
- **Input:** Research task handoff
- **Output:** `web_search` → `fs_write` (doc) → `rag_ingest_text` → optional `create_skill`
- **Target size:** 200–800 trajectories

### 6.3 Trace collection hooks (repo)

| Source | Location | Content |
|--------|----------|---------|
| Langfuse | optional `.env` | Full LLM spans |
| Agent logs | `backend/agent_logs/` | Event stream |
| Postgres tickets | subtasks with test_cases | Ground truth for Lead/Dev |
| Tests | `backend/tests/test_*.py` | Valid tool argument examples |
| Docs | `docs/tool-calling-and-prompts-analysis.md` | Failure modes to oversample |

**Suggested new instrumentation (for research agent to evaluate):**
- Export checkpoint + tool messages before truncation in `runner.py`
- Structured JSONL logger behind env flag `EXPORT_TRAINING_TRACES=1`

### 6.4 Data quality filters

Include only trajectories where:
- PM routing parsed successfully OR DB fallback matched intended route
- Dev subtask reached `done` with passing `run_tests` (or explicit `blocked` with stderr)
- Lead plan persisted without coordinator errors
- No human interrupt mid-trajectory (or segment before interrupt)

---

## 7. Training Stack Comparison

| Stack | Platform | Best for | Export to llama.cpp | Tool-call SFT | Est. cost |
|-------|----------|----------|---------------------|---------------|-----------|
| **mlx-tune** | M2 Max | Qwen 3.6 MoE, ≤14B coders | GGUF export built-in | Yes (SFT/DPO/GRPO) | Free (time) |
| **mlx-lm** | M2 Max | Manual LoRA, function-call templates | Via convert scripts | Yes | Free |
| **Unsloth + TRL** | Cloud NVIDIA | Coder-Next, 32B+ | Merge then GGUF | Yes, mature | $20–100/run |
| **HF Jobs + TRL** | Cloud | Hands-off training | Hub → GGUF | Yes | HF Pro + GPU |
| **Axolotl / LLaMA-Factory** | Cloud or Linux | Multi-method, Qwen templates | Yes | Yes | Self-hosted GPU |
| **No train — prompts only** | Any | Immediate gains | N/A | N/A | Free |

---

## 8. Tool-Calling Compatibility Checklist

Before merging any fine-tuned model, verify on `llama-server`:

- [ ] **Native tool calling:** Server started with tool-capable chat template (`--jinja` / model-specific flags).
- [ ] **Parallel tool calls:** LangGraph batches tool calls in one step (`runner.py`); model must emit multiple `tool_calls` when appropriate.
- [ ] **JSON mode vs tools:** Lead/PM JSON outputs — decide if trained as plain text JSON or separate no-tools model.
- [ ] **Schema fidelity:** UUIDs copied verbatim; enums match `AgentRole`, `SubtaskStatus`, `TicketStatus`.
- [ ] **Context length:** Dev traces are long — test at 8k–16k context.
- [ ] **Timeout:** `OPENAI_REQUEST_TIMEOUT=300` for slow local inference.
- [ ] **Structured output:** Consider `response_format` / JSON schema if llama-server build supports it for PM/Lead.

**Regression test suite (build from repo):**
1. `backend/tests/test_pm_handoff.py` — routing
2. `backend/tests/test_dev_subtask_resolution_gate.py` — dev gates
3. Manual: start run on seed project with local models only
4. Compare: tool calls per subtask, `% valid JSON`, time-to-done

---

## 9. Recommended Phased Plan

### Phase 0 — Baseline (1–2 days)
- [ ] Configure `OPENAI_BASE_URL` + role-specific models (§4.2)
- [ ] Set `WEB_SEARCH_PROVIDER=tavily`, `EMBEDDING_PROVIDER=local`
- [ ] Run 2–3 seed projects; record failure modes by role
- [ ] Enable Langfuse tracing

### Phase 1 — Prompt/tool hardening (no train)
- [ ] PM structured output (eliminate regex parse failures)
- [ ] Complete USE WHEN / AVOID WHEN on all tools
- [ ] Re-measure baseline metrics

### Phase 2 — Trace collection (1–2 weeks parallel use)
- [ ] Export 500+ successful dev trajectories
- [ ] Export 200+ PM routing decisions
- [ ] Export 100+ lead plans from ticket DB

### Phase 3 — First LoRA (Mac)
- [ ] **Dev adapter first** — highest call volume, clearest reward signal (`run_tests` pass/fail)
- [ ] Base: Qwen3-Coder-7B/14B 4-bit MLX (not 50 GB)
- [ ] 1–3 epochs, rank 8–16
- [ ] A/B vs base on 10 subtasks

### Phase 4 — PM + Lead adapters
- [ ] Train on JSON-heavy datasets
- [ ] Evaluate routing accuracy vs DB ground truth

### Phase 5 — Optional cloud pass
- [ ] If Mac adapters insufficient, HF Jobs on larger coder with same JSONL
- [ ] Merge → GGUF Q4_K_M → replace local Dev server

### Phase 6 — RAG embedding tune (optional)
- [ ] If skill retrieval weak, fine-tune MiniLM-style encoder on `_skills/` + project docs

---

## 10. Integration Back into tdd-agentic

After training:

```bash
# Example: Dev server on 8081 with fine-tuned coder GGUF
llama-server --host 0.0.0.0 --port 8081 -hf <user>/tdd-dev-qwen-coder-lora-merged:Q4_K_M

# .env — note: only one OPENAI_BASE_URL unless you extend llm.py for per-role URLs
# Workaround: run one model at a time, OR extend Settings with PM_OPENAI_BASE_URL etc.
PM_MODEL=openai/qwen3.6-instruct
DEV_MODEL=openai/tdd-dev-coder
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
```

**Gap to research:** `backend/agents/llm.py` supports a **single** `OPENAI_BASE_URL`. Multi-model local setup may require:
- Multiple ports + custom env vars per role, or
- One merged “do everything” model (worse quality), or
- OpenRouter pointing at local tunnels (hacky)

**Research task:** Design minimal patch for per-role `base_url` overrides.

---

## 11. Decision Matrix (Quick Reference)

| Your goal | Best option | Model |
|-----------|-------------|-------|
| Fastest time to value | A + B (prompts/skills) | Existing GGUFs |
| Cheapest training | C (mlx-tune LoRA) | Qwen 3.6 4-bit |
| Best Dev code+tools | E + F (coder adapter + traces) | Coder 7–14B, not 50 GB |
| Best PM routing JSON | E + F on instruct | Qwen 3.6 |
| Train the 50 GB coder | D (cloud Unsloth/HF Jobs) | Qwen3-Coder-Next |
| Better skill retrieval | H (embeddings) | MiniLM / small ST model |
| No training budget | A + J (4B specialists) | Multiple small GGUFs |

---

## 12. Open Research Questions (for follow-up agent)

1. **Exact model IDs:** What are the Hugging Face / local paths for “Qwen 3-coder-next” and “Qwen 3.6” on disk? Quantization? MoE vs dense?
2. **llama-server tool template:** Does the current local build emit OpenAI-compatible `tool_calls` for these Qwen versions?
3. **Per-role base URL:** Smallest code change in `llm.py` for PM on :8080 and Dev on :8081?
4. **Trace export:** Best hook point — Langfuse API vs new middleware in `runner.py`?
5. **Dataset size:** Minimum SFT examples per role before diminishing returns?
6. **DPO pairs:** Auto-generate rejected trajectories from common failures in `tool-calling-and-prompts-analysis.md`?
7. **Lead without tools:** Separate small JSON-only model vs shared instruct with zero tools bound?
8. **GRader model:** Worth fine-tuning 4B for CRAG relevance vs keeping Haiku cloud?
9. **Merge strategy:** Single multi-task LoRA vs per-role adapters on same base — interference risk?
10. **Evaluation harness:** Script to replay frozen ticket states against local model and score tool accuracy?

---

## 13. Key Repository Files

| File | Relevance |
|------|-----------|
| `backend/agents/llm.py` | Local OpenAI-compatible routing |
| `backend/agents/prompts.py` | Training prompt sources |
| `backend/agents/runner.py` | Tool loop, truncation, summaries |
| `backend/agents/project_manager/supervisor.py` | PM routing parse |
| `backend/tools/*.py` | Tool schemas for dataset synthesis |
| `backend/agents/skills/` | Skill injection behavior |
| `docs/tool-calling-and-prompts-analysis.md` | Known failure modes |
| `docs/agent-efficiency-plan.md` | Context/token optimization |
| `docs/agent-context-and-workflow.md` | Agent context model |
| `.env.example` | Local model configuration |

---

## 14. External References

- [mlx-tune (Apple Silicon LoRA)](https://github.com/ARahim3/mlx-tune)
- [Qwen function-call fine-tune examples](https://github.com/QwenLM/Qwen/blob/main/examples/function_call_finetune_examples.py)
- [MLX-LM function calling walkthrough](https://medium.com/@levchevajoana/fine-tuning-a-model-for-function-calling-with-mlx-lm-d00d587e2559)
- [Hugging Face TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer)
- [llama.cpp server OpenAI API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Unsloth (CUDA efficient LoRA)](https://github.com/unslothai/unsloth)

---

## 15. Suggested Deliverables from Research Agent

1. **Model identification sheet** — exact GGUF paths, params, quant, context limits for both local models.
2. **Tool-call smoke test report** — pass/fail for each role with current local setup.
3. **Training data spec** — JSONL schema + 10 gold example rows per role.
4. **Stack recommendation** — mlx-tune vs cloud for each adapter with cost/time estimate.
5. **Implementation PR plan** — per-role `base_url`, trace export flag, eval script.
6. **Go/no-go** — whether fine-tuning beats prompt fixes for your top-3 failure modes.

---

*Document version: 2026-06-20. Generated for research handoff; update after baseline local runs.*
