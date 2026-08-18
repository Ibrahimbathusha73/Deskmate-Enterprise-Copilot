# Deskmate — Enterprise Multi-Agent Knowledge & Ops Copilot

**One-liner:** An internal "ChatGPT for the company" that routes employee questions across specialized agents (docs RAG, structured data, visual documents, ticket triage) with production-grade evaluation, observability, and cost controls — not a chatbot wrapper, but a governed AI platform a real enterprise could actually run.

---

## 1. The Problem It Solves

Mid-size companies (200–5,000 employees) have knowledge scattered across Confluence/Notion pages, PDFs, spreadsheets, scanned receipts/forms, and a helpdesk ticket queue. Employees waste time hunting for answers, and support/ops teams re-answer the same questions. A single LLM chat window isn't enough because the underlying data is heterogeneous (text, tables, images, tickets) and enterprises need **auditability, cost control, and measurable accuracy** — not just a slick demo.

Deskmate is positioned as an **internal IT/HR/Ops copilot**: employees ask questions in natural language; a router agent decides whether the answer lives in a document, a spreadsheet, an image/scanned form, or requires opening/classifying a support ticket — then a specialist agent handles it, with every answer traceable to a source and every deployment gated by an eval suite.

---

## 2. Architecture — Multi-Agent Design (LangGraph)

```
                     ┌─────────────────────┐
   User query  ───▶  │  Orchestrator Agent  │  (LangGraph state graph)
                     └──────────┬───────────┘
                                │ intent + zero-shot classification
        ┌───────────┬──────────┼──────────┬────────────────┐
        ▼           ▼          ▼          ▼                ▼
    ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐ ┌───────────┐
    │ Docs RAG│ │ Table/SQL│ │ Vision │ │ Ticket     │ │ Tool/MCP  │
    │  Agent  │ │  Agent   │ │ Agent  │ │ Triage     │ │ Agent     │
    └─────────┘ └──────────┘ └────────┘ └────────────┘ └───────────┘
         │           │          │          │                │
         └───────────┴────┬─────┴──────────┴────────────────┘
                           ▼
                  Response Synthesizer + Citations
                           ▼
                  Confidence check → low? → human-in-the-loop escalation
```

- **Orchestrator Agent** — routes using zero-shot classification on intent + conversation state; LangGraph gives you explicit state, retries, and conditional edges (better portfolio signal than a single ReAct loop).
- **Docs RAG Agent** — hybrid search (dense + BM25) over Confluence/PDF/Notion exports; answers with inline citations.
- **Table/SQL Agent** — table question answering over spreadsheets and a Postgres warehouse; text-to-SQL with a validation step.
- **Vision Agent** — handles screenshots, scanned invoices/receipts, diagrams via document QA + visual document retrieval + OCR fallback.
- **Ticket Triage Agent** — classifies and routes incoming support requests (zero-shot classification + text classification for priority/sentiment).
- **Tool/MCP Agent** — calls external systems (calendar, GitHub issues, internal APIs) through MCP servers, demonstrating standardized tool-use rather than hand-rolled function calling.
- **Human-in-the-loop gate** — if retrieval confidence or eval-time faithfulness score is low, escalate instead of hallucinate. This feature is impressive because it shows you understand LLM failure modes.

---

## 3. Hugging Face Tasks Used

| Task | Where it's used |
|---|---|
| Document Question Answering | Docs RAG agent over PDFs/policies |
| Visual Document Retrieval | Retrieving scanned forms/invoices by visual layout |
| Image-Text-to-Text / Visual QA | Vision agent reading screenshots, receipts, diagrams |
| Table Question Answering | Structured-data agent over spreadsheets |
| Zero-Shot Classification | Intent routing + ticket triage |
| Text Classification | Ticket priority/sentiment tagging |
| Summarization | Long-thread and meeting-notes condensation |
| Sentence Similarity / Feature Extraction | Embedding generation for hybrid retrieval |
| Text Ranking | Re-ranking retrieved chunks before generation |
| Fill-Mask / Token Classification | PII redaction pass before storage (compliance angle) |

Using 4–6 of these well (not all 10+) is more credible than checkbox-stuffing — pick Document QA, Visual Document Retrieval, Table QA, Zero-Shot Classification, Sentence Similarity, and Text Ranking as your core six.

---

## 4. Recommended Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | explicit multi-agent state machine, industry-standard for 2026 JDs |
| Backend API | **FastAPI** | async, OpenAPI docs auto-generated, industry default |
| Tool standard | **MCP (Model Context Protocol)** | shows you're current with the 2025–26 tool-use standard, not just LangChain tools |
| Vector DB | **Qdrant** (or Weaviate) | native hybrid dense+sparse search, self-hostable via Docker |
| Embeddings | `sentence-transformers` / HF Feature Extraction | open-source, cost-controlled |
| Re-ranking | Cross-encoder (HF Text Ranking task) | measurable recall/precision lift, good eval story |
| Doc parsing | `unstructured.io`, Donut/LayoutLM for visual docs | handles PDFs, scans, tables uniformly |
| LLMs | Claude (tiered: Haiku for routing/simple, Sonnet for synthesis) | cost-optimization narrative for resume |
| Eval | **RAGAS** + custom golden-set harness + `promptfoo` for regression | this is the differentiator vs. a demo |
| Observability | **Langfuse** or LangSmith | trace every agent hop, token cost, latency |
| Guardrails | Llama Guard / NeMo Guardrails | PII leakage + prompt-injection defense |
| Caching | Redis semantic cache | cost control, demoable latency win |
| CI/CD | GitHub Actions → Docker → (optional) k8s manifests | eval suite runs as a CI gate on every PR |
| Frontend | Next.js chat UI (or Streamlit for MVP) | keep this thin; it's not the point of the project |

---

## 5. What Makes This "Production-Grade" (not a wrapper)

1. **Eval-gated deployments** — a golden set of ~100–150 Q&A pairs across all agent types; RAGAS scores (faithfulness, answer relevancy, context precision/recall) computed in CI; a PR that drops faithfulness below a threshold fails the build.
2. **Model tiering for cost** — router uses Haiku for classification/routing, Sonnet for final synthesis; log $ per query.
3. **Hybrid search + re-ranking** — dense + BM25 fusion, then cross-encoder re-rank; benchmark recall@5 against dense-only baseline.
4. **Human-in-the-loop escalation** — low-confidence answers routed to a queue instead of hallucinated.
5. **Observability** — full trace per request (agent path, retrieved chunks, token cost, latency) in Langfuse.
6. **Guardrails** — PII redaction on ingestion, prompt-injection tests as part of the eval suite (adversarial red-team set).
7. **Regression testing** — every prompt/agent change re-runs the eval harness before merge.

---

## 6. Real-World Data / APIs

Avoid synthetic-only data — pull from at least one real source to make the demo credible:
- **GitHub Issues API** — stand in for an internal ticket queue (real, freely available, structured).
- **SEC EDGAR / public company 10-Ks** — realistic long-document corpus for the Docs RAG agent.
- **A public invoice/receipt dataset** (e.g., SROIE) — for the Vision agent's document QA.
- Optionally layer in **your own synthetic "company handbook"** to show you can handle proprietary-style content too.

---

## 7. Suggested 1–3 Month Roadmap

- **Weeks 1–2:** Ingestion pipeline (PDF/GitHub Issues/receipts) + hybrid vector store + baseline single-agent RAG.
- **Weeks 3–4:** LangGraph orchestrator + specialist agents (table, vision, ticket triage) + MCP tool integration.
- **Weeks 5–6:** Eval harness (golden set, RAGAS, promptfoo regression) wired into GitHub Actions.
- **Weeks 7–8:** Observability (Langfuse), semantic caching, model tiering, guardrails/PII redaction.
- **Weeks 9–10:** Human-in-the-loop escalation flow, re-ranking, cost/latency dashboards.
- **Weeks 11–12:** Polish frontend, write the case-study README with before/after eval numbers, record a demo video, Dockerize + deploy (Fly.io/Render/small k8s cluster).

---

## 8. Mapping to Real AI Engineering JDs

| JD phrase you'll see | What in Deskmate proves it |
|---|---|
| "Build and maintain RAG pipelines" | Hybrid search + re-ranking + citation grounding |
| "Experience with agentic workflows / LangGraph / multi-agent systems" | Orchestrator + 5 specialist agents with explicit state |
| "LLM evaluation and observability (LLMOps)" | RAGAS golden set in CI, Langfuse tracing |
| "Tool use / function calling / MCP" | MCP-based tool agent |
| "Cost and latency optimization" | Model tiering + semantic cache with logged $/query |
| "Responsible AI / guardrails" | PII redaction, prompt-injection red-team eval |
| "Cross-functional collaboration, ship to production" | Docker/CI-CD pipeline, deployed demo, case-study writeup |

---

## 9. Resume-Ready Impact Lines (fill in real numbers once built)

- *"Designed and shipped a multi-agent enterprise copilot (LangGraph, FastAPI, Qdrant) routing queries across 5 specialist agents; achieved 90%+ faithfulness on a 150-example RAGAS golden set, gated via CI."*
- *"Improved retrieval recall@5 by ~30% over dense-only baseline using hybrid BM25+dense search with cross-encoder re-ranking."*
- *"Cut average per-query LLM cost by ~40% via model tiering (Haiku for routing, Sonnet for synthesis) and Redis semantic caching."*
- *"Built an automated eval + regression suite (RAGAS, promptfoo) integrated into GitHub Actions, blocking prompt/agent regressions before merge."*
- *"Implemented human-in-the-loop escalation for low-confidence responses, reducing hallucinated answers to 0 in adversarial red-team testing."*

---

## 10. Portfolio Presentation Tips

- Ship a **1–2 page case study** (problem → architecture diagram → eval numbers before/after → cost numbers → what you'd do at 10x scale).
- Record a **3-minute demo video** showing the router picking different agents for different question types — interviewers rarely read code but will watch this.
- Put the **eval dashboard front and center** in your README; it's the single biggest differentiator between "I called an API" and "I built an AI system."tions, blocking prompt/agent regressions before merge."*
- *"Implemented human-in-the-loop escalation for low-confidence responses, reducing hallucinated answers to 0 in adversarial red-team testing."*

---

## 10. Portfolio Presentation Tips

- Ship a **1–2 page case study** (problem → architecture diagram → eval numbers before/after → cost numbers → what you'd do at 10x scale).
- Record a **3-minute demo video** showing the router picking different agents for different question types — interviewers rarely read code but will watch this.
- Put the **eval dashboard front and center** in your README; it's the single biggest differentiator between "I called an API" and "I built an AI system."
