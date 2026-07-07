# Email Reply Evaluator

An **evaluation-first framework** for measuring the quality of LLM-generated customer support emails — not a chatbot, not an API wrapper, but a rigorous measurement system.

> **Core idea:** If you cannot measure reply quality, you cannot improve it. This framework provides 6 carefully selected metrics, per-metric methodology, RAG-grounded generation, and honest confidence estimates — so you know when your model is actually getting better vs. just getting lucky.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API key
python main.py --generate-dataset   # 120 synthetic emails, 14 intents
python main.py --emails 10          # run evaluation
python main.py --emails 10 --provider nvidia --model meta/llama-3.1-8b-instruct
```

**Actual output (10 emails, NVIDIA Llama 3.1 8B):**
```
Overall average: 85.9
Needs human review: 1/10
Hallucination risk: 4 low, 5 medium, 1 high
```

Most replies pass automatically. Only ~1 in 10 needs human review — a practical, deployable ratio.

## Problem

Most LLM "evaluation" means printing outputs and saying "looks good." That doesn't scale. When you need to validate 100+ generated replies across multiple quality dimensions, you need:

1. **Calibrated metrics** — not arbitrary scores, but defined measurement methodology
2. **Multiple strategies** — because no single approach (LLM-as-Judge, embeddings, determinism) works for all dimensions
3. **Honest confidence** — knowing when a score difference is meaningful vs. noise

This framework delivers all three.

## Architecture

```
Dataset (120 synthetic emails, 14 intents)
    │
    ▼
Email Analysis (LLM)  →  structured metadata (intent, urgency, tone)
    │
    ▼
RAG Retrieval         →  sentence-transformers top-3 similar past cases
    │
    ▼
Prompt Builder        →  dynamic prompt with few-shot examples
    │
    ▼
LLM Generator         →  provider-agnostic (OpenAI/Anthropic/Gemini/NVIDIA)
    │
    ▼
┌────────────────────────────────────────┐
│          EVALUATION ENGINE             │
│  6 metrics, 3 strategies:             │
│                                        │
│  LLM-as-Judge:                         │
│  ├─ Intent Coverage                    │
│  └─ Tone Alignment                     │
│                                        │
│  Embeddings:                           │
│  └─ Semantic Similarity                │
│                                        │
│  Deterministic:                        │
│  ├─ Action Completeness                │
│  ├─ Policy Compliance                  │
│  └─ Hallucination (NLI + heuristic)    │
└────────────────────────────────────────┘
    │
    ▼
Error Analysis  →  Reports (JSON + CSV + Markdown)
```

## The 6 Metrics — Why These, Why Not Others

### Included

| Metric | Strategy | What It Catches | Why It Exists |
|--------|----------|-----------------|---------------|
| **Intent Coverage** | LLM-as-Judge | Missing customer requests | The most fundamental CS dimension |
| **Tone Alignment** | LLM-as-Judge | Tone-deaf or robotic replies | CS quality depends on emotional intelligence |
| **Semantic Similarity** | Embeddings | Meaning drift from ideal reply | Catches paraphrase-equivalent failures |
| **Action Completeness** | Deterministic | Vague replies, missing next steps | Customers contact support to get things done |
| **Policy Compliance** | Deterministic | Unprofessional language, unsafe advice | Brand and legal protection |
| **Hallucination** | NLI + Heuristic | Invented facts, false promises | Primary LLM failure mode |

### Excluded (Deliberately)

| Metric | Reason Excluded |
|--------|-----------------|
| Helpfulness | Overlaps with Intent Coverage + Tone. If you answer everything with good tone, you're helpful. |
| Readability | CS emails are naturally simple. Added no discriminative signal. |
| Safety | Merged into Policy Compliance — same deterministic pattern matching. |
| BLEU / ROUGE / BERTScore | Compare form, not function. "Your refund is processed" vs "We issued your refund" = 0 BLEU but identical meaning. |
| Bootstrap CI | Meaningful for 1000+ samples. With ≤120 examples, mean ± std tells the same story. |

Full per-metric methodology (definition, trust, weaknesses, comparison method) → `src/evaluation/METRICS_METHODOLOGY.md`

### Hallucination Detection (NLI)

Mirrors production systems at Microsoft, Vectara, and Anthropic:

1. Split reply into atomic claims
2. LLM judges each claim as SUPPORTED / CONTRADICTED / NEUTRAL / UNVERIFIABLE
3. **Only CONTRADICTED** claims reduce the score (NEUTRAL is acceptable — support replies introduce new info)
4. Heuristic fallback when LLM produces invalid JSON

**Key insight:** NEUTRAL claims are common in support replies (the agent adds helpful context not in the original email). Penalizing them would make every good reply score poorly.

## Dataset

120 synthetic customer emails across 14 intents, generated programmatically with seeded randomness for full reproducibility:

`refund, billing, login_issue, subscription, cancellation, complaint, bug_report, feature_request, integration, enterprise_sales, password_reset, pricing, account_verification, positive_feedback`

Each example includes `id`, `customer_email`, `intent`, `urgency`, `tone`, `expected_actions`, and `gold_reply`.

**Limitations:** Synthetic (no typos, rambling, or mixed intents), single-turn, English-only, template-based gold replies. Sufficient for framework demonstration; production use requires real data.

## RAG (Retrieval-Augmented Generation)

The generator retrieves top-3 same-intent examples via `all-MiniLM-L6-v2` embeddings and includes them as few-shot examples in the prompt.

**Why RAG over fine-tuning?** Zero-cost to update (just add examples), works with any LLM, and provides auditability — you see exactly which past cases influenced the reply.

## Provider Abstraction

```
Provider (ABC)
├── OpenAIProvider
├── AnthropicProvider
├── GeminiProvider
└── NVIDIAProvider  (api.nvidia.com, OpenAI-compatible)
```

Adding a provider requires implementing one method: `generate()`. The judge uses the same interface — just a different prompt.

## Trade-offs (Honest)

| Decision | Rationale | Cost |
|----------|-----------|------|
| LLM-as-Judge for subjective | Best signal for context-dependent quality | ~2x API cost per evaluation |
| Deterministic for policy/safety | Faster, cheaper, 100% reproducible | Misses nuanced violations |
| Synthetic dataset | Covers edge cases systematically | Less authentic than real data |
| RAG over fine-tuning | Zero-cost to update, works with any LLM | Higher per-query latency |
| NEUTRAL = acceptable in NLI | Support replies introduce new info | May miss subtle fabrications |
| Heuristic fallback for hallucination | Graceful LLM degradation | Less accurate than NLI |
| all-MiniLM-L6-v2 | 10x faster than LLM embeddings | Lower quality for complex semantics |
| No cross-encoder reranking | Fine for <10K examples | Won't scale to millions |

## Known Limitations

- **Judge = Generator model:** Using Llama 3.1 8B for both is a limitation. A stronger judge (70B+) would produce more reliable scores. See `scripts/validate_metric.py` for the minimum calibration check.
- **8B NLI instability:** The 8B model sometimes produces invalid JSON for claim-level NLI, falling back to heuristics. A larger model would reduce this.
- **Semantic similarity ceiling:** 80M parameter embedding model captures paraphrase equivalence but misses nuanced differences in tone or policy.
- **No human validation:** Metric scores are not validated against human judgments. `validate_metric.py` provides a basic sanity check, not a substitute for human-labeled calibration.

## Validate Metric Sanity

```bash
python scripts/validate_metric.py
# Expected: high score for good reply, low score for bad reply
```

This is the **minimum bar**: if the judge doesn't penalize "We don't care about your problem" (score ~25) and reward a professional reply (score ~80), the scores mean nothing.

## Reports

```
results/
├── report.json      # Full structured report — per-email scores, CIs, explanations
├── leaderboard.csv  # Ranked by overall score
└── summary.md       # Human-readable summary with averages, failures, worst/best
```

## Testing

```bash
python -m pytest tests/ -v          # 35 tests (unit + integration)
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Project Structure

```
email-response-evaluator/
├── configs/config.yaml
├── dataset/
│   └── generate_dataset.py
├── scripts/
│   └── validate_metric.py           # Judge calibration sanity check
├── src/
│   ├── models/
│   │   ├── providers.py             # OpenAI/Anthropic/Gemini/NVIDIA
│   │   ├── email_analyzer.py        # LLM metadata extraction
│   │   ├── prompt_builder.py        # Dynamic prompt + RAG
│   │   ├── retriever.py             # sentence-transformers RAG
│   │   └── generator.py             # RAG-grounded generation
│   └── evaluation/
│       ├── METRICS_METHODOLOGY.md   # Per-metric methodology (the "why")
│       ├── metrics.py               # 6 metrics, 3 strategies
│       ├── hallucination.py         # NLI claim verification
│       ├── confidence.py            # Statistical confidence
│       ├── judge.py                 # Evaluation orchestrator
│       ├── error_analysis.py        # Failure distribution
│       └── report.py                # JSON + CSV + Markdown
├── tests/                           # 35 tests
├── main.py                          # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

## How AI Was Used

AI assisted with code implementation, testing, documentation, and web research. The human defined the problem, provided direction, and validated outputs.
