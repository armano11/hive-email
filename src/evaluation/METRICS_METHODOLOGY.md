# Evaluation Methodology

## Metric Selection Rationale

The challenge says: *"What makes an AI reply good? Why is this metric appropriate? Why should a reviewer trust this score? What are the weaknesses of this metric? How does it compare generated replies against reference replies?"*

We answer those questions per-metric below.

### Why Not BLEU / ROUGE / BERTScore?

| Metric | Problem for Customer Support |
|--------|-----------------------------|
| BLEU | Punishes valid rephrasing. "Your refund is processed" vs "We issued your refund" = low BLEU despite identical meaning |
| ROUGE | Same limitation as BLEU. Measures n-gram overlap, not meaning |
| BERTScore | Better than BLEU/ROUGE but requires a reference. Does not measure business-relevant dimensions (tone, policy, safety) |

All three share the same flaw: they compare form, not function. A reply can have perfect BLEU but be unsafe, and a reply with zero n-gram overlap can perfectly resolve the customer's issue. For customer support evaluation, functional quality matters more than lexical similarity.

---

## Metric 1: Intent Coverage

### What makes a reply "good"?
A good reply addresses every request and question the customer raised. If the customer asks three questions, all three are answered. If they request an action, the reply confirms it.

### Why is this metric appropriate?
Intent coverage is the most fundamental dimension of customer support quality. A perfectly written reply that ignores half the customer's requests is a failed reply. This metric directly measures whether the reply solves the customer's stated problems.

### Why should a reviewer trust this score?
The LLM judge is given both the customer email and the generated reply, with explicit instructions to identify whether each customer request was addressed. The output includes a list of missed items (`missing`), making the score auditable.

### Weaknesses
- Requires an LLM call (cost, latency)
- The judge model may miss subtle requests
- Overlapping or implicit requests are harder to detect
- Uses the same model family as the generator (though separate instance)

### Comparison Methodology
The judge receives: `(customer email, generated reply, expected actions)`. It checks each expected action against the reply. The score reflects the fraction of customer requests that are explicitly addressed.

---

## Metric 2: Tone Alignment

### What makes a reply "good"?
The tone matches what the situation demands: empathetic when the customer is frustrated, clear when they are confused, warm when they are appreciative, professional when they are neutral.

### Why is this metric appropriate?
Customer support quality depends heavily on emotional intelligence. A technically correct reply delivered in the wrong tone can escalate frustration. Measuring tone alignment captures this human dimension.

### Why should a reviewer trust this score?
The judge receives the customer's detected emotion (from the email analyzer) and evaluates whether the reply's tone is appropriate for that emotion. The output includes the observed tone, making the judgment transparent.

### Weaknesses
- Highly subjective — different evaluators may disagree
- The judge model's tone perception may not match human perception
- Cultural differences in tone appropriateness are not modeled

### Comparison Methodology
The judge receives: `(customer email, customer emotion from analyzer, generated reply)`. It rates appropriateness based on the detected emotion and the reply's observed tone.
---

## Metric 3: Semantic Similarity

### What makes a reply "good"?
The reply conveys the same meaning as the gold/ideal reply, even if phrased differently. High semantic similarity means the generated reply captures the correct content.

### Why is this metric appropriate?
When a gold standard exists (as it does in our dataset), comparing meaning — not wording — is the right approach. Semantic similarity via sentence embeddings captures paraphrase equivalence naturally, avoiding the brittleness of n-gram metrics.

### Why should a reviewer trust this score?
sentence-transformers (`all-MiniLM-L6-v2`) generates fixed-length embeddings that capture sentence-level meaning. Cosine similarity between normalized embeddings is a well-understood, reproducible measure. The model is trained on NLI and STS datasets specifically for semantic equivalence.

### Weaknesses
- Requires a gold reply (not always available)
- The embedding model (80M parameters) is smaller than modern LLM-based embeddings
- Semantic similarity does not capture tone or safety
- Position bias: longer replies with more content have higher chance of overlap

### Comparison Methodology
Both the generated and gold reply are encoded with `all-MiniLM-L6-v2`. Cosine similarity is computed on the normalized embeddings. The score is the similarity mapped to 0-100.

---

## Metric 4: Action Completeness

### What makes a reply "good"?
The reply contains specific action language, provides timelines when applicable, and invites follow-up contact. It tells the customer what will happen next.

### Why is this metric appropriate?
Customers contact support because they want something done. The reply should clearly communicate actions taken or planned. This is a well-defined deterministic check — no subjectivity.

### Why should a reviewer trust this score?
The rules are explicit and auditable: does the reply contain action verbs? Does it mention a timeline? Does it invite follow-up? Any developer can inspect the logic and understand exactly why a score is what it is.

### Weaknesses
- Heuristic — misses nuanced action language
- Does not verify whether the actions are correct, only that action language exists
- Overly generous if the reply contains action phrases but wrong actions

### Comparison Methodology
Deterministic phrase matching against a curated list of action signals, timeline keywords, and contact invitations. The reply does not need a reference — this is a self-contained quality check.
---

## Metric 5: Policy Compliance

### What makes a reply "good"?
The reply contains no unprofessional language, no unsupported promises, and no guarantees that the company cannot fulfill.

### Why is this metric appropriate?
Customer support replies are subject to brand guidelines and legal requirements. A reply that promises "guaranteed" results or uses "not my problem" creates liability and damages trust.

### Why should a reviewer trust this score?
The rules are explicit pattern matches against a curated list of flagged phrases. The output lists every issue found. The logic is simple, auditable, and 100% deterministic.

### Weaknesses
- Misses context-dependent violations that don't match the phrase list
- False positives if flagged words are used innocuously
- Limited to English pattern matching

### Comparison Methodology
Self-contained. Searches the reply for unprofessional phrases and unsupported promises using substring matching against a curated blocklist.
---

## Metric 6: Hallucination Detection

### What makes a reply "good"?
Every claim in the reply is supported by or at least not contradicted by the customer's original email. The reply does not invent facts, timelines, or promises.

### Why is this metric appropriate?
Hallucination is the primary failure mode of LLM-generated content. For customer support, a hallucinated promise (e.g., "I've issued a refund" when the agent cannot do so) creates real customer anger and legal exposure.

### Why should a reviewer trust this score?
The NLI-based approach extracts atomic claims from the reply and evaluates each claim individually against the source email. The output lists every claim with its verdict (SUPPORTED / CONTRADICTED / NEUTRAL / UNVERIFIABLE) and an explanation. When the LLM judge is unavailable, a heuristic fallback provides a conservative estimate.

### Weaknesses
- The LLM judge used for NLI (same 8B model as the generator) may not be reliable enough for fine-grained NLI — ideal would be a larger dedicated NLI model or a human-in-the-loop
- NEUTRAL claims (new information not in the email) are common in support replies and are not penalized heavily, but some genuinely hallucinated claims may be misclassified as NEUTRAL
- A single CONTRADICTED claim drops the score significantly, which is intentional but may be harsh for minor errors

### Comparison Methodology
The reply is split into atomic claims (by sentence). Each claim is evaluated by an LLM judge against the original customer email. Verdicts are aggregated into a hallucination risk (low/medium/high) and a numerical score (0-100).

---

## Score Aggregation

The overall score is the unweighted mean of all available metric scores. Individual metric scores are kept separate so reviewers can see the profile: a reply might score 95 on intent but 45 on policy compliance.

Human review is triggered if:
- Hallucination risk is **high**
- Intent coverage is below **50**
- Safety score is below **80**
- Overall score is below **50**

This creates a practical human-review workflow: most replies pass automatically; only outliers need manual inspection.
