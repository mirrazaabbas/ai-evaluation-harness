# Architecture — AI Evaluation Harness

## Current pipeline

```text
Benchmark JSON
    ↓
Case validation
    ↓
Metric functions
    ├─ required-term recall
    ├─ groundedness proxy
    └─ concision
    ↓
Weighted per-case score
    ↓
Aggregate average + pass rate
```

## Design goals

- Keep metrics deterministic and reproducible.
- Separate case loading, scoring, and report aggregation.
- Make evaluation datasets easy to extend.
- Fail clearly on malformed cases and invalid thresholds.

## Production target

```text
Candidate model/prompt
        ↓
Benchmark runner
        ↓
Deterministic metrics + semantic metrics + LLM judge
        ↓
Latency / tokens / cost / citations / groundedness
        ↓
Baseline comparison
        ↓
Regression gate
        ↓
JSON + HTML report
```

## Recommended future metrics

- semantic answer similarity
- retrieval recall@k
- citation precision/recall
- hallucination rate
- tool-call correctness
- latency percentiles
- token usage and estimated cost
- baseline-vs-candidate deltas
