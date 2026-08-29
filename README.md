# AI Evaluation Harness

A deterministic Python benchmarking framework for scoring AI-system outputs against repeatable quality criteria.

## Current metrics

- Required-term recall
- Groundedness proxy based on context overlap
- Concision
- Weighted per-case score
- Dataset-level average score
- Pass rate

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current pipeline and production roadmap.

## Run

```bash
python evaluate.py sample_cases.json
```

## Design goals

- Reproducible evaluation
- Clear case validation
- Extensible benchmark datasets
- Separation between metric functions and reporting
- CI-friendly regression checks

## Roadmap

- Semantic similarity
- Citation correctness
- Hallucination/factuality metrics
- Retrieval recall@k
- Latency, token, and cost tracking
- Baseline-vs-candidate comparison
- HTML reports
- CI regression thresholds

## Skills demonstrated

Python · LLM Evaluation · Benchmarking · Quality Engineering · Metrics · JSON · Testing

## Current scope

The current metrics are deterministic proxies. They are intentionally presented as a foundation rather than claiming full production-grade LLM judging or factuality evaluation.
