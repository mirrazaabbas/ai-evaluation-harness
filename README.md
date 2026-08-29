# AI Evaluation Harness

[![CI](https://github.com/mirrazaabbas/ai-evaluation-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/mirrazaabbas/ai-evaluation-harness/actions/workflows/ci.yml)

A deterministic Python benchmarking framework for scoring AI-system outputs, tracking operational metadata, comparing a candidate against a baseline, and producing machine-readable or HTML reports.

## Implemented metrics and reporting

- Required-term recall
- Context-overlap groundedness proxy
- Concision
- Optional expected-citation coverage
- Weighted per-case score
- Configurable pass threshold
- Dataset-level average score and pass rate
- Optional average latency aggregation
- Optional average cost aggregation
- Baseline-vs-candidate score deltas
- JSON output
- Standalone HTML report generation

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current pipeline and remaining production roadmap.

## Run a benchmark

```bash
python evaluate.py sample_cases.json
```

Write an HTML report:

```bash
python evaluate.py sample_cases.json --html evaluation-report.html
```

Compare a candidate dataset with a baseline:

```bash
python evaluate.py candidate_cases.json --baseline baseline_cases.json --html comparison.html
```

Use a custom pass threshold:

```bash
python evaluate.py sample_cases.json --threshold 0.75
```

## Dataset fields

Each case requires `id` and `output`. Optional fields include:

- `context`
- `required_terms`
- `expected_citations`
- `max_words`
- `latency_ms`
- `cost_usd`

Malformed cases and invalid negative operational metrics fail validation instead of being silently accepted.

## Design goals

- Reproducible deterministic evaluation
- Clear case validation
- Extensible benchmark datasets
- Separation between metrics, aggregation, comparison, and reporting
- CI-friendly regression checks
- HTML escaping for safe report rendering

## Quality checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
coverage run -m unittest discover -s tests -v
coverage report --fail-under=85
python evaluate.py sample_cases.json --html /tmp/evaluation-report.html
```

CI runs compile, lint, coverage, JSON-report, and HTML-report checks on Python 3.10–3.12.

## Remaining roadmap

- Semantic answer similarity
- Retrieval recall@k / MRR / nDCG
- Citation precision/recall against structured source metadata
- Hallucination/factuality evaluation
- Tool-call correctness metrics
- Token accounting from real model runs
- RAGAS/DeepEval or similar framework adapters
- Optional LLM-as-a-judge with calibrated rubrics
- Candidate regression policy that can fail CI on measured model-quality deltas

## Skills demonstrated

Python · LLM Evaluation · Benchmarking · Regression Testing · Quality Engineering · Metrics · HTML Reporting · JSON · Testing · CI/CD

## Current scope

The current quality scores are deterministic proxies. Latency and cost are aggregated when supplied by an upstream run; the harness does not invent those measurements. It does not yet claim semantic factuality judging, real model cost collection, or production LLM-as-a-judge evaluation.
