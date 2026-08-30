# AI Evaluation Harness

[![CI](https://github.com/mirrazaabbas/ai-evaluation-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/mirrazaabbas/ai-evaluation-harness/actions/workflows/ci.yml)

A deterministic Python benchmarking framework for measuring AI-system quality, retrieval behavior, citations, tool calls, latency/cost metadata, and baseline-vs-candidate regressions. It is designed to run in CI without requiring model credentials.

## Implemented evaluation capabilities

### Core benchmark metrics

- Required-term recall
- Context-overlap groundedness proxy
- Concision
- Expected-citation coverage
- Weighted per-case score
- Configurable pass threshold
- Dataset-level average score and pass rate
- Optional latency aggregation
- Optional cost aggregation
- Baseline-vs-candidate score deltas
- JSON output
- Standalone HTML reporting

### Advanced RAG and agent metrics

`advanced_metrics.py` adds deterministic metrics for deeper system evaluation:

- Token-level answer F1
- Retrieval recall@k
- Reciprocal rank / MRR-compatible scoring
- nDCG@k
- Citation precision
- Citation recall
- Exact ordered tool-call accuracy
- Configurable quality-regression policies
- Maximum average-score drop
- Maximum pass-rate drop
- Maximum latency increase ratio
- Maximum cost increase ratio

## Cross-project evaluation

The repository consumes the same `portfolio-evidence/v1` JSON contract produced by the RAG and Agent projects.

```text
RAG Knowledge Assistant
       ↓
Agent Workflow Engine
       ↓ portfolio-evidence/v1
AI Evaluation Harness
       ↓
quality + retrieval + citation + tool metrics
```

`portfolio_bridge.py` validates the contract and evaluates:

- answer similarity
- retrieval quality
- citation behavior
- tool-call correctness
- pass/fail threshold
- reported latency metadata

A fully credential-free integration sample is committed to the repository:

```bash
python portfolio_bridge.py \
  sample_portfolio_run.json \
  sample_portfolio_expected.json \
  --output portfolio-evaluation.json
```

This command is also executed in CI.

## Run the core benchmark

```bash
python evaluate.py sample_cases.json
```

Write an HTML report:

```bash
python evaluate.py sample_cases.json --html evaluation-report.html
```

Compare a candidate dataset with a baseline:

```bash
python evaluate.py candidate_cases.json \
  --baseline baseline_cases.json \
  --html comparison.html
```

Use a custom pass threshold:

```bash
python evaluate.py sample_cases.json --threshold 0.75
```

## Core dataset fields

Each core benchmark case requires `id` and `output`. Optional fields include:

- `context`
- `required_terms`
- `expected_citations`
- `max_words`
- `latency_ms`
- `cost_usd`

Malformed cases and invalid negative operational metrics fail validation rather than being silently accepted.

## Design goals

- Reproducible deterministic evaluation
- Clear schema validation
- Credential-free CI execution
- Extensible benchmark datasets
- Separation between metrics, aggregation, comparison and reporting
- Cross-project evidence compatibility
- CI-friendly regression checks
- Safe HTML escaping

## Quality checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
coverage run -m unittest discover -s tests -v
coverage report --fail-under=85
python evaluate.py sample_cases.json --html /tmp/evaluation-report.html
python portfolio_bridge.py sample_portfolio_run.json sample_portfolio_expected.json
```

CI runs compile, lint, branch coverage, JSON reporting, HTML reporting, and the cross-project evidence evaluation on Python 3.10–3.12.

## Dependency maintenance

Dependabot is configured for weekly Python and GitHub Actions dependency updates.

## Remaining roadmap

The deterministic metrics are deliberately transparent and reproducible. Additional optional layers can be added without replacing them:

- Embedding-based semantic answer similarity
- Calibrated hallucination/factuality evaluation
- Optional LLM-as-a-judge with versioned rubrics
- RAGAS/DeepEval-style adapter layer
- Token accounting captured directly from real model runs
- Larger committed benchmark suites for RAG, agent routing, prompt injection and tool use
- Trend/history storage across repeated benchmark runs

## Skills demonstrated

Python · LLMOps · AI Evaluation · RAG Evaluation · Retrieval Metrics · Agent Evaluation · Tool-call Evaluation · Regression Testing · Benchmarking · Quality Engineering · HTML Reporting · JSON · Testing · CI/CD

## Scope and evidence

The repository implements deterministic quality, retrieval, citation, tool-call and regression measurements. Latency and cost are evaluated when supplied by the upstream system; they are not fabricated. The project does not claim semantic factuality judging or a production LLM judge unless those optional evaluators are explicitly added and tested.
