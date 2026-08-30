# Advanced Evaluation Upgrade

This branch extends the deterministic evaluation harness with CI-friendly metrics for RAG and agent systems.

Implemented in this upgrade:
- token-level deterministic answer F1
- retrieval recall@k
- reciprocal rank
- nDCG@k
- citation precision and recall
- exact ordered tool-call accuracy
- baseline-vs-candidate regression policy for quality, latency and cost
- validation of regression thresholds
- CI coverage for all new metrics across Python 3.10–3.12

The new metrics remain provider-independent and require no model credentials, which keeps regression testing reproducible in CI.
