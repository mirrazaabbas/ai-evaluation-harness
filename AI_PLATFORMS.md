# AI platform compatibility

The deterministic evaluation metrics work without any model provider. An optional LLM judge uses the shared `AIClient` interface and can run with OpenAI/OpenAI-compatible APIs, Anthropic Claude, or Google Gemini.

## Offline verification

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

These tests do not use live API credits.

## Provider selection

```bash
# OpenAI or OpenAI-compatible
export AI_PROVIDER=openai
export AI_API_KEY="YOUR_KEY"
export AI_MODEL="YOUR_CHAT_MODEL"
# Optional: export AI_BASE_URL="https://provider.example/v1"
```

```bash
# Anthropic Claude
export AI_PROVIDER=anthropic
export AI_API_KEY="YOUR_KEY"
export AI_MODEL="YOUR_CLAUDE_MODEL"
```

```bash
# Google Gemini
export AI_PROVIDER=gemini
export AI_API_KEY="YOUR_KEY"
export AI_MODEL="YOUR_GEMINI_MODEL"
```

## Run deterministic metrics

```bash
python evaluate.py sample_cases.json
```

## Run the optional AI judge

```bash
python - <<'PY'
from ai_features import judge_case
from ai_platform import create_ai_client
from evaluate import load_cases, DEFAULT_DATASET

case = load_cases(DEFAULT_DATASET)[0]
print(judge_case(case, create_ai_client()))
PY
```

Use the LLM judge as an additional qualitative signal, not as a replacement for deterministic regression metrics.
