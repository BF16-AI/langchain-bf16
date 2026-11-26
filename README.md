# langchain-bf16

Official [LangChain](https://python.langchain.com/) integration for [**bf16.ai**](https://bf16.ai) — 1M-context uncensored open-weight LLMs with strict Zero Data Retention (`ZDR`) and behavioral verification gates.

## Installation

```bash
pip install langchain-bf16
```

## Quick Start

```python
from langchain_bf16 import ChatBF16, BF16_MODELS

llm = ChatBF16(model=BF16_MODELS["deepseek_v4_1_flash"])
resp = llm.invoke("Audit this TypeScript auth middleware for race conditions.")
print(resp.content)
```

Set `BF16_API_KEY` (or pass `api_key=...`). You can also use `"bf16/auto"`, `"bf16/auto:fast"`, `"bf16/auto:coding"`, or `"moonshotai/kimi-k3.1-pro-uncensored"`.

## Links

- Website: https://bf16.ai
- Docs: https://bf16.ai/docs
- Live Models & Pricing: https://bf16.ai/models
- Open-Source `$50` API Grants: https://bf16.ai/open-source
