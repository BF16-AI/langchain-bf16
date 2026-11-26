import os
from langchain_bf16 import BF16_BASE_URL, BF16_MODELS, ChatBF16


def test_defaults(monkeypatch):
    monkeypatch.setenv("BF16_API_KEY", "sk-bf16-test-123")
    llm = ChatBF16()
    assert llm.model_name == "deepseek/deepseek-v4.1-flash-uncensored"
    assert llm.openai_api_base == BF16_BASE_URL
    assert llm.openai_api_key.get_secret_value() == "sk-bf16-test-123"


def test_custom_model_and_key():
    llm = ChatBF16(model=BF16_MODELS["kimi_k3_1_pro"], api_key="sk-custom")
    assert llm.model_name == "moonshotai/kimi-k3.1-pro-uncensored"
    assert llm.openai_api_key.get_secret_value() == "sk-custom"
