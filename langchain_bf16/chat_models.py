import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

BF16_BASE_URL = "https://api.bf16.ai/v1"

BF16_MODELS = {
    "auto": "bf16/auto",
    "auto_fast": "bf16/auto:fast",
    "auto_coding": "bf16/auto:coding",
    "auto_reasoning": "bf16/auto:reasoning",
    "deepseek_v4_1_flash": "deepseek/deepseek-v4.1-flash-uncensored",
    "deepseek_v4_1_flash_fast": "deepseek/deepseek-v4.1-flash-uncensored-fast",
    "glm_5_3_flash": "zai-org/glm-5.3-flash-uncensored",
    "glm_5_3": "zai-org/glm-5.3-uncensored",
    "kimi_k3_1_pro": "moonshotai/kimi-k3.1-pro-uncensored",
}


class ChatBF16(ChatOpenAI):
    """LangChain ChatModel pre-configured for https://api.bf16.ai/v1."""

    model_name: str = Field(
        default="deepseek/deepseek-v4.1-flash-uncensored",
        alias="model",
    )
    openai_api_base: Optional[str] = Field(default=BF16_BASE_URL, alias="base_url")

    def __init__(self, **kwargs: Any) -> None:
        if "api_key" not in kwargs and "openai_api_key" not in kwargs:
            env_key = os.environ.get("BF16_API_KEY") or os.environ.get("OPENAI_API_KEY") or "missing-bf16-api-key"
            kwargs["api_key"] = SecretStr(env_key)
        kwargs.setdefault("base_url", BF16_BASE_URL)
        kwargs.setdefault("model", "deepseek/deepseek-v4.1-flash-uncensored")
        super().__init__(**kwargs)
