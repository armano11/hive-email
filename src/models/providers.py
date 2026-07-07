"""LLM provider abstraction layer.

Supports OpenAI, Anthropic, and Gemini with a unified interface.
Adding a new provider requires implementing Provider.backend().
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    content: str
    model: str
    provider: str
    usage: dict | None = None


class Provider(ABC):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ProviderResponse:
        ...


class OpenAIProvider(Provider):
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ProviderResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return ProviderResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )


class AnthropicProvider(Provider):
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ProviderResponse:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.3),
        )
        return ProviderResponse(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


class GeminiProvider(Provider):
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ProviderResponse:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.3),
                max_output_tokens=kwargs.get("max_tokens", 512),
            ),
        )
        return ProviderResponse(
            content=response.text,
            model=self.model,
            provider="gemini",
            usage={"total_tokens": getattr(response, "usage_metadata", {}).get("total_token_count", 0)},
        )


class NVIDIAProvider(Provider):
    """NVIDIA AI Foundation Models endpoint (OpenAI-compatible API).

    Uses the OpenAI client library with a custom base_url pointing to
    NVIDIA's API endpoint. Supports all models from NVIDIA's catalog.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ProviderResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return ProviderResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="nvidia",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )


PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "nvidia": NVIDIAProvider,
}


def get_provider(provider_name: str, api_key: str, model: str) -> Provider:
    cls = PROVIDER_MAP.get(provider_name)
    if cls is None:
        msg = f"Unknown provider '{provider_name}'. Choose from: {list(PROVIDER_MAP.keys())}"
        raise ValueError(msg)
    logger.info("Using provider=%s model=%s", provider_name, model)
    return cls(api_key=api_key, model=model)
