#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Async LLM client for anthropomorphic safety evaluation pipelines."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by OpenAI and Anthropic API clients."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 300.0
    max_retries: int = 20
    retry_delay: float = 1.0
    max_concurrent: int = 1
    temperature: float = 0.7
    max_tokens: int | None = 4096
    use_json_response_format: bool = False


def extract_json_object(text: str) -> str:
    """Return the first balanced JSON object string from a model response."""

    if not isinstance(text, str):
        raise ValueError(f"Expected string output, got {type(text).__name__}")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped, count=1, flags=re.I).strip()
    if not stripped:
        raise ValueError("Empty model output")

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]

    raise ValueError("No balanced JSON object found")


def loads_json_object(text: str) -> dict[str, Any]:
    """Parse model output into a JSON object dictionary."""

    obj_text = extract_json_object(text)
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


class LLMClient:
    """Async client for OpenAI Chat Completions and Anthropic Messages APIs."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        self.provider = "anthropic" if "claude" in config.model.lower() else "openai"
        if self.provider == "anthropic":
            base_url = config.base_url.rstrip("/") if config.base_url else None
            if base_url and base_url.endswith("/v1"):
                base_url = base_url[:-3]
            self._client = AsyncAnthropic(api_key=config.api_key or "", base_url=base_url, timeout=config.timeout)
        else:
            self._client = AsyncOpenAI(api_key=config.api_key or "", base_url=config.base_url, timeout=config.timeout)

    async def chat(
        self,
        user_prompt: str = "",
        *,
        system_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generate a response from a current user prompt and optional message history."""

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return await self._request_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except OverflowError:
                raise
            except Exception as exc:
                last_error = exc
                await self._sleep_before_retry(attempt)
        raise RuntimeError(
            f"Model call failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    async def chat_json(
        self,
        user_prompt: str = "",
        *,
        system_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse one JSON response from a current user prompt and optional message history."""

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                text = await self._request_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True,
                )
                return loads_json_object(text)
            except OverflowError:
                raise
            except Exception as exc:
                last_error = exc
                await self._sleep_before_retry(attempt)
        raise RuntimeError(
            f"JSON model call failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    async def _request_once(
        self,
        system_prompt: str | None,
        user_prompt: str,
        messages: list[dict[str, str]] | None,
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> str:
        request_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        request_temperature = self.config.temperature if temperature is None else temperature
        request_messages = [message.copy() for message in messages] if messages is not None else []
        if user_prompt or messages is None:
            request_messages.append({"role": "user", "content": user_prompt})
        async with self._semaphore:
            if self.provider == "anthropic":
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": request_messages,
                    "max_tokens": 4096 if request_max_tokens is None else request_max_tokens,
                }
                # Bedrock-backed Claude models reject the deprecated temperature parameter.
                if system_prompt:
                    kwargs["system"] = system_prompt
                response = await self._client.messages.create(**kwargs)
                if response.stop_reason == "max_tokens":
                    raise OverflowError(f"Model response exceeded max_tokens={kwargs['max_tokens']}")
                return "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                ).strip()

            api_messages: list[dict[str, str]] = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            api_messages.extend(request_messages)
            kwargs = {
                "model": self.config.model,
                "messages": api_messages,
                "temperature": request_temperature,
            }
            if request_max_tokens is not None:
                # GPT-5+ and o-series reasoning models reject max_tokens; the official parameter is max_completion_tokens.
                token_key = "max_completion_tokens" if re.match(r"^(gpt-[5-9]|o[1-9])", self.config.model) else "max_tokens"
                kwargs[token_key] = request_max_tokens
            if json_mode and self.config.use_json_response_format:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Some models only support the default temperature (1); retry once without it.
                if "temperature" in str(exc) and "does not support" in str(exc):
                    kwargs.pop("temperature", None)
                    response = await self._client.chat.completions.create(**kwargs)
                else:
                    raise
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise OverflowError(f"Model response exceeded max_tokens={request_max_tokens}")
            content = choice.message.content
            return (content or "").strip()

    async def _sleep_before_retry(self, attempt: int) -> None:
        if attempt >= self.config.max_retries - 1:
            return
        delay = self.config.retry_delay * (attempt + 1)
        if delay > 0:
            await asyncio.sleep(delay + random.uniform(0, min(delay, 1.0)))


async def test_llm_client(args: argparse.Namespace) -> None:
    config = ModelConfig(
        model=args.generate_model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
    )

    print("test API ")
    print(f"model: {args.generate_model}")
    print(f"base_url: {args.base_url}")

    try:
        llm = LLMClient(config)
        print(f"provider: {llm.provider}")
        text = await llm.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt="Who are you?",
        )
        print(f"text response: {text}")

        obj = await llm.chat_json(
            system_prompt="You only output JSON objects, no markdown.",
            user_prompt='Please output only a JSON object containing fields "status" and "message", where "status" has the value "ok".',
        )
        print(f"json response: {obj}")

        if args.num_requests > 1:
            print(f"concurrent test: num_requests={args.num_requests}, max_concurrent={args.max_concurrent}")
            test_prompts = [
                "Answer in one short sentence: who are you?",
                "Output exactly three words describing a stable API client.",
                "Give one Python list comprehension example.",
                "Explain concurrent requests in one short sentence.",
                "Write a short safety evaluation task description.",
                "Explain JSON mode in plain text.",
                "Give one smoke test checkpoint phrase.",
                "Explain max_tokens in one short sentence.",
            ]
            responses = await asyncio.gather(
                *[
                    llm.chat(
                        system_prompt="You are a helpful assistant.",
                        user_prompt=f"[request-{index}] {test_prompts[index % len(test_prompts)]}"
                    )
                    for index in range(args.num_requests)
                ]
            )
            for index, response in enumerate(responses):
                print(f"concurrent input {index}: {test_prompts[index % len(test_prompts)]}")
                print(f"concurrent response {index}: {response}")
    except Exception as exc:
        print(f"API test failed: {exc}")
        return

    print("API test completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible or Anthropic LLM client")
    parser.add_argument("--generate_model", type=str, default="local-model", help="Model name; names containing claude use Anthropic")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="API Base URL")
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument("--max_concurrent", type=int, default=1, help="max concurrent requests for testing")
    parser.add_argument("--num_requests", type=int, default=1, help="number of requests to send for concurrent testing")

    args = parser.parse_args()
    asyncio.run(test_llm_client(args))
