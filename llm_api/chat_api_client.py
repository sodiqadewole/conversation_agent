"""Call the local vLLM server with the OpenAI Python SDK."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.completion_usage import CompletionUsage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try the local chat completion API.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LOCAL_LLM_API_KEY", "local-dev-key"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LOCAL_LLM_MODEL", "local-qwen3"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Send one local image to a vision model instead of running text demos.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Question to ask about --image (default: Describe this image.).",
    )
    return parser.parse_args()


def multiply(a: float, b: float) -> float:
    return a * b


TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]
CHAT_TEMPLATE_OPTIONS = {"chat_template_kwargs": {"enable_thinking": False}}


def get_chat_completion(
    client: OpenAI,
    messages: Sequence[ChatCompletionMessageParam],
    *,
    model: str = "local-qwen3",
    max_tokens: int = 500,
    temperature: float = 0.0,
    stop: str | list[str] | None = None,
    tools: Sequence[ChatCompletionToolParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
    seed: int | None = 42,
    extra_body: dict[str, Any] | None = CHAT_TEMPLATE_OPTIONS,
) -> tuple[ChatCompletionMessage, CompletionUsage | None]:
    """Create a non-streaming chat completion and preserve structured tool calls."""
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if extra_body is not None:
        params["extra_body"] = extra_body
    if stop is not None:
        params["stop"] = stop
    if tools is not None:
        params["tools"] = tools
    if tool_choice is not None:
        if tools is None:
            raise ValueError("tool_choice requires at least one tool.")
        params["tool_choice"] = tool_choice

    completion = client.chat.completions.create(**params)
    return completion.choices[0].message, completion.usage


def image_data_url(image_path: Path) -> str:
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Image file does not exist: {resolved_path}")

    media_type, _ = mimetypes.guess_type(resolved_path.name)
    if media_type is None or not media_type.startswith("image/"):
        raise ValueError(
            f"Could not determine a supported image type from: {resolved_path.name}"
        )

    encoded_image = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded_image}"


def run_vision_chat(
    client: OpenAI,
    model: str,
    image_path: Path,
    prompt: str,
) -> None:
    message, usage = get_chat_completion(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                ],
            }
        ],
        model=model,
        temperature=0,
        max_tokens=500,
        extra_body=None,
    )
    print("\nVision chat:")
    print(message.content)
    print(usage)


def run_basic_chat(client: OpenAI, model: str) -> None:
    message, _ = get_chat_completion(
        client,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain transfer learning in one sentence."},
        ],
        model=model,
        temperature=0.2,
        max_tokens=100,
    )
    print("\nBasic chat:")
    print(message.content)


def run_streaming_chat(client: OpenAI, model: str) -> None:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from one to five."}],
        temperature=0,
        max_tokens=100,
        stream=True,
        extra_body=CHAT_TEMPLATE_OPTIONS,
    )
    print("\nStreaming chat:")
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


def run_tool_chat(client: OpenAI, model: str) -> None:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What is 12.5 multiplied by 8?"}
    ]
    assistant_message, _ = get_chat_completion(
        client,
        messages=messages,
        model=model,
        tools=TOOLS,
        # This small checkpoint may answer directly under "auto"; "required"
        # makes the structured tool-call path deterministic for the example.
        tool_choice="required",
        temperature=0,
        max_tokens=200,
    )

    print("\nTool chat:")
    if not assistant_message.tool_calls:
        print(assistant_message.content)
        print("The model answered directly instead of requesting the tool.")
        return

    messages.append(
        cast(
            ChatCompletionMessageParam,
            assistant_message.model_dump(exclude_none=True),
        )
    )
    for tool_call in assistant_message.tool_calls:
        if tool_call.function.name != "multiply":
            raise RuntimeError(f"Model requested unknown tool: {tool_call.function.name}")

        arguments = json.loads(tool_call.function.arguments)
        result = multiply(a=float(arguments["a"]), b=float(arguments["b"]))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"result": result}),
            }
        )

    final_message, _ = get_chat_completion(
        client,
        messages=messages,
        model=model,
        tools=TOOLS,
        tool_choice="none",
        temperature=0,
        max_tokens=200,
    )
    print(final_message.content)


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.image is not None:
        run_vision_chat(client, args.model, args.image, args.prompt)
        return

    run_basic_chat(client, args.model)
    run_streaming_chat(client, args.model)
    run_tool_chat(client, args.model)


if __name__ == "__main__":
    main()
