from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from llm_api import serve_chat_api


def test_qwen_vl_uses_hermes_parser_and_custom_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serve_chat_api.importlib.util, "find_spec", lambda name: object())
    args = argparse.Namespace(
        served_model_name="local-qwen-vl",
        host="127.0.0.1",
        port=8000,
        api_key="local-dev-key",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        tool_call_parser=None,
        disable_tool_calling=False,
        reasoning_parser=None,
        disable_reasoning_parser=False,
        max_images_per_prompt=1,
        max_videos_per_prompt=0,
        allowed_media_domain=[],
        enforce_eager=False,
    )

    command = serve_chat_api.build_command(
        args,
        Path("/models/Qwen2.5-VL-3B-Instruct"),
        {"model_type": "qwen2_5_vl", "vision_config": {}},
    )

    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--chat-template") + 1] == str(
        serve_chat_api.QWEN2_5_VL_TOOL_TEMPLATE
    )
    assert "--enable-auto-tool-choice" in command
