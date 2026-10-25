from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "llm_api"
    / "tool_chat_template_qwen2_5_vl.jinja"
)


def test_template_preserves_vision_and_tool_calls() -> None:
    template = Environment(autoescape=False).from_string(
        TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    rendered = template.render(
        messages=[
            {"role": "system", "content": "Use tools when useful."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                    {"type": "text", "text": "What time is it?"},
                ],
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "current_date_time",
                    "description": "Get time",
                    "parameters": {"type": "object"},
                },
            }
        ],
        add_generation_prompt=True,
        add_vision_id=False,
    )

    assert "<tools>" in rendered
    assert '"name": "current_date_time"' in rendered
    assert "<|vision_start|><|image_pad|><|vision_end|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")

