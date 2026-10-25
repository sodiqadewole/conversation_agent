"""Launch a local model behind vLLM's OpenAI-compatible HTTP API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_API_KEY = "local-dev-key"
REQUIRED_MODEL_FILES = ("config.json",)
TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer.model", "spiece.model")
QWEN3_MODEL_TYPE = "qwen3"
QWEN2_5_VL_MODEL_TYPE = "qwen2_5_vl"
QWEN2_5_VL_TOOL_TEMPLATE = PROJECT_DIR / "tool_chat_template_qwen2_5_vl.jinja"
BLOCKED_MEDIA_DOMAIN = "disabled.invalid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local Hugging Face text or multimodal model with vLLM."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--served-model-name",
        default="local-qwen3",
        help="Model name clients send in OpenAI API requests.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=os.getenv("LOCAL_LLM_API_KEY", DEFAULT_API_KEY),
        help="Bearer token required by the API.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context length. Lower this if the GPU runs out of memory.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory vLLM may reserve.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism.",
    )
    parser.add_argument(
        "--tool-call-parser",
        help=(
            "vLLM tool parser matching the model's chat template. "
            "Defaults to qwen3_xml for Qwen3 text models and disabled otherwise."
        ),
    )
    parser.add_argument(
        "--disable-tool-calling",
        action="store_true",
        help="Disable automatic tool calling, including the Qwen3 default.",
    )
    parser.add_argument(
        "--reasoning-parser",
        help=(
            "vLLM reasoning parser matching the model. "
            "Defaults to qwen3 for Qwen3 text models and disabled otherwise."
        ),
    )
    parser.add_argument(
        "--disable-reasoning-parser",
        action="store_true",
        help="Disable reasoning parsing, including the Qwen3 default.",
    )
    parser.add_argument(
        "--max-images-per-prompt",
        type=int,
        default=1,
        help="Maximum images accepted in each multimodal prompt (default: 1).",
    )
    parser.add_argument(
        "--max-videos-per-prompt",
        type=int,
        default=0,
        help="Maximum videos accepted in each multimodal prompt (default: 0).",
    )
    parser.add_argument(
        "--allowed-media-domain",
        action="append",
        default=[],
        metavar="DOMAIN",
        help=(
            "Domain allowed for remote media URLs. Repeat for multiple domains. "
            "Base64 data URLs do not require this option."
        ),
    )
    parser.add_argument(
        "--allow-media-url-redirects",
        action="store_true",
        help="Allow redirects while vLLM fetches remote media URLs.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for faster, more reliable startup at lower throughput.",
    )
    parser.add_argument(
        "--use-flashinfer-sampler",
        action="store_true",
        help="Use FlashInfer sampling, which may require a lengthy first-time CUDA build.",
    )
    return parser.parse_args()


def load_model_config(model_dir: Path) -> dict[str, Any]:
    try:
        with (model_dir / "config.json").open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read model config: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit("Model config.json must contain a JSON object.")
    return config


def validate_args(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    model_dir = args.model.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    missing = [
        name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()
    ]
    if not (model_dir / "tokenizer_config.json").is_file():
        missing.append("tokenizer_config.json")
    if not any((model_dir / name).is_file() for name in TOKENIZER_ASSETS):
        missing.append("a tokenizer asset")
    if missing:
        raise SystemExit(
            f"Model directory is missing {', '.join(missing)}. "
            "Save both the model and tokenizer with save_pretrained()."
        )

    if not list(model_dir.glob("*.safetensors")) and not list(
        model_dir.glob("pytorch_model*.bin")
    ):
        raise SystemExit(f"No model weights found in {model_dir}")

    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be greater than 0 and at most 1.")
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive.")
    if args.tensor_parallel_size <= 0:
        raise SystemExit("--tensor-parallel-size must be positive.")
    if args.max_images_per_prompt < 0:
        raise SystemExit("--max-images-per-prompt cannot be negative.")
    if args.max_videos_per_prompt < 0:
        raise SystemExit("--max-videos-per-prompt cannot be negative.")
    if args.disable_tool_calling and args.tool_call_parser:
        raise SystemExit(
            "--disable-tool-calling cannot be combined with --tool-call-parser."
        )
    if args.disable_reasoning_parser and args.reasoning_parser:
        raise SystemExit(
            "--disable-reasoning-parser cannot be combined with --reasoning-parser."
        )
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        if not args.api_key or args.api_key == DEFAULT_API_KEY:
            raise SystemExit(
                "Set a strong --api-key or LOCAL_LLM_API_KEY before exposing "
                "the server beyond localhost."
            )

    return model_dir, load_model_config(model_dir)


def is_multimodal_model(model_config: dict[str, Any]) -> bool:
    return "vision_config" in model_config or "audio_config" in model_config


def build_command(
    args: argparse.Namespace,
    model_dir: Path,
    model_config: dict[str, Any],
) -> list[str]:
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            "The vllm package was not found. Install it in the active "
            'environment with: python -m pip install "vllm"'
        )

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_dir),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]
    model_type = model_config.get("model_type")
    tool_call_parser = args.tool_call_parser
    if tool_call_parser is None and model_type == QWEN3_MODEL_TYPE:
        tool_call_parser = "qwen3_xml"
    if tool_call_parser is None and model_type == QWEN2_5_VL_MODEL_TYPE:
        tool_call_parser = "hermes"
    if tool_call_parser and not args.disable_tool_calling:
        command.extend(
            ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
        )
        if model_type == QWEN2_5_VL_MODEL_TYPE:
            command.extend(["--chat-template", str(QWEN2_5_VL_TOOL_TEMPLATE)])

    reasoning_parser = args.reasoning_parser
    if reasoning_parser is None and model_type == QWEN3_MODEL_TYPE:
        reasoning_parser = "qwen3"
    if reasoning_parser and not args.disable_reasoning_parser:
        command.extend(["--reasoning-parser", reasoning_parser])

    if is_multimodal_model(model_config):
        modality_limits = {
            "image": args.max_images_per_prompt,
            "video": args.max_videos_per_prompt,
        }
        command.extend(["--limit-mm-per-prompt", json.dumps(modality_limits)])
        allowed_media_domains = args.allowed_media_domain or [BLOCKED_MEDIA_DOMAIN]
        command.append("--allowed-media-domains")
        command.extend(allowed_media_domains)
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def main() -> None:
    args = parse_args()
    model_dir, model_config = validate_args(args)
    command = build_command(args, model_dir, model_config)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", "")]
    )
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.use_flashinfer_sampler else "0"
    )
    environment["VLLM_MEDIA_URL_ALLOW_REDIRECTS"] = (
        "1" if args.allow_media_url_redirects else "0"
    )

    print(f"Serving {model_dir}")
    print(
        "Model input: "
        + ("multimodal" if is_multimodal_model(model_config) else "text")
    )
    print(f"API base URL: http://{args.host}:{args.port}/v1")
    print(f"Client model name: {args.served_model_name}")
    print("Press Ctrl+C to stop the server.")

    try:
        subprocess.run(command, check=True, env=environment)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
