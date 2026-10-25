"""Download a Hugging Face model repository for local vLLM serving."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer.model", "spiece.model")


def model_directory_name(model_id: str) -> str:
    repository_name = model_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", repository_name).strip(".-")
    if not safe_name:
        raise ValueError(f"Cannot derive a directory name from model ID: {model_id!r}")
    return safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a complete Hugging Face model repository for local serving."
        )
    )
    parser.add_argument(
        "model_id",
        help="Hugging Face model ID, for example Qwen/Qwen3-1.7B.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination directory. By default, the repository name is created "
            f"under {DEFAULT_MODELS_DIR}."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Branch, tag, or commit hash to download. Defaults to the repository default.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download files again even when matching cached files are available.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent file downloads (default: 8).",
    )
    return parser.parse_args()


def validate_download(model_dir: Path) -> None:
    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not (model_dir / "tokenizer_config.json").is_file():
        missing.append("tokenizer_config.json")
    if not any((model_dir / name).is_file() for name in TOKENIZER_ASSETS):
        missing.append("a tokenizer asset")

    weight_patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "*.gguf",
    )
    if not any(next(model_dir.glob(pattern), None) for pattern in weight_patterns):
        missing.append("model weights")

    if missing:
        raise RuntimeError(
            f"Downloaded repository is missing {', '.join(missing)}. "
            "Confirm that the repository contains a complete text-generation model."
        )


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive.")

    output_dir = args.output_dir
    if output_dir is None:
        try:
            output_dir = DEFAULT_MODELS_DIR / model_directory_name(args.model_id)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id}")
    print(f"Destination: {output_dir}")

    try:
        snapshot_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=output_dir,
            force_download=args.force_download,
            local_files_only=args.local_files_only,
            max_workers=args.max_workers,
        )
    except HfHubHTTPError as error:
        raise SystemExit(f"Hugging Face download failed: {error}") from error
    except FileNotFoundError as error:
        raise SystemExit(
            "The requested model is not available in the local cache. "
            "Remove --local-files-only or download it while online."
        ) from error

    model_dir = Path(snapshot_path).resolve()
    validate_download(model_dir)

    print("\nDownload complete.")
    print(f"Local model path: {model_dir}")
    print("\nStart the server with:")
    print(
        f"  python {PROJECT_DIR / 'serve_chat_api.py'} "
        f"--model {model_dir} "
        f"--served-model-name {model_directory_name(args.model_id)} "
        "--enforce-eager"
    )


if __name__ == "__main__":
    main()
