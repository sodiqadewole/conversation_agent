# Local LLM Chat Completion API

This directory demonstrates how to serve a locally saved Hugging Face text or
vision-language model through an OpenAI-compatible API. The server uses
[vLLM](https://docs.vllm.ai/) for GPU inference, continuous batching, streaming,
and structured tool calls. Existing clients that support the OpenAI Chat
Completions API can connect by changing their base URL and API key.

The included examples support a locally saved Qwen3 text model and
Qwen2.5-VL vision-language model.

To call the server from another machine, see
[Remote access](./REMOTE_ACCESS.md).

## Files

- `serve_chat_api.py`: validates the model directory and launches vLLM's
  OpenAI-compatible server.
- `download_model.py`: downloads a complete Hugging Face model repository into
  the local models directory.
- `chat_api_client.py`: demonstrates regular chat, streaming, and a complete
  tool-call round trip, or sends a local image to a vision model.
- `REMOTE_ACCESS.md`: explains LAN access, SSH tunnels, VPN access, firewall
  configuration, and security considerations for remote clients.
- `finetune_llm_for_function_calling.ipynb`: experiments with loading and
  fine-tuning the language model for function-calling tasks.
- `models/`: local model weights, configuration, tokenizer, and chat template.

## Requirements

- Linux
- Python 3.12
- A CUDA-compatible NVIDIA GPU
- A compatible CUDA/PyTorch installation
- Enough GPU memory for the selected model and KV cache

Install the serving and client dependencies in the project environment:

```bash
source myvenv/bin/activate
python -m pip install vllm openai ninja huggingface_hub
```

Check that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Preparing a Local Model

Download a model by passing its Hugging Face model ID:

```bash
./myvenv/bin/python LLM/download_model.py Qwen/Qwen3-1.7B
```

By default, this creates:

```text
LLM/models/Qwen3-1.7B/
```

Choose a different destination with `--output-dir`:

```bash
./myvenv/bin/python LLM/download_model.py Qwen/Qwen3-1.7B \
    --output-dir LLM/models/my-qwen-model
```

Download a specific branch, tag, or commit:

```bash
./myvenv/bin/python LLM/download_model.py Qwen/Qwen3-1.7B \
    --revision main
```

For gated or private repositories, authenticate with the Hugging Face CLI or
set `HF_TOKEN` before running the command:

```bash
export HF_TOKEN="your-hugging-face-token"
./myvenv/bin/python LLM/download_model.py organization/private-model
```

The script downloads the repository without loading the model onto the CPU or
GPU, verifies the model and tokenizer files, and prints the corresponding
server command when finished.

The model directory must contain both the model and tokenizer assets. A typical
layout is:

```text
models/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

Save a model and tokenizer together with Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
output_directory = "LLM/models"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)
```

For an adapter-based fine-tune, merge the adapter into the base model before
serving it, or configure vLLM's LoRA support separately.

### Preparing a Vision Model

Download the Qwen2.5-VL checkpoint:

```bash
./myvenv/bin/python llm_api/download_model.py \
    Qwen/Qwen2.5-VL-3B-Instruct
```

The downloaded repository includes the image processor and vision
configuration required by vLLM. A text-only checkpoint such as Qwen3 cannot
interpret images even if image bytes are included in its request.

## Starting the Server

Run commands from the repository root.

For the most reliable first startup, disable CUDA graph compilation:

```bash
./myvenv/bin/python LLM/serve_chat_api.py \
    --max-model-len 2048 \
    --enforce-eager
```

The API becomes available at:

```text
http://127.0.0.1:8000/v1
```

Default client configuration:

```text
Model:   local-qwen3
API key: local-dev-key
```

After confirming that serving works, omit `--enforce-eager` to let vLLM use
CUDA graphs and compilation for potentially higher throughput:

```bash
./myvenv/bin/python LLM/serve_chat_api.py
```

The first compiled startup may take several minutes. Later startups can reuse
the generated caches.

### Common Server Options

```text
--model PATH                    Local Hugging Face model directory
--served-model-name NAME        Model name used in API requests
--host HOST                     Bind address; defaults to 127.0.0.1
--port PORT                     HTTP port; defaults to 8000
--api-key KEY                   Bearer token required from clients
--max-model-len TOKENS          Maximum context length
--gpu-memory-utilization FLOAT  Fraction of GPU memory vLLM may reserve
--tensor-parallel-size COUNT    Number of GPUs used for tensor parallelism
--max-images-per-prompt COUNT   Maximum images in one multimodal prompt
--max-videos-per-prompt COUNT   Maximum videos in one multimodal prompt
--allowed-media-domain DOMAIN   Allow a remote media host; repeat as needed
--enforce-eager                 Disable compilation and CUDA graphs
--use-flashinfer-sampler        Enable the optional FlashInfer sampler
```

View every option:

```bash
./myvenv/bin/python LLM/serve_chat_api.py --help
```

## Starting a Vision Model

Start the downloaded Qwen2.5-VL model under the API name `local-qwen-vl`:

```bash
./myvenv/bin/python llm_api/serve_chat_api.py \
    --model llm_api/models/Qwen2.5-VL-3B-Instruct \
    --served-model-name local-qwen-vl \
    --max-model-len 4096 \
    --max-images-per-prompt 1 \
    --enforce-eager
```

The launcher detects the model's `vision_config`, enables multimodal prompt
limits, and does not apply the Qwen3 text model's tool or reasoning parsers.
Video input is disabled by default; opt in with
`--max-videos-per-prompt COUNT`.

Base64 data URLs keep image transfer local. To allow vLLM to fetch remote
images, allow each expected host explicitly:

```bash
./myvenv/bin/python llm_api/serve_chat_api.py \
    --model llm_api/models/Qwen2.5-VL-3B-Instruct \
    --served-model-name local-qwen-vl \
    --allowed-media-domain images.example.com
```

Media URL redirects are disabled by default to prevent bypassing the domain
allowlist. Only use `--allow-media-url-redirects` when redirects are required
and every destination is trusted.

### Sending a Local Image

Use the client with a PNG, JPEG, WebP, or another image format recognized by
Python:

```bash
./myvenv/bin/python llm_api/chat_api_client.py \
    --model local-qwen-vl \
    --image path/to/image.png \
    --prompt "Describe the image and transcribe any visible text."
```

The client validates the local path and MIME type, base64-encodes the image,
and sends an OpenAI-compatible message containing `text` and `image_url`
parts. When `--image` is omitted, the existing text, streaming, and tool-call
examples run as before.

## API Authentication

The default `local-dev-key` is only suitable for local development. Set a
private key before exposing the server beyond localhost:

```bash
export LOCAL_LLM_API_KEY="replace-with-a-long-random-value"

./myvenv/bin/python LLM/serve_chat_api.py \
    --host 0.0.0.0 \
    --api-key "$LOCAL_LLM_API_KEY"
```

The launcher rejects non-local bind addresses when the development key is
still in use. Network exposure should also be protected with a firewall,
TLS-enabled reverse proxy, request limits, and appropriate access controls.

## Checking Server Health

```bash
curl http://127.0.0.1:8000/health
```

List the available models:

```bash
curl http://127.0.0.1:8000/v1/models \
    -H "Authorization: Bearer local-dev-key"
```

## Basic Chat Completion

The client module includes a reusable `get_chat_completion` helper. It accepts
the modern `tools` and `tool_choice` arguments and returns both the structured
assistant message and token-usage metadata:

```python
from openai import OpenAI

from LLM.chat_api_client import get_chat_completion

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-dev-key",
)

message, usage = get_chat_completion(
    client,
    messages=[
        {"role": "user", "content": "Explain transfer learning."},
    ],
    model="local-qwen3",
    max_tokens=500,
)

print(message.content)
print(usage)
```

When tools are supplied, `message.tool_calls` preserves the structured tool
requests returned by the model. The helper intentionally does not expose the
deprecated `functions` parameter.

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-dev-key",
)

response = client.chat.completions.create(
    model="local-qwen3",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is transfer learning?"},
    ],
    temperature=0.2,
    max_tokens=200,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)

print(response.choices[0].message.content)
```

Disabling thinking is useful for short responses from Qwen3. If thinking is
enabled, reasoning may be returned separately from the final message content.

### cURL

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Authorization: Bearer local-dev-key" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-qwen3",
        "messages": [
            {
                "role": "user",
                "content": "Explain transfer learning in one sentence."
            }
        ],
        "temperature": 0.2,
        "max_tokens": 100,
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }'
```

## Streaming

Set `stream=True` and process each response chunk:

```python
stream = client.chat.completions.create(
    model="local-qwen3",
    messages=[{"role": "user", "content": "Count from one to five."}],
    stream=True,
    max_tokens=100,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

vLLM returns streaming responses as server-sent events using the same interface
as the OpenAI SDK.

## Tool and Function Calling

The server enables vLLM's Qwen3 tool-call parser. Tools use the OpenAI function
schema:

```python
tools = [
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

response = client.chat.completions.create(
    model="local-qwen3",
    messages=[
        {
            "role": "user",
            "content": "Use the multiply tool to multiply 12.5 by 8.",
        }
    ],
    tools=tools,
    tool_choice="required",
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)
print(tool_call.function.arguments)
```

The application—not the model server—must validate the function name and
arguments, execute the approved function, and return its result in a message
with `role="tool"`. Never execute arbitrary model-generated code or unvalidated
tool arguments.

Small models may answer directly when `tool_choice="auto"`. Use
`tool_choice="required"` when testing the structured tool path.

## Running the Complete Client Example

With the server running:

```bash
./myvenv/bin/python -m LLM.chat_api_client
```

The client demonstrates:

1. A normal chat completion
2. A streamed chat completion
3. A structured `multiply` tool request
4. Local tool execution
5. A final model response using the tool result

Client settings can be overridden with environment variables:

```bash
export LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_API_KEY="local-dev-key"
export LOCAL_LLM_MODEL="local-qwen3"
```

## Troubleshooting

### CUDA out of memory during startup

Stop other GPU workloads or reduce the context length and vLLM's memory target:

```bash
./myvenv/bin/python LLM/serve_chat_api.py \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager
```

Inspect active GPU processes with:

```bash
nvidia-smi
```

### Startup spends a long time compiling

Use eager mode:

```bash
./myvenv/bin/python LLM/serve_chat_api.py --enforce-eager
```

Compiled mode normally has a larger initial startup cost but may provide better
steady-state throughput.

### FlashInfer waits for a CUDA build

FlashInfer sampling is disabled by default. Only enable it when the CUDA
toolchain and `ninja` are available:

```bash
./myvenv/bin/python LLM/serve_chat_api.py --use-flashinfer-sampler
```

Its first run may compile CUDA extensions and take several minutes.

### Missing tokenizer files

Save the tokenizer into the same directory as the model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.save_pretrained("LLM/models")
```

### Empty content from a reasoning model

Qwen3 may use its token budget for reasoning or return reasoning separately.
Disable thinking through `chat_template_kwargs` for concise chat responses:

```python
extra_body={
    "chat_template_kwargs": {
        "enable_thinking": False,
    }
}
```

### Tool call appears as literal XML

Confirm that the server uses the parser matching the model and template. This
project defaults to:

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
--reasoning-parser qwen3
```

Also disable thinking for short tool requests and use
`tool_choice="required"` when testing a small checkpoint.

## Stopping the Server

When running in a terminal, press `Ctrl+C`. If it runs under another process
manager, stop it through that manager so both the API server and vLLM engine
worker exit cleanly.
