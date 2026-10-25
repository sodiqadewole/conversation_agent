# Accessing the Local Chat API from Another Machine

The vLLM chat API can be called from another computer using the same
OpenAI-compatible client interface. The connection method determines whether
the server is available only on the host, across a local network, through an
encrypted SSH tunnel, or over a private VPN.

For most development environments, an **SSH tunnel is the recommended
approach** because it encrypts traffic without exposing the vLLM port directly.

## Before You Start

The server machine must:

- Have the local model and vLLM environment configured
- Be running the chat API
- Be reachable from the client machine
- Allow the selected connection through its firewall

The client machine needs either:

- The OpenAI Python SDK
- `curl`
- Another OpenAI-compatible HTTP client

## Connection Options

| Method | Encryption | Exposes API port | Recommended use |
|---|---:|---:|---|
| Local network | No, unless a proxy adds TLS | Yes | Trusted private LAN |
| SSH tunnel | Yes | No | Development and remote administration |
| Private VPN | Depends on VPN; normally yes | Only inside VPN | Persistent remote access |
| Public HTTPS proxy | Yes | Through proxy | Carefully secured deployments |

## Option 1: Local Network Access

By default, the server binds to `127.0.0.1`, which only accepts connections
from the server machine. To accept LAN traffic, bind it to `0.0.0.0`.

### 1. Generate a private API key

The default `local-dev-key` must not be used when exposing the server to the
network.

```bash
export LOCAL_LLM_API_KEY="$(
    python -c 'import secrets; print(secrets.token_urlsafe(32))'
)"

echo "$LOCAL_LLM_API_KEY"
```

Save the generated value securely. Clients need it to authenticate.

### 2. Start the server on all interfaces

Run this command from the repository root on the server machine:

```bash
./myvenv/bin/python LLM/serve_chat_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "$LOCAL_LLM_API_KEY" \
    --max-model-len 2048 \
    --enforce-eager
```

The launcher rejects non-local bind addresses while the development API key is
still in use.

### 3. Find the server's LAN address

On Linux:

```bash
hostname -I
```

An address might look like:

```text
192.168.1.50
```

Prefer the address belonging to the same network as the client machine.

### 4. Configure the firewall

With UFW, allow only the local subnet:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

Replace `192.168.1.0/24` with the correct subnet. Avoid opening port 8000 to
all internet addresses.

Check the firewall:

```bash
sudo ufw status
```

### 5. Test from the client machine

```bash
curl http://192.168.1.50:8000/v1/models \
    -H "Authorization: Bearer YOUR_API_KEY"
```

Replace the address and API key with your values.

### 6. Use the OpenAI Python SDK

Install the SDK on the client machine:

```bash
python -m pip install openai
```

Then connect to the server's LAN address:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.50:8000/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="local-qwen3",
    messages=[
        {
            "role": "user",
            "content": "Hello from another machine.",
        }
    ],
    max_tokens=200,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)

print(response.choices[0].message.content)
```

The `model` value must match a name returned by:

```text
GET /v1/models
```

## Option 2: SSH Tunnel

An SSH tunnel lets the server remain bound to `127.0.0.1`. Port 8000 is not
opened to the LAN or internet, and traffic between the machines is encrypted.

### 1. Keep the API bound to localhost

On the server:

```bash
./myvenv/bin/python LLM/serve_chat_api.py \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 2048 \
    --enforce-eager
```

### 2. Open the tunnel from the client

```bash
ssh -L 8000:127.0.0.1:8000 username@192.168.1.50
```

This maps port 8000 on the client to port 8000 on the server through SSH.
Keep this SSH session open while using the API.

If port 8000 is occupied on the client, choose another local port:

```bash
ssh -L 8080:127.0.0.1:8000 username@192.168.1.50
```

### 3. Connect through the tunnel

For a tunnel using local port 8000:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-dev-key",
)
```

For a tunnel using local port 8080:

```python
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-dev-key",
)
```

Test the tunnel:

```bash
curl http://127.0.0.1:8000/v1/models \
    -H "Authorization: Bearer local-dev-key"
```

### Running the tunnel in the background

Create the tunnel without opening a remote shell:

```bash
ssh -N -L 8000:127.0.0.1:8000 username@192.168.1.50
```

Use your operating system's SSH service or process manager if the tunnel needs
to reconnect automatically.

## Option 3: Private VPN

A private VPN such as Tailscale or WireGuard is useful for persistent access
from outside the local network.

General process:

1. Install the VPN on the server and client.
2. Confirm that both machines can reach each other's VPN addresses.
3. Start the API with a strong API key.
4. Bind to the VPN interface or use a firewall rule that permits only the VPN
   subnet.
5. Set the client's `base_url` to the server's VPN address.

Example:

```python
client = OpenAI(
    base_url="http://100.64.0.10:8000/v1",
    api_key="YOUR_API_KEY",
)
```

VPN addresses and firewall configuration depend on the selected VPN product.

## Public Internet Access

Do not expose the raw vLLM port directly to the public internet.

An API key authenticates requests, but plain HTTP does not encrypt:

- The API key
- User prompts
- Model responses
- Tool arguments and results

For public access, put the API behind an HTTPS reverse proxy such as Caddy or
Nginx and add:

- TLS certificates
- Firewall restrictions
- Strong authentication
- Rate limiting
- Request-size limits
- Token and concurrency limits
- Access logging and monitoring
- Timeouts

A private VPN or SSH tunnel is preferable when access is limited to trusted
users.

## Remote Tool Calling

Tool calls are returned to the client as structured data. The functions execute
where the client application runs, not automatically on the model server.

For example:

1. The remote client sends tools to the chat API.
2. The model returns a structured tool request.
3. The remote client validates and executes the requested function.
4. The remote client sends the tool result back to the API.
5. The model produces its final response.

Never execute arbitrary model-generated code, function names, shell commands,
or unvalidated arguments.

## Troubleshooting

### Connection refused

Confirm that the API is running:

```bash
curl http://127.0.0.1:8000/health
```

For LAN access, confirm that the server is listening on all interfaces:

```bash
ss -ltn | grep 8000
```

Expected LAN binding:

```text
0.0.0.0:8000
```

If it shows `127.0.0.1:8000`, only local connections and SSH tunnels can reach
it.

### Request times out

Check:

- The server's IP address
- Firewall rules on both machines
- Router or VLAN isolation
- Whether both machines are on the same LAN or VPN
- Whether the model is still loading

### Unauthorized response

The client API key must exactly match the key passed to the server:

```python
client = OpenAI(
    base_url="http://SERVER_ADDRESS:8000/v1",
    api_key="THE_SAME_SERVER_API_KEY",
)
```

### Model not found

List the served models:

```bash
curl http://SERVER_ADDRESS:8000/v1/models \
    -H "Authorization: Bearer YOUR_API_KEY"
```

Use the returned model ID in `client.chat.completions.create`.

### SSH tunnel does not connect

Verify ordinary SSH access first:

```bash
ssh username@SERVER_ADDRESS
```

Then retry the tunnel. Ensure the chosen local port is not already occupied.

## Recommended Development Setup

For two trusted machines:

1. Keep the API bound to `127.0.0.1`.
2. Connect using an SSH tunnel.
3. Use the OpenAI SDK against the client's local tunnel port.
4. Keep tool execution and validation in the client application.

This configuration provides encrypted transport without exposing the model
server directly.

