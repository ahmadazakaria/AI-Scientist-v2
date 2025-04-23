import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage

import anthropic
import backoff
import openai

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
	"GandalfBaum/llama3.1-claude3.7:latest",
"deepseek-r1:latest",
"llama3:latest",  
"mistral:latest"
]

# AVAILABLE_LLMS = [
    
#     "claude-3-5-sonnet-20240620",
#     "claude-3-5-sonnet-20241022",
#     # OpenAI models
#     "gpt-4o-mini",
#     "gpt-4o-mini-2024-07-18",
#     "gpt-4o",
#     "gpt-4o-2024-05-13",
#     "gpt-4o-2024-08-06",
#     "gpt-4.1",
#     "gpt-4.1-2025-04-14",
#     "gpt-4.1-mini",
#     "gpt-4.1-mini-2025-04-14",
#     "o1",
#     "o1-2024-12-17",
#     "o1-preview-2024-09-12",
#     "o1-mini",
#     "o1-mini-2024-09-12",
#     "o3-mini",
#     "o3-mini-2025-01-31",
#     # DeepSeek Models
#     "deepseek-coder-v2-0724",
#     "deepcoder-14b",
#     # Llama 3 models
#     "llama3.1-405b",
#     # Anthropic Claude models via Amazon Bedrock
#     "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
#     "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
#     "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
#     "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
#     "bedrock/anthropic.claude-3-opus-20240229-v1:0",
#     # Anthropic Claude models Vertex AI
#     "vertex_ai/claude-3-opus@20240229",
#     "vertex_ai/claude-3-5-sonnet@20240620",
#     "vertex_ai/claude-3-5-sonnet@20241022",
#     "vertex_ai/claude-3-sonnet@20240229",
#     "vertex_ai/claude-3-haiku@20240307",
#     "ollama:phi3:mini",
#     "ollama:llama3",
# 	"GandalfBaum/llama3.1-claude3.7:latest",
# "deepseek-r1:latest",
# "llama3:latest",  
# "mistral:latest"
# ]


def _ensure_string_content(messages):
    """Convert all message 'content' fields to string if not already."""
    fixed = []
    for m in messages:
        msg = dict(m)
        if isinstance(msg.get("content"), (dict, list)):
            msg["content"] = json.dumps(msg["content"])
        elif msg.get("content") is None:
            msg["content"] = ""
        fixed.append(msg)
    return fixed


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        messages = [{"role": "system", "content": system_message}, *new_msg_history]
        messages = _ensure_string_content(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    if "gpt" in model:
        messages = [{"role": "system", "content": system_message}, *prompt]
        messages = _ensure_string_content(messages)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        messages = [{"role": "user", "content": system_message}, *prompt]
        messages = _ensure_string_content(messages)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            n=1,
            seed=0,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        # response = make_llm_call(client, model, temperature, system_message=system_message, prompt=new_msg_history)
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        messages = [{"role": "system", "content": system_message}, *new_msg_history]
        messages = _ensure_string_content(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "o1" in model or "o3" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        # Fix: ensure all content is string
        messages = [{"role": "user", "content": system_message}, *new_msg_history]
        messages = _ensure_string_content(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepcoder-14b":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            response = client.chat.completions.create(
                model="agentica-org/DeepCoder-14B-Preview",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
        except Exception as e:
            # Fallback to direct API call if OpenAI client doesn't work with HuggingFace
            import requests
            headers = {
                "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": {
                    "system": system_message,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in new_msg_history]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": MAX_NUM_TOKENS,
                    "return_full_text": False
                }
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                content = response.json()["generated_text"]
            else:
                raise ValueError(f"Error from HuggingFace API: {response.text}")

        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found



# --- OLLAMA SUPPORT -----------------------------------------------------
def _create_ollama_client(model_str: str):
    """Return (OpenAI-compatible client, pure_model_name) talking to local Ollama."""
    import openai
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    client = openai.OpenAI(api_key="ollama", base_url=base_url)
    if model_str.startswith("ollama:"):
        return client, model_str.split(":", 1)[1]
    return client, model_str
# ------------------------------------------------------------------------
def create_client(model) -> tuple[Any, str]:
    # Only allow supported models (OpenAI, Ollama, etc.)
    if model.startswith("claude-") or model.startswith("bedrock") or model.startswith("vertex_ai"):
        print(f"ERROR: Anthropic Claude/Bedrock/Vertex AI models are not supported. Please use Ollama models.")
        # Always fallback to preferred local Ollama model
        return _create_ollama_client("ollama:GandalfBaum/llama3.1-claude3.7:latest")
    elif model.startswith("ollama:"):
        return _create_ollama_client(model)
    elif model in [
        "GandalfBaum/llama3.1-claude3.7:latest",
        "llama3:latest",
        "mistral:latest",
        "deepseek-r1:latest",
        "phi3:mini",
    ]:
        # Allow bare model names for local models
        return _create_ollama_client(f"ollama:{model}")
    # Fallback to preferred local Ollama model
    print(f"Model {model} not supported or not available, falling back to ollama:GandalfBaum/llama3.1-claude3.7:latest.")
    return _create_ollama_client("ollama:GandalfBaum/llama3.1-claude3.7:latest")

