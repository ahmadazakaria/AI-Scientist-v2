import os
import json
import logging
import time

from .utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
    compile_prompt_to_md,
)
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("ai-scientist")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    # If you’ve exported OLLAMA_BASE_URL, use Ollama locally:
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_url:
        _client = openai.OpenAI(
            api_key="ollama",
            base_url=ollama_url,
            max_retries=0,
        )
        logger.info(f"🔗 Routing OpenAI client to Ollama at {ollama_url}")
    else:
        # Fallback to cloud
        _client = openai.OpenAI(max_retries=0)
        logger.info("🔗 Using official OpenAI API")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    model: str = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query via the OpenAI/Ollama client.
    Ensures both system_message and user_message are strings.
    """
    _setup_openai_client()

    # Ensure system_message and user_message are strings
    if system_message is not None and not isinstance(system_message, str):
        system_message = compile_prompt_to_md(system_message)
    if user_message is not None and not isinstance(user_message, str):
        user_message = compile_prompt_to_md(user_message)

    # Prepare messages for the API
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})

    # Ensure required arguments are passed
    if not messages or not model:
        raise TypeError("Missing required arguments; Expected both 'messages' and 'model'.")

    # Handle optional function-calling
    if func_spec is not None:
        model_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        model_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    # Make the API call
    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        model=model,
        **model_kwargs,
    )
    latency = time.time() - t0

    # Parse the response
    choice = completion.choices[0]
    if func_spec is None:
        output = choice.message.content
    else:
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode function arguments: "
                f"{choice.message.tool_calls[0].function.arguments}"
            )
            raise

    in_tok = completion.usage.prompt_tokens
    out_tok = completion.usage.completion_tokens
    info = {"model": completion.model, "created": completion.created}

    return output, latency, in_tok, out_tok, info
