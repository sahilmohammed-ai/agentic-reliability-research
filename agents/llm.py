import anthropic
import ollama

# ollama model tags always contain a colon (e.g. "qwen2.5:3b-instruct"), anthropic model
# names never do, so the colon is enough to route between backends
_anthropic_client = None


def _get_anthropic_client() -> anthropic.Anthropic:
    """lazy init and cache the anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def is_ollama_model(model: str) -> bool:
    return ":" in model


def complete(model: str, system: str, prompt: str, max_tokens: int) -> str:
    """single-turn completion, routed to anthropic or ollama based on the model string."""
    if is_ollama_model(model):
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"num_predict": max_tokens},
        )
        return response["message"]["content"].strip()

    message = _get_anthropic_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
