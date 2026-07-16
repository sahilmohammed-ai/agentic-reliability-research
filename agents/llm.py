import anthropic
import ollama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# three backends, routed by model string shape:
#   - "hf:<repo_id>" (e.g. "hf:Qwen/Qwen2.5-3B-Instruct") -> local hf transformers, full
#     precision, same weights verifier/model.py trains on. explicit prefix rather than
#     detecting the "/" in repo ids, so it can't collide with ollama or anthropic names.
#   - anything else containing ":" (e.g. "qwen2.5:3b-instruct") -> ollama, gguf-quantized,
#     served locally through the ollama daemon.
#   - anything else (e.g. "claude-haiku-4-5-20251001") -> anthropic api.
_anthropic_client = None
_hf_cache: dict[str, tuple] = {}  # repo_id -> (model, tokenizer), loaded once per process


def _get_anthropic_client() -> anthropic.Anthropic:
    """lazy init and cache the anthropic client.

    max_retries=10 (SDK default is 2). a real collection run hit repeated transient
    529 OverloadedError responses; confirmed directly (isolated single calls to the same model
    succeeded reliably, a 10-call burst hit one real 529 that survived 5 retries with exponential
    backoff) that this is genuine bursty overload on anthropic's side, not a bug here -- the SDK
    already retries 429/5xx/timeout errors with exponential backoff by default (see
    BaseClient._should_retry), max_retries just wasn't a high enough budget to reliably outlast it."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(max_retries=10)
    return _anthropic_client


def _get_hf_device() -> torch.device:
    """prefer cuda, then apple silicon mps, then cpu. same preference order as verifier/train.py."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_hf_model(repo_id: str):
    """lazy-load and cache an hf model + tokenizer, so repeated calls in one rollout
    collection run don't reload the weights from disk every turn."""
    if repo_id not in _hf_cache:
        device = _get_hf_device()
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16).to(device)
        model.eval()
        _hf_cache[repo_id] = (model, tokenizer, device)
    return _hf_cache[repo_id]


def is_hf_model(model: str) -> bool:
    return model.startswith("hf:")


def is_ollama_model(model: str) -> bool:
    return ":" in model and not is_hf_model(model)


def complete(model: str, system: str, prompt: str, max_tokens: int) -> str:
    """single-turn completion, text only. thin wrapper over complete_with_usage for callers
    that don't care about token counts."""
    text, _ = complete_with_usage(model, system, prompt, max_tokens)
    return text


def complete_with_usage(model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """single-turn completion returning (text, usage), routed to hf, ollama, or anthropic.

    usage is {"prompt_tokens": int, "completion_tokens": int}. all three backends report these
    natively (anthropic via message.usage, ollama via *_eval_count, hf via tensor lengths), so
    the counts are exact per-backend rather than estimated. this is what lets us track cost /
    token usage per rollout turn, the mentor-requested efficiency metric."""
    if is_hf_model(model):
        repo_id = model[len("hf:"):]
        hf_model, tokenizer, device = _get_hf_model(repo_id)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # apply_chat_template handles qwen2.5-instruct's chat format so we don't hand-roll it.
        # tokenize=True + return_dict=True gives a BatchEncoding (input_ids + attention_mask),
        # not a bare tensor, which generate() needs to correctly handle the unpadded single input
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = hf_model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy, matches the deterministic single-choice framing of act()/plan()
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # slice off the prompt tokens, only decode what the model actually generated
        prompt_len = encoded["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = {"prompt_tokens": int(prompt_len), "completion_tokens": int(generated.shape[0])}
        return text, usage

    if is_ollama_model(model):
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"num_predict": max_tokens},
        )
        # ollama returns token counts as prompt_eval_count / eval_count (absent on rare edge cases)
        usage = {
            "prompt_tokens": int(response.get("prompt_eval_count", 0)),
            "completion_tokens": int(response.get("eval_count", 0)),
        }
        return response["message"]["content"].strip(), usage

    message = _get_anthropic_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = {
        "prompt_tokens": int(message.usage.input_tokens),
        "completion_tokens": int(message.usage.output_tokens),
    }
    return message.content[0].text.strip(), usage
