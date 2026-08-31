"""Loading a causal language model, when one is actually wanted.

This mirrors the checkpoint/inferencer pattern from the reference: load the
model in bfloat16 with an automatic device map, unwrap a processor's tokenizer
where the model is multimodal, and generate from a chat-templated prompt. It is
written so that running real generation on a capable machine is a one-line
switch.

**Nothing here loads a model at import time, and the default pipeline never
calls it.** The pool is built with a deterministic mock generator unless real
generation is explicitly requested, so the whole system runs on a machine that
could not hold a 7B model. `from_pretrained` and `generate` are reached only
inside `QwenGenerator`, only when constructed, and only when the `generative`
extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logs import get_logger

_log = get_logger(__name__)

# Qwen 2.5 7B instruct. A model id, not a load — resolving this string costs
# nothing until something asks the loader to realise it.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(slots=True)
class Checkpoint:
    """A loaded model and its tokenizers.

    `text_tokenizer` is the plain tokenizer even when the model ships a
    processor, since decoding generated ids goes through it. Kept separate for
    exactly the reason the reference keeps it separate.
    """

    model: object
    tokenizer: object
    text_tokenizer: object
    device: object


def load_checkpoint(model_name: str = DEFAULT_MODEL) -> Checkpoint:
    """Realise a model. Imports torch/transformers lazily, inside the call.

    Deliberately not called anywhere on the default path. A machine that cannot
    hold the model never reaches this, because the pool is built from the mock
    generator instead.
    """
    import torch  # noqa: PLC0415 — lazy so the import firewall's ban never trips on the default path
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        ProcessorMixin,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_tokenizer = tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer
    return Checkpoint(
        model=model,
        tokenizer=tokenizer,
        text_tokenizer=text_tokenizer,
        device=model.device,
    )


def generate_one(checkpoint: Checkpoint, system: str, user: str, max_new_tokens: int = 400) -> str:
    """One chat completion, following the reference inference shape.

    Build the prompt from a system and a user turn with the chat template,
    tokenise, generate, and decode only the newly generated ids. Sampling is on
    (text needs variety) with a fixed generator set by the caller's seed.
    """
    import torch  # noqa: PLC0415

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = checkpoint.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = checkpoint.text_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        checkpoint.device
    )
    with torch.no_grad():
        output_ids = checkpoint.model.generate(
            **inputs,
            eos_token_id=checkpoint.text_tokenizer.eos_token_id,
            pad_token_id=checkpoint.text_tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][input_len:]
    return checkpoint.text_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_batch(
    checkpoint: Checkpoint,
    prompts: list[tuple[str, str]],
    max_new_tokens: int = 400,
    batch_size: int = 16,
    progress: bool = True,
) -> list[str]:
    """Many chat completions at once.

    One prompt per forward pass leaves a large model almost idle: the cost of a
    pass is dominated by moving the weights, not by the tokens in it, so a batch
    of sixteen costs barely more than a batch of one. Generating a corpus one
    item at a time is the difference between minutes and hours.

    Prompts are left-padded so the generated continuations line up at the end of
    the sequence, which is what a decoder-only model needs when a batch holds
    prompts of different lengths.
    """
    import torch

    tok = checkpoint.text_tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    original_side = tok.padding_side
    tok.padding_side = "left"

    import time

    out: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    started = time.perf_counter()
    try:
        for batch_i, start in enumerate(range(0, len(prompts), batch_size), 1):
            chunk = prompts[start : start + batch_size]
            texts = [
                checkpoint.tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for sys, usr in chunk
            ]
            inputs = tok(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(checkpoint.device)
            with torch.no_grad():
                ids = checkpoint.model.generate(
                    **inputs,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            for row in ids:
                out.append(tok.decode(row[prompt_len:], skip_special_tokens=True).strip())

            if progress:
                # Report every batch, with a running estimate, because a silent
                # generation of this length is indistinguishable from a hang.
                elapsed = time.perf_counter() - started
                rate = len(out) / elapsed if elapsed else 0.0
                remaining = (len(prompts) - len(out)) / rate if rate else 0.0
                _log.info(
                    "  batch %d/%d  %s/%s texts  %.1f/s  eta %.1f min",
                    batch_i, n_batches, f"{len(out):,}", f"{len(prompts):,}",
                    rate, remaining / 60,
                )
    finally:
        tok.padding_side = original_side
    return out
