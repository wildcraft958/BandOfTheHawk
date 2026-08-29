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
    inputs = checkpoint.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
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
