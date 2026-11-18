"""speculative_decoding_demo.py

Companion example for: topnotchnote.com/gpu/speculative_decoding.html

Greedy text generation with a target model, with and without a small draft
model proposing tokens (Hugging Face "assisted generation", which is
speculative decoding). Prints tokens per second for both, the number of
target-model forward passes, and whether the outputs are identical.

Models: gpt2-medium (355M parameters) as the target, distilgpt2 (82M) as the
draft. They share a tokenizer, which assisted generation requires. Both are
downloaded from the Hugging Face Hub on first run.

Run:  python3 speculative_decoding_demo.py       (PyTorch + transformers, any GPU)
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET, DRAFT = "gpt2-medium", "distilgpt2"
PROMPTS = [
    "The GPU executes thousands of threads at once, and the memory system",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "In 1969, the Apollo 11 mission",
]
NEW_TOKENS = 128


class CountCalls:
    """Counts forward passes of a model."""
    def __init__(self, model):
        self.n = 0
        model.register_forward_pre_hook(self._hook)

    def _hook(self, *_):
        self.n += 1


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET).to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT).to(dev).eval()
    t_calls, d_calls = CountCalls(target), CountCalls(draft)
    print(f"target {TARGET}, draft {DRAFT}, {NEW_TOKENS} new tokens, greedy, "
          f"{torch.cuda.get_device_name()}\n")

    def generate(ids, assistant):
        kw = dict(max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False,
                  pad_token_id=tok.eos_token_id)
        if assistant is not None:
            kw["assistant_model"] = assistant
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(ids, **kw)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    # warm-up
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.to(dev)
    generate(ids, None)
    generate(ids, draft)

    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        t_calls.n = d_calls.n = 0
        plain, t_plain = generate(ids, None)
        plain_calls = t_calls.n
        t_calls.n = d_calls.n = 0
        spec, t_spec = generate(ids, draft)
        same = torch.equal(plain, spec)
        print(repr(p[:40]) + "...")
        print(f"  plain      {NEW_TOKENS / t_plain:6.1f} tok/s   target passes {plain_calls:4d}")
        print(f"  speculative {NEW_TOKENS / t_spec:6.1f} tok/s   target passes {t_calls.n:4d}"
              f"   draft passes {d_calls.n:4d}   speedup {t_plain / t_spec:4.2f}x"
              f"   identical output: {same}")


if __name__ == "__main__":
    main()
