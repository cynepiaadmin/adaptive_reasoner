"""End-to-end inference smoke test for AdaptiveReasoner.generate()."""
import torch
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

_TINY = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=50257)


def _fake(cls, name, **k):
    return GPT2LMHeadModel(_TINY)


transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fake)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_reasoner import AdaptiveReasoner

model = AdaptiveReasoner(model_name="local")
device = model.llm.device
model.to(device)
model.eval()

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

prompts = [
    "What is 2 + 3?",
    "A farmer has chickens and cows. There are 20 heads and 56 legs. How many chickens?",
]

for p in prompts:
    enc = tok(p, return_tensors="pt", padding=True)
    ids = enc.input_ids.to(device)
    am = enc.attention_mask.to(device)

    with torch.no_grad():
        # adaptive reasoning forward (eval mode) -- the path generate() uses
        out = model(ids, attention_mask=am, training=False)
        gen = model.generate(ids, attention_mask=am, max_tokens=20)

    new = gen[0][ids.size(1):]
    print(f"PROMPT : {p}")
    print(f"  action={out['action'].tolist()} steps={out['steps']} "
          f"latent={tuple(out['latent'].shape)}")
    print(f"  GOLDEN generate() produced {new.numel()} new tokens, "
          f"finite logits={torch.isfinite(out['logits']).all().item()}")
    print(f"  decoded: {tok.decode(new, skip_special_tokens=True)!r}")
    print()

print("generate() inference path OK")
