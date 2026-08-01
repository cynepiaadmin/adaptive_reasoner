"""Runtime check of AdaptiveReasoner.forward (training + eval modes)."""
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

_TINY = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=50257)


def _fake(cls, name, **k):
    return GPT2LMHeadModel(_TINY)


transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fake)

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from adaptive_reasoner import AdaptiveReasoner, DEPTHS
from datasets.dataset import get_dataloader

model = AdaptiveReasoner(model_name="local")
device = model.llm.device
model.to(device)
model.eval()   # dropout off -> reasoner deterministic, so manual snaps match forward's

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

loader = get_dataloader(
    os.path.join(_ROOT, "datasets/reasoning_train.json"), tok, batch_size=3, max_len=32, limit=3
)
b = next(iter(loader))
ids = b["input_ids"].to(device)
am = b["attention_mask"].to(device)
V = 50257

# ---- training forward ----
out = model(ids, attention_mask=am, training=True)  # model.eval() -> dropout off
print("TRAIN logits", tuple(out["logits"].shape),
      "latent", tuple(out["latent"].shape),
      "policy", tuple(out["policy"].shape),
      "soft_steps", tuple(out["soft_steps"].shape),
      "action", out["action"].tolist())
assert out["logits"].shape == (3, ids.size(1), V)
assert out["latent"].shape == (3, 16, 64)
assert out["policy"].shape == (3, 4)
assert out["soft_steps"].shape == (3,)
assert torch.isfinite(out["logits"]).all()
assert torch.isfinite(out["latent"]).all()

# ---- eval forward ----
model.eval()
with torch.no_grad():
    oute = model(ids, attention_mask=am, training=False)
print("EVAL action", oute["action"].tolist(), "steps", oute["steps"])
assert oute["latent"].shape == (3, 16, 64)
assert torch.isfinite(oute["logits"]).all()

# ---- consistency: eval latent must equal reasoner applied
#      DEPTHS[action] times to that sample's base latent (original semantics) ----
with torch.no_grad():
    h0 = model.llm(ids, attention_mask=am,
                   output_hidden_states=True).hidden_states[-1].detach()
    base = model.workspace(3) + h0.mean(dim=1, keepdim=True)
    h = base
    snaps = {}
    for d in range(1, model.max_depth + 1):
        h = model.reasoner(h)
        if d in DEPTHS:
            snaps[d] = h
    act = oute["action"]
    for i in range(3):
        manual = snaps[DEPTHS[act[i].item()]][i]
        diff = (manual - oute["latent"][i]).abs().max().item()
        assert diff < 1e-4, f"sample {i} eval latent mismatch {diff}"

# ---- soft-combo sanity: training latent == weighted sum of per-depth snaps
#      using the EXACT same g the forward returned (gumbel is stochastic) ----
with torch.no_grad():
    g = out["g"]                      # the sample forward actually used
    w = g.to(base.dtype).view(4, 3, 1, 1)
    soft = (torch.stack([snaps[d] for d in DEPTHS], 0) * w).sum(0)
    # out is the training forward already computed above
    diff = (soft - out["latent"]).abs().max().item()
    assert diff < 1e-4, f"soft-combo mismatch {diff}"

print("All forward checks passed: shapes OK, finite, "
      "eval==per-depth semantics, soft-combo==training latent")
