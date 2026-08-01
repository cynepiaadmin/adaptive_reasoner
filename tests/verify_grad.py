"""
Fast, conclusive proof that autograd reaches every trainable module of
AdaptiveReasoner -- no hand-written backward needed.

We swap the LLM for a tiny RANDOM GPT2 built from config (instant, offline),
while exercising the EXACT same forward/autograd graph as the real Qwen model.
The point is to verify gradient flow through workspace/controller/reasoner/
latent_attention, which is independent of the backbone.
"""

import torch
import torch.nn.functional as F

import transformers
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config
)


# --- Build a tiny random causal LM locally (no download) -----------------
_TINY_CFG = GPT2Config(
    n_layer=2,
    n_head=4,
    n_embd=64,
    vocab_size=50257   # matches the gpt2 tokenizer we use below
)


def _fake_from_pretrained(cls, name, **kwargs):
    return GPT2LMHeadModel(_TINY_CFG)


transformers.AutoModelForCausalLM.from_pretrained = classmethod(
    _fake_from_pretrained
)

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from adaptive_reasoner import (
    AdaptiveReasoner,
    DEPTHS
)
from datasets.dataset import get_dataloader


def grad_norm(params):
    g = [p.grad for p in params if p.grad is not None]
    if not g:
        return None
    return (sum((x ** 2).sum().item() for x in g)) ** 0.5


def main():
    print("Building AdaptiveReasoner on a tiny random GPT2 (offline)...")
    model = AdaptiveReasoner(model_name="local-tiny-gpt2")
    device = model.llm.device
    model.to(device)
    model.train()

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    loader = get_dataloader(
        os.path.join(_ROOT, "datasets/reasoning_train.json"),
        tok,
        batch_size=2,
        max_len=32,
        limit=4
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    action_target = batch["action_target"].to(device)

    depth_tensor = torch.tensor(DEPTHS, dtype=torch.float, device=device)

    # One training forward (soft Gumbel path).
    out = model(input_ids, attention_mask=attention_mask, training=True)
    loss_answer = F.cross_entropy(
        out["logits"].view(-1, out["logits"].size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    loss_ctrl = F.cross_entropy(out["policy"], action_target)
    loss_compute = out["soft_steps"].mean()
    loss = loss_answer + 0.3 * loss_ctrl + 0.01 * loss_compute

    print(f"loss={loss.item():.4f} "
          f"(ans={loss_answer.item():.4f}, "
          f"ctrl={loss_ctrl.item():.4f}, "
          f"compute={loss_compute.item():.4f})")

    # This single call walks the autograd graph in reverse. If any module
    # were disconnected, its .grad would be None -- so this IS the backward.
    loss.backward()

    gn_workspace = grad_norm([model.workspace.memory])
    gn_controller = grad_norm(model.controller.parameters())
    gn_reasoner = grad_norm(model.reasoner.parameters())
    gn_attn = grad_norm(model.latent_attention.parameters())

    print("\nPer-module gradient norms after loss.backward():")
    print(f"  workspace.memory : {gn_workspace}")
    print(f"  controller       : {gn_controller}")
    print(f"  reasoner         : {gn_reasoner}")
    print(f"  latent_attention : {gn_attn}")

    all_ok = all(
        g is not None and g > 0
        for g in (gn_workspace, gn_controller, gn_reasoner, gn_attn)
    )
    print("\nRESULT:", "ALL MODULES RECEIVED GRADIENTS" if all_ok
          else "FAILURE: some module has no gradient")
    assert all_ok, "backward did not reach all trainable modules"


if __name__ == "__main__":
    main()
