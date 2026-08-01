# Adaptive Reasoner

A trainable **adaptive-compute transformer** built on top of a frozen
causal-LM backbone (default: `Qwen/Qwen2.5-0.5B`). Instead of always
running the LLM the same way, a small controller learns to spend *more or
fewer* reasoning steps on each prompt, conditioned on difficulty. Only the
small reasoning modules are trained; the backbone stays frozen and is used
as a fixed feature extractor.

This repository is the **training + evaluation** implementation. The model
itself lives in `adaptive_reasoner.py`; the training loop,
dataset generator, and evaluation harness are separate modules.

---

## How it works

```
            prompt tokens
                 │
                 ▼
        ┌─────────────────┐
        │  FROZEN LLM      │  (Qwen2.5-0.5B, requires_grad=False)
        │  hidden states   │
        └────────┬────────┘
                 │ detach()  ── gradients do NOT flow into the backbone
                 ▼
        ┌─────────────────┐
        │ LatentWorkspace  │  learnable (slots × dim) "scratchpad",
        │  + prompt mean   │  biased by the prompt representation
        └────────┬────────┘
                 ▼
        ┌─────────────────┐        ┌──────────────────────┐
        │ ReasoningController      → │ action logits (4)    │
        │  (prompt → budget) │        │ depths = [1,2,4,8]  │
        └─────────────────┘        └──────────┬───────────┘
                 │                              │
                 │            training: Gumbel-softmax (soft, tau=1.0)
                 │                        → weighted sum of per-depth latents
                 │            eval    : argmax → fixed depth
                 ▼                              ▼
        ┌─────────────────┐        ┌──────────────────────┐
        │  Reasoner        │  (1× TransformerEncoderLayer,
        │  ×DEPTHS[action] │   applied 1..max_depth times,
        │                 │   snapshot at each configured depth)
        └────────┬────────┘        └──────────┬───────────┘
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
                  ┌──────────────────────┐
                  │  LatentAttention      │  MultiheadAttention:
                  │  hidden + attn(h, L, L)│  injects latent into every position
                  └──────────┬───────────┘
                            ▼
                  lm_head(hidden) → logits
```

* **Adaptive compute.** `ReasoningController` maps the prompt's latent mean
  to 4 logits, one per candidate depth in `DEPTHS = [1, 2, 4, 8]`. The reasoner
  (a single `TransformerEncoderLayer`) is run up to `max_depth = 8` times and
  snapshots are taken at each configured depth.
* **Training (differentiable).** Depth selection uses a soft
  `F.gumbel_softmax(..., hard=False)` over the 4 actions, so the controller
  receives a gradient from the answer loss *despite* the discrete budget.
  The training latent is the Gumbel-weighted combination of the per-depth
  snapshots.
* **Inference (discrete).** `argmax` picks one depth; the reasoner is applied
  exactly that many times (original per-sample semantics, verified equal to
  the soft combination in `tests/check_forward.py`).

Only four modules are trainable (~10.3M params): `workspace.memory`,
`controller`, `reasoner`, `latent_attention`. Backprop uses plain PyTorch
autograd — there is **no hand-written backward** (see
`tests/verify_grad.py`).

---

## Install

```bash
pip install torch transformers accelerate
```

* `torch` — tested with 2.7.x (CUDA or CPU).
* `transformers` — 4.51.x.
* `accelerate` — required by the model's `device_map="auto"` loader.
* The default backbone (`Qwen/Qwen2.5-0.5B`, ~1 GB) is downloaded from the
  HuggingFace Hub on first run. Set `HF_HOME` / `HF_HUB_OFFLINE` as needed.

> The `datasets/` directory in this repo is **our own code + data**, not the
> third-party `datasets` PyPI package. An `__init__.py` marker makes it a real
> package so the local `datasets` shadows any installed library on `sys.path`.

---

## Project structure

```
adaptive_reasoner/
├── adaptive_reasoner.py   # core: AdaptiveReasoner + LatentWorkspace,
│                                      #       ReasoningController, forward, generate,
│                                      #       save/load_trainables
├── train_reasoner.py                  # training loop (3-term loss) + checkpoints
├── evaluate_reasoner.py               # eval: adaptive vs baseline Qwen + metrics
├── README.md
├── datasets/
│   ├── __init__.py
│   ├── dataset.py                     # synthetic generator + ReasoningDataset + DataLoader
│   ├── reasoning_train.json           # 240 generated training rows
│   └── reasoning_eval.json            # 6 hand-written eval rows
├── tests/
│   ├── __init__.py
│   ├── check_forward.py               # forward shape + train/eval consistency
│   ├── check_generate.py              # end-to-end generate() smoke
│   └── verify_grad.py                 # autograd gradient-flow proof
└── checkpoints/                       # created at train time
```

---

## Quick start

### 1. Generate training data (optional)

`datasets/reasoning_train.json` already exists (240 rows: 80 easy / 80 medium
/ 80 hard, with *computed* correct answers). Regenerate with either:

```bash
python3 datasets/dataset.py                 # writes datasets/reasoning_train.json
# or
python3 train_reasoner.py --generate_data   # also starts training after
```

### 2. Train

```bash
python3 train_reasoner.py --epochs 3 --batch 8
```

Key flags (defaults in parentheses):

| flag             | default                        | meaning |
|------------------|--------------------------------|---------|
| `--train_json`   | `datasets/reasoning_train.json`| training data |
| `--epochs`       | `3`                            | epochs |
| `--batch`        | `4`                            | batch size |
| `--lr`           | `2e-4`                         | AdamW learning rate |
| `--alpha`        | `0.3`                          | weight of aux controller (difficulty→depth) loss |
| `--beta`         | `0.01`                         | weight of compute-efficiency regularizer |
| `--tau`          | `1.0`                          | Gumbel temperature |
| `--grad_clip`    | `1.0`                          | gradient norm clip |
| `--max_len`      | `128`                          | tokenization length |
| `--limit`        | `None`                         | train on first N rows only (smoke test) |
| `--ckpt_dir`     | `checkpoints`                  | checkpoint directory |
| `--eval_every`   | `50`                           | steps between controller sanity prints |
| `--generate_data`| off                            | regenerate training data first |

**Loss** = `answer_CE  +  alpha · controller_aux_CE  +  beta · compute_efficiency`
where `answer_CE` is cross-entropy on answer tokens only (prompt is masked out
of the labels so the model never sees the answer it predicts), `controller_aux_CE`
supervises difficulty→preferred-depth, and `compute_efficiency` penalizes wasted
steps on easy items.

Checkpoints (`adaptive_ep{epoch}.pt`, `adaptive_final.pt`) save only the four
trainable modules via `model.save_trainables(...)`.

> Training a 0.5B backbone on CPU is very slow (~1 step/min). Run on a GPU
> machine. Use `--limit 32 --epochs 1` for a quick smoke test.

### 3. Evaluate

```bash
python3 evaluate_reasoner.py --checkpoint checkpoints/adaptive_final.pt
```

Compares the adaptive reasoner against the frozen baseline Qwen on
`datasets/reasoning_eval.json` and prints, per difficulty, the average chosen
reasoning depth plus an *adaptive score* (`avg(hard steps) − avg(easy steps)` —
larger is better adaptivity) and accuracy for both systems. Writes
`evaluation_results_v6_1.json`.

After successful training you should see the controller pick **fewer** steps
for easy prompts and **more** for hard ones (the opposite of the random
init, which is what `tests/check_generate.py` shows before training).

---

## Verification (offline, no GPU, fast)

`tests/` uses a tiny **random** GPT-2 (built from config, no download) by
monkeypatching `AutoModelForCausalLM.from_pretrained`. This exercises the
*exact same forward / autograd graph* as the real Qwen model, so it proves
code correctness (shapes, gradient flow, inference path) in seconds — it does
**not** measure real language quality (the tiny model's text output is
meaningless by design).

```bash
python3 tests/verify_grad.py     # proves all 4 trainable modules get gradients
python3 tests/check_forward.py   # forward shapes + train/eval consistency
python3 tests/check_generate.py  # end-to-end generate() path
```

All three print a PASS / OK line on success.

---

## API reference

```python
from adaptive_reasoner import AdaptiveReasoner, DEPTHS

# Build (downloads Qwen2.5-0.5B on first use; backbone frozen)
model = AdaptiveReasoner(tau=1.0)          # or AdaptiveReasoner(model_name="...")

# Training forward — soft (Gumbel) depth selection, returns grad-carrying dict
out = model(input_ids, attention_mask=am, training=True)
# out: logits, hidden, latent, action, steps, policy (controller logits),
#      g (gumbel probs), soft_steps (continuous step count)

# Inference forward — discrete argmax depth
out = model(input_ids, attention_mask=am, training=False)

# Generate (budget chosen once from the prompt, then held fixed)
ids = model.generate(input_ids, attention_mask=am, max_tokens=50)

# Checkpoint the trainable modules only
model.save_trainables("checkpoints/adaptive_final.pt")
model.load_trainables("checkpoints/adaptive_final.pt")
```

---

## Design notes & limitations

* **Frozen backbone.** LLM activations are `detach()`ed; gradients flow only
  through the four small modules. The latent is injected as a global additive
  residual at every token position via `MultiheadAttention`.
* **Latent is prompt-driven, not per-token recurrent.** The reasoner refines a
  single workspace latent conditioned on the prompt; it is not re-fed token by
  token during generation (the architecture re-encodes each step but the action
  is fixed). Making the latent recurrent per decode step is future work.
* **Gumbel temperature.** `tau` controls training-time stochasticity of the
  soft depth selector; set `tau→0` to approach the discrete policy.
* **`device_map="auto"`** requires `accelerate`; on a single device it simply
  maps to that device. The model loads in `float16`.

## Future work

* Per-token recurrent latent (true chain-of-thought in latent space).
* Policy-gradient (REINFORCE / PPO) on the discrete action, bootstrapped from
  the aux controller loss.
* Larger / harder eval sets; curriculum on difficulty.
