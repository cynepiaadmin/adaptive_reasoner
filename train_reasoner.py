import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from adaptive_reasoner import (
    AdaptiveReasoner,
    MODEL,
    DEPTHS
)

from datasets.dataset import get_dataloader, write_training_data


def trainable_params(model):
    return [
        p for p in [
            model.workspace.memory,
            *model.controller.parameters(),
            *model.reasoner.parameters(),
            *model.latent_attention.parameters(),
        ]
        if p.requires_grad
    ]


def token_accuracy(logits, labels):
    # Token-level accuracy over non-masked (answer) positions.
    pred = logits.argmax(-1)
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (pred[mask] == labels[mask]).float().mean().item()
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="datasets/reasoning_train.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=0.3,
                   help="weight of aux controller (difficulty) loss")
    ap.add_argument("--beta", type=float, default=0.01,
                   help="weight of compute-efficiency regularization")
    ap.add_argument("--tau", type=float, default=1.0,
                   help="Gumbel temperature")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None,
                   help="train on first N examples only (smoke test)")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--generate_data", action="store_true",
                   help="regenerate datasets/reasoning_train.json first")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.generate_data:
        write_training_data(args.train_json, per_difficulty=80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AdaptiveReasoner(tau=args.tau)
    device = model.llm.device
    model.to(device)

    params = trainable_params(model)
    n_params = sum(p.numel() for p in params)
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    loader = get_dataloader(
        args.train_json,
        tokenizer,
        batch_size=args.batch,
        max_len=args.max_len,
        limit=args.limit
    )

    depth_tensor = torch.tensor(DEPTHS, dtype=torch.float, device=device)

    global_step = 0
    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            action_target = batch["action_target"].to(device)

            out = model(
                input_ids,
                attention_mask=attention_mask,
                training=True
            )

            logits = out["logits"]
            policy = out["policy"]
            soft_steps = out["soft_steps"]

            # (1) Answer cross-entropy on answer tokens only.
            loss_answer = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # (2) Auxiliary controller loss: difficulty -> preferred depth.
            loss_ctrl = F.cross_entropy(policy, action_target)

            # (3) Compute-efficiency regularizer.
            loss_compute = soft_steps.mean()

            loss = (
                loss_answer
                + args.alpha * loss_ctrl
                + args.beta * loss_compute
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                acc = token_accuracy(logits, labels)

                # Explicit proof that autograd reached every trainable module.
                def _gn(ps):
                    g = [p.grad for p in ps if p.grad is not None]
                    if not g:
                        return "NO_GRAD"
                    return f"{math.sqrt(sum((x ** 2).sum().item() for x in g)):.4f}"

                gn = {
                    "workspace": _gn([model.workspace.memory]),
                    "controller": _gn(model.controller.parameters()),
                    "reasoner": _gn(model.reasoner.parameters()),
                    "latent_attn": _gn(model.latent_attention.parameters()),
                }
                print(
                    f"[ep{epoch} step{global_step}] "
                    f"loss={loss.item():.3f} "
                    f"ans={loss_answer.item():.3f} "
                    f"ctrl={loss_ctrl.item():.3f} "
                    f"compute={loss_compute.item():.3f} "
                    f"tok_acc={acc:.3f}"
                )
                print(f"  grad_norm={gn}")

            if global_step % args.eval_every == 0:
                # Lightweight controller sanity: average chosen depth per diff.
                with torch.no_grad():
                    pol = policy.softmax(-1)
                    chosen = (pol * depth_tensor).sum(-1).mean().item()
                print(
                    f"  -> avg soft steps chosen: {chosen:.2f} "
                    f"(easy~{DEPTHS[0]}, hard~{DEPTHS[-1]})"
                )

        # End of epoch: save.
        ckpt = os.path.join(args.ckpt_dir, f"adaptive_ep{epoch}.pt")
        model.save_trainables(ckpt)
        print(f"Saved {ckpt}")

    final = os.path.join(args.ckpt_dir, "adaptive_final.pt")
    model.save_trainables(final)
    print(f"Saved final checkpoint {final}")


if __name__ == "__main__":
    main()
