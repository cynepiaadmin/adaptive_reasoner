import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from adaptive_reasoner import (
    MODEL,
    DEPTHS
)


# Map difficulty -> target controller action index (which depth to prefer).
DIFFICULTY_TO_ACTION = {
    "easy": 0,    # 1 step
    "medium": 2,  # 4 steps
    "hard": 3,    # 8 steps
}


# ============================================
# Synthetic training-data generator
# ============================================
# The eval set has only 6 hand-written rows; that cannot train anything.
# We generate many math word problems with *computed* answers so the
# supervision is guaranteed correct.

def _gen_easy(n, rng):
    out = []
    for _ in range(n):
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        op = rng.choice(["add", "sub", "mul"])
        if op == "add":
            out.append({
                "difficulty": "easy",
                "prompt": f"What is {a} + {b}?",
                "answer": str(a + b),
            })
        elif op == "sub":
            lo, hi = sorted((a, b))
            out.append({
                "difficulty": "easy",
                "prompt": f"What is {hi} - {lo}?",
                "answer": str(hi - lo),
            })
        else:
            x = rng.randint(1, 12)
            y = rng.randint(1, 12)
            out.append({
                "difficulty": "easy",
                "prompt": f"What is {x} * {y}?",
                "answer": str(x * y),
            })
    return out


def _gen_medium(n, rng):
    out = []
    for _ in range(n):
        kind = rng.choice(["apples", "cost", "score"])
        if kind == "apples":
            x = rng.randint(3, 20)
            y = rng.randint(1, 10)
            z = rng.randint(0, x + y - 1)
            ans = x + y - z
            prompt = (
                f"{'John' if rng.random() < 0.5 else 'Mary'} has {x} apples. "
                f"He buys {y} more. He gives away {z}. "
                f"How many apples remain?"
            )
            out.append({
                "difficulty": "medium",
                "prompt": prompt,
                "answer": str(ans),
            })
        elif kind == "cost":
            p = rng.randint(2, 20)
            q = rng.randint(1, 15)
            ans = p + q
            out.append({
                "difficulty": "medium",
                "prompt": (
                    f"A book costs {p} dollars. A pen costs {q} dollars. "
                    f"What is the total cost?"
                ),
                "answer": str(ans),
            })
        else:
            base = rng.randint(10, 40)
            bonus = rng.randint(1, 10)
            pen = rng.randint(1, base + bonus - 1)
            ans = base + bonus - pen
            out.append({
                "difficulty": "medium",
                "prompt": (
                    f"A student scored {base} points, earned {bonus} bonus "
                    f"points, then lost {pen} points for a mistake. "
                    f"What is the final score?"
                ),
                "answer": str(ans),
            })
    return out


def _gen_hard(n, rng):
    out = []
    for _ in range(n):
        kind = rng.choice(["animals", "train"])
        if kind == "animals":
            heads = rng.randint(10, 40)
            # chickens=2 legs, cows=4 legs. Solve c = 2*heads - legs/2.
            cows = rng.randint(1, heads - 1)
            chickens = heads - cows
            legs = 2 * chickens + 4 * cows
            out.append({
                "difficulty": "hard",
                "prompt": (
                    f"A farmer has chickens and cows. There are {heads} heads "
                    f"and {legs} legs. How many chickens are there?"
                ),
                "answer": str(chickens),
            })
        else:
            s1 = rng.randint(30, 90)
            t1 = rng.randint(1, 4)
            s2 = rng.randint(20, 70)
            t2 = rng.randint(1, 4)
            dist = s1 * t1 + s2 * t2
            out.append({
                "difficulty": "hard",
                "prompt": (
                    f"A train travels {s1} km per hour for {t1} hours and then "
                    f"{s2} km per hour for {t2} hours. What distance did it "
                    f"travel?"
                ),
                "answer": str(dist),
            })
    return out


def generate_training_data(
    per_difficulty=80,
    seed=42
):
    rng = random.Random(seed)
    data = []
    data += _gen_easy(per_difficulty, rng)
    data += _gen_medium(per_difficulty, rng)
    data += _gen_hard(per_difficulty, rng)
    for i, item in enumerate(data, start=1):
        item["id"] = i
    rng.shuffle(data)
    return data


def write_training_data(path, per_difficulty=80, seed=42):
    data = generate_training_data(per_difficulty, seed)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ============================================
# Torch Dataset (teacher-forcing, masked labels)
# ============================================

class ReasoningDataset(Dataset):
    """
    Tokenizes prompt + answer and builds NON-LEAKY masked labels:
      input_ids = prompt_ids + answer_ids + [eos]
      labels    = [-100]*(len(prompt)-1) + answer_ids + [eos] + [-100]
    Position (len(prompt)-1) predicts answer_ids[0] from prompt context only,
    so the model never sees the answer token it is predicting.
    """

    def __init__(
        self,
        data,
        tokenizer,
        max_len=128
    ):
        self.tokenizer = tokenizer
        self.samples = data
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tok = self.tokenizer

        prompt_ids = tok.encode(
            item["prompt"],
            add_special_tokens=False
        )
        ans_ids = tok.encode(
            item["answer"],
            add_special_tokens=False
        )
        eos = tok.eos_token_id

        input_ids = prompt_ids + ans_ids + [eos]
        # Trim to max_len (keep tail = answer so supervision survives).
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]

        labels = (
            [-100] * (len(prompt_ids) - 1)
            + ans_ids
            + [eos]
            + [-100]
        )
        labels = labels[-len(input_ids):]
        # Pad labels to input length if trimming changed lengths.
        if len(labels) < len(input_ids):
            labels = [-100] * (len(input_ids) - len(labels)) + labels

        action_target = DIFFICULTY_TO_ACTION[item["difficulty"]]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficulty": item["difficulty"],
            "action_target": torch.tensor(action_target, dtype=torch.long),
            "answer": item["answer"],
            "prompt": item["prompt"],
        }


def collate(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    targets = [b["action_target"] for b in batch]

    pad_id = 0  # padding with a harmless id; attention_mask handles it.
    max_len = max(t.size(0) for t in input_ids)

    inp = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, t in enumerate(input_ids):
        L = t.size(0)
        inp[i, :L] = t
        attn[i, :L] = 1
        lab[i, :L] = labels[i][:L]

    act = torch.stack(targets, dim=0)

    return {
        "input_ids": inp,
        "labels": lab,
        "attention_mask": attn,
        "action_target": act,
    }


def get_dataloader(
    data_path,
    tokenizer,
    batch_size=4,
    max_len=128,
    limit=None
):
    with open(data_path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    ds = ReasoningDataset(data, tokenizer, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))  # this file lives in datasets/
    out = os.path.join(here, "reasoning_train.json")
    write_training_data(out, per_difficulty=80)
    print(f"Wrote {out}")
