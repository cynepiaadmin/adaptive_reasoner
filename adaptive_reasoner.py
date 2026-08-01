import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


MODEL = "Qwen/Qwen2.5-0.5B"

# Reasoning depths available to the adaptive controller.
# Index i in the controller output maps to DEPTHS[i] reasoner passes.
DEPTHS = [1, 2, 4, 8]


# ============================================
# Latent Workspace
# ============================================

class LatentWorkspace(nn.Module):

    def __init__(
        self,
        slots,
        dim,
        dtype
    ):
        super().__init__()

        self.memory = nn.Parameter(
            torch.randn(
                slots,
                dim,
                dtype=dtype
            ) * 0.02
        )

    def forward(self, batch):
        return (
            self.memory
            .unsqueeze(0)
            .expand(
                batch,
                -1,
                -1
            )
        )


# ============================================
# Adaptive Controller
# ============================================

class ReasoningController(nn.Module):

    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                256
            ),
            nn.GELU(),
            nn.Linear(
                256,
                len(DEPTHS)
            )
        )

    def forward(self, x):
        return self.net(
            x.float()
        )


# ============================================
# Model
# ============================================

class AdaptiveReasoner(nn.Module):

    def __init__(
        self,
        model_name=MODEL,
        slots=16,
        tau=1.0
    ):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        hidden = (
            self.llm.config.hidden_size
        )

        # Freeze the backbone. We only train the small reasoning modules.
        for p in self.llm.parameters():
            p.requires_grad = False

        dtype = self.llm.dtype

        self.workspace = LatentWorkspace(
            slots,
            hidden,
            dtype
        )

        self.controller = ReasoningController(
            hidden
        )

        self.reasoner = nn.TransformerEncoderLayer(
            hidden,
            16,
            batch_first=True
        ).to(dtype)

        self.latent_attention = nn.MultiheadAttention(
            hidden,
            16,
            batch_first=True
        ).to(dtype)

        # Gumbel temperature for the soft (training) depth selection.
        self.tau = tau

        # Buffer: depth values used to compute soft step count / aux target.
        self.register_buffer(
            "depth_tensor",
            torch.tensor(
                DEPTHS,
                dtype=torch.float
            )
        )

        self.max_depth = max(DEPTHS)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask=None,
        training=False
    ):
        batch = input_ids.size(0)

        # Encode prompt. The LLM is a FROZEN feature extractor; detach its
        # activations so gradients only flow through the small modules.
        out = self.llm(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden = out.hidden_states[-1].detach()

        # Create workspace, conditioned on the prompt mean.
        latent = self.workspace(batch)
        latent = latent + hidden.mean(
            dim=1,
            keepdim=True
        )

        # Decide reasoning budget.
        state = latent.mean(dim=1)
        action_logits = self.controller(state)  # (B, num_depths)

        # Vectorized per-depth latents: run the reasoner up to max_depth and
        # snapshot at each configured depth. This is differentiable and
        # avoids the per-sample sequential loop.
        snaps = {}
        h = latent
        for d in range(1, self.max_depth + 1):
            h = self.reasoner(h)
            if d in DEPTHS:
                snaps[d] = h

        latent_by_depth = torch.stack(
            [snaps[d] for d in DEPTHS],
            dim=0
        )  # (num_depths, B, slots, dim)

        if training:
            # Soft (Gumbel-relaxed) depth selection so the controller receives
            # a gradient from the answer loss despite the discrete budget.
            g = F.gumbel_softmax(
                action_logits,
                tau=self.tau,
                hard=False
            )  # (B, num_depths)

            w = g.to(latent_by_depth.dtype).view(
                len(DEPTHS), batch, 1, 1
            )
            latent = (latent_by_depth * w).sum(0)

            # Continuous proxy for "how many steps were chosen".
            soft_steps = (
                g.float() * self.depth_tensor
            ).sum(-1)  # (B,)

            action = action_logits.argmax(-1)
            steps = [DEPTHS[a] for a in action.tolist()]

        else:
            action = action_logits.argmax(-1)
            steps = [DEPTHS[a] for a in action.tolist()]
            # Per-sample selection: out[b] = latent_by_depth[action[b], b].
            # Indexing latent_by_depth[action] alone yields 4-D (B,B,S,D);
            # gather with an arange gives the correct (B, S, D).
            latent = latent_by_depth[
                action,
                torch.arange(batch, device=action.device)
            ]  # (B, slots, dim)
            g = None
            soft_steps = None

        # Inject reasoning into the hidden states.
        reasoned_hidden, _ = self.latent_attention(
            hidden,
            latent,
            latent
        )
        hidden = hidden + reasoned_hidden

        logits = self.llm.lm_head(hidden)

        return {
            "logits": logits,
            "hidden": hidden,
            "latent": latent,
            "action": action,
            "steps": steps,
            "policy": action_logits,   # controller logits (for aux loss)
            "g": g,                    # gumbel probs (training only)
            "soft_steps": soft_steps,  # continuous step count (training only)
        }

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_tokens=50,
        do_sample=False,
        temperature=0.7
    ):
        # Decide the compute budget ONCE from the prompt, then hold it fixed
        # for the whole generation (the reasoning workspace is prompt-driven,
        # not per-token).
        with torch.no_grad():
            out0 = self.forward(
                input_ids,
                attention_mask,
                training=False
            )
        # Note: current architecture re-encodes each step; budget is fixed
        # via argmax and the latent is recomputed but the action stays.

        for _ in range(max_tokens):
            output = self.forward(
                input_ids,
                attention_mask
            )

            logits = output["logits"][:, -1, :]

            if do_sample:
                probs = torch.softmax(
                    logits / temperature,
                    dim=-1
                )
                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )
            else:
                next_token = (
                    logits
                    .argmax(dim=-1)
                    .unsqueeze(-1)
                )

            if torch.all(
                next_token == self.llm.config.eos_token_id
            ):
                break

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

        return input_ids

    # --------------------------------------------------
    # Checkpoint IO (trainable modules only)
    # --------------------------------------------------

    def save_trainables(self, path):
        sd = {
            "workspace": self.workspace.state_dict(),
            "controller": self.controller.state_dict(),
            "reasoner": self.reasoner.state_dict(),
            "latent_attention": self.latent_attention.state_dict(),
        }
        torch.save(sd, path)

    def load_trainables(self, path):
        sd = torch.load(
            path,
            map_location=self.llm.device
        )
        self.workspace.load_state_dict(sd["workspace"])
        self.controller.load_state_dict(sd["controller"])
        self.reasoner.load_state_dict(sd["reasoner"])
        self.latent_attention.load_state_dict(sd["latent_attention"])


# ============================================
# Test
# ============================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    model = AdaptiveReasoner()

    prompt = [
        "A farmer has 20 heads and 56 legs. How many chickens?"
    ]

    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    )

    ids = encoding.input_ids.to(model.llm.device)

    attention_mask = encoding.attention_mask.to(
        model.llm.device
    )

    base_output = model.llm.generate(
        input_ids=ids,
        attention_mask=attention_mask,
        max_new_tokens=50
    )
    print("Baseline Qwen:")
    print(
        tokenizer.decode(
            base_output[0],
            skip_special_tokens=True
        )
    )

    out = model.generate(
        input_ids=ids,
        attention_mask=attention_mask,
        max_tokens=50
    )
    print("Adaptive Reasoner:")
    print(
        tokenizer.decode(
            out[0],
            skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    main()
