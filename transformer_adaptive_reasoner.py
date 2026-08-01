import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


MODEL="Qwen/Qwen2.5-0.5B"



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


    def forward(self,batch):

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

        self.net=nn.Sequential(

            nn.Linear(
                dim,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                4
            )
        )


    def forward(self,x):

        return self.net(
            x.float()
        )



# ============================================
# Model
# ============================================
class AdaptiveReasoner(nn.Module):

    def __init__(self):

        super().__init__()


        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )


        hidden = (
            self.llm.config.hidden_size
        )


        for p in self.llm.parameters():
            p.requires_grad=False



        dtype=self.llm.dtype


        #
        # Scratchpad
        #

        self.workspace=LatentWorkspace(
            16,
            hidden,
            dtype
        )


        #
        # Adaptive compute
        #

        self.controller=ReasoningController(
            hidden
        )


        #
        # Thinking module
        #

        self.reasoner=nn.TransformerEncoderLayer(
            hidden,
            16,
            batch_first=True
        ).to(dtype)



        #
        # Latent -> token injection
        #

        self.latent_attention=nn.MultiheadAttention(
            hidden,
            16,
            batch_first=True
        ).to(dtype)



        self.depth=[
            1,
            2,
            4,
            8
        ]



    def forward(
        self,
        input_ids,
        attention_mask=None
    ):


        batch=input_ids.size(0)


        #
        # Encode prompt
        #

        out=self.llm(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )


        hidden=out.hidden_states[-1]



        #
        # Create workspace
        #

        latent=self.workspace(
            batch
        )


        latent = latent + hidden.mean(
            dim=1,
            keepdim=True
        )



        #
        # Decide reasoning budget
        #

        state=latent.mean(
            dim=1
        )


        action_logits=self.controller(
            state
        )


        action=action_logits.argmax(
            -1
        )


        steps=[
            self.depth[x]
            for x in action.tolist()
        ]



        #
        # Think
        #

        for i,s in enumerate(steps):

            h=latent[i:i+1]


            for _ in range(s):

                h=self.reasoner(
                    h
                )


            latent[i:i+1]=h



        #
        # Inject reasoning
        #
        reasoned_hidden, _ = self.latent_attention(
            hidden,
            latent,
            latent
        )

        hidden = hidden + reasoned_hidden

        logits = self.llm.lm_head(
            hidden
        )

        return {
            "logits": logits,
            "hidden":hidden,
            "latent":latent,
            "action":action,
            "steps":steps,
            "policy":action_logits
        }

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_tokens=50,
        do_sample=False,
        temperature=0.7
    ):
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


# ============================================
# Test
# ============================================
def main():
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL
    )


    model=AdaptiveReasoner()


    prompt=[
        "A farmer has 20 heads and 56 legs. How many chickens?"
    ]


    encoding=tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    )

    ids= encoding.input_ids.to(
        model.llm.device
    )

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

    out=model.generate(
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


if __name__=="__main__":
    main()
