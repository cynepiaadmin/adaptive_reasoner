import json
import torch

from transformers import AutoTokenizer

from transformer_adaptive_reasoner import (
    AdaptiveReasoner,
    MODEL
)


DATASET = "datasets/reasoning_eval.json"


def load_dataset():

    with open(DATASET) as f:
        return json.load(f)


def normalize(text):
    return (
        text.lower()
        .replace(".", "")
        .replace(",", "")
        .strip()
    )


def check_answer(
    generated,
    expected
):
    return (
        normalize(expected)
        in
        normalize(generated)
    )


def evaluate():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL
    )


    model = AdaptiveReasoner()

    model.eval()


    dataset = load_dataset()

    results = []


    with torch.no_grad():

        for item in dataset:

            print("\n----------------")
            print(
                item["prompt"]
            )

            messages = [
                {
                    "role": "user",
                    "content": item['prompt']
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            attention_mask = inputs.attention_mask.to(
                model.llm.device
            )

            input_ids = inputs.input_ids.to(
                model.llm.device
            )


            #
            # Adaptive reasoning pass
            #

            output = model(
                input_ids
            )


            steps = output["steps"]

            action = output["action"]

            policy = (
                output["policy"]
                .softmax(-1)
            )

            #
            # Baseline Qwen
            #
            baseline_ids = model.llm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False
            )


            baseline_answer = tokenizer.decode(
                baseline_ids[0],
                skip_special_tokens=True
            )

            #
            # Generate adaptive answer
            #
            generated = model.generate(
                input_ids,
                attention_mask,
                max_tokens=50,
                do_sample=False
            )

            new_tokens = generated[0][
                input_ids.shape[1]:
            ]


            adaptive_answer = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            )

            result = {
                "id":
                    item["id"],

                "prompt":
                    item["prompt"],

                "difficulty":
                    item["difficulty"],


                "expected":
                    item["answer"],


                #
                # Adaptive compute
                #

                "action":
                    action.cpu().tolist(),

                "steps":
                    steps,


                "policy":
                    policy.cpu().tolist(),


                #
                # Reasoning memory
                #

                "latent_shape":
                    list(
                        output["latent"].shape
                    ),


                #
                # Baseline
                #
                "baseline_answer":
                    baseline_answer,

                "baseline_correct":
                    check_answer(
                        baseline_answer,
                        item["answer"]
                    ),


                #
                # Generated answer
                #
                "adaptive_answer": adaptive_answer,

                "adaptive_correct":
                    check_answer(
                        adaptive_answer,
                        item["answer"]
                    ),


            }


            results.append(
                result
            )


            print(
                "difficulty:",
                item["difficulty"]
            )

            print(
                "steps:",
                steps
            )


            print(
                "policy:",
                policy.squeeze().tolist()
            )


            print(
                "answer:"
            )

            print(
                adaptive_answer
            )



    #
    # Adaptive compute metrics
    #

    easy=[]
    medium=[]
    hard=[]


    for r in results:

        if r["difficulty"]=="easy":
            easy.extend(
                r["steps"]
            )

        elif r["difficulty"]=="medium":
            medium.extend(
                r["steps"]
            )

        elif r["difficulty"]=="hard":
            hard.extend(
                r["steps"]
            )



    print(
        "\n========== Adaptive Compute =========="
    )


    if easy:
        print(
            "easy:",
            sum(easy)/len(easy)
        )

    if medium:
        print(
            "medium:",
            sum(medium)/len(medium)
        )

    if hard:
        print(
            "hard:",
            sum(hard)/len(hard)
        )


    if easy and hard:

        print(
            "adaptive score:",
            (
                sum(hard)/len(hard)
                -
                sum(easy)/len(easy)
            )
        )


        baseline_correct = sum(
            r["baseline_correct"]
            for r in results
        )


        adaptive_correct = sum(
            r["adaptive_correct"]
            for r in results
        )


        total = len(results)


        print("\n========== Accuracy ==========")


        print(
            "Baseline Qwen:",
            baseline_correct / total
        )


        print(
            "Adaptive Reasoner:",
            adaptive_correct / total
        )

        avg_steps = sum(
            r["steps"][0]
            for r in results
        ) / total


        print(
            "Average reasoning steps:",
            avg_steps
        )

    with open(
        "evaluation_results_v6_1.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    print(
        "\nSaved evaluation_results_v6_1.json"
    )



if __name__=="__main__":
    evaluate()
