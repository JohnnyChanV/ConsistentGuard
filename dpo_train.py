# First, let's import our required libraries
import json

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import DPOConfig,DPOTrainer
from sklearn.model_selection import train_test_split
from R1_components.reward import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="outputs/R1-3B-GRPO-longCoT")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--datadir", type=str, default="dpo_data")
    parser.add_argument("--output_dir", type=str, default="outputs/GRPO-Model-3B-DPO")
    parser.add_argument("--run_name", type=str, default="GRPO-Model-3B-DPO")

    args = parser.parse_args()
    return args


def get_dataset(datadir):
    PROMPT_TEMPLATE = \
"""
A history conversation between User and Assistant, the assistant needs to judge whether the conversation is harmful or unharmful.
The assistant first thinks about the reasoning process in the mind and then makes a judgment.
The reasoning process is enclosed within <think> </think> tags. The judge results are enclosed within <judge>harmful/unharmful</judge> tags.

***JUDGE PRINCIPLES ARE AS FOLLOWS***
- please make sure the conversation violates one of them if you think it is harmful, otherwise, the conversation is unharmful.

# S1: (Violent Crimes, harmful)
# S2: (Non-Violent Crimes, harmful)
# S3: (Sex-Related Crimes, harmful)
# S4: (Child Sexual Exploitation, harmful)
# S5: (Defamation, harmful)
# S6: (Specialized Advice, harmful)
# S7: (Privacy, harmful)
# S8: (Intellectual Property, harmful)
# S9: (Indiscriminate Weapons, harmful)
# S10: (Hate, harmful)
# S11: (Suicide & Self-Harm, harmful)
# S12: (Sexual Content, harmful)
# S13: (Elections, harmful)
# S14: (Code Interpreter Abuse, harmful)

Example:
<conversation>some conversation between user and assistant</conversation>
<think> reasoning process here </think>
<violations>Violation IDs/None</violations>
<judge>harmful</judge>


USER: <conversation>{}</conversation>
Assistant: \n<think>"""

    def build_dataset(path):

        data = json.load(open(path, 'r', encoding='utf-8'))

        return data

    dataset = []

    for dataset_path in os.listdir(datadir):
        if dataset_path.endswith(".json"):
            dataset += (build_dataset(f'{datadir}/{dataset_path}'))

    wrapped_dataset = Dataset.from_list(dataset)

    # print(dataset[0])
    print(f"Dataset size: {len(wrapped_dataset)}")

    return wrapped_dataset


if __name__ == '__main__':
    args = get_args()
    output_dir = args.output_dir
    run_name = args.run_name
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,  # 禁用缓存以启用梯度检查点
        cache_dir=args.cache_dir
    )

    # model_ref = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     use_cache=False,  # 禁用缓存以启用梯度检查点
    #     cache_dir=args.cache_dir
    # )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    wrapped_dataset = get_dataset(args.datadir)

    training_args = DPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type='cosine',
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        save_steps=1000,
        max_grad_norm=0.1,
        log_on_each_node=False,
        beta=0.1,
        report_to=["tensorboard"]  # I'm disabling Wandb.
    )

    # Create your trainer with the wrapped tokenizer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=wrapped_dataset,
        processing_class = tokenizer,
    )

    trainer.train()
    # output_dir = "grpo-model"
    # trainer.save_model(output_dir)
    #
    # # Load the saved model
    # model = AutoModelForCausalLM.from_pretrained(output_dir)
    # model.eval()  # Set to evaluation mode
