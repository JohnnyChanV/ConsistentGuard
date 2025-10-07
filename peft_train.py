# First, let's import our required libraries
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

import json
import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig
from trl import GRPOTrainer
# from R1_components.grpo_trainer import GRPOTrainer

from sklearn.model_selection import train_test_split
from R1_components.reward import *
import os

from numpy import nan

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

LABEL_SEMANTIC = {
    0: "Without Explanation",
    1: "With Explanation",
    'nan': "Without Explanation"
}


def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--datadir", type=str, default="data/dev")
    parser.add_argument("--output_dir", type=str, default="outputs/Exp-3B")
    parser.add_argument("--run_name", type=str, default="Exp-3B")

    args = parser.parse_args()
    return args


def get_dataset(datadir):
    PROMPT_TEMPLATE = \
        """
        You are provided with a comment that offers feedback on a piece of writing. Your task is to determine whether the comment includes an explanation.
        
        A comment includes an explanation if, in addition to expressing an opinion or judgment (e.g., "This paragraph is strong"), it justifies that opinion by providing reasons, examples, elaboration, or clarification (e.g., "This paragraph is strong because it clearly outlines the author's argument and uses specific evidence").
        
        **Task:**
        
        1. Identify and extract the judgments or comments made in the text.
        2. For each judgment, determine if it is accompanied by a justification, explanation, or supporting details.
        3. Determine whether the comment has at least one explanation.
        
        ------
        
        # **Analysis Pipeline:**
        
        # Comment:
        # "They did use evidence despite it not greatly emphasizing Louv's writing."
        
        # **Breakdown:**
        
        # - ✅ **Judgment/Observation:** The commenter acknowledges that evidence was used.
        # - ✅ **Qualification:** The commenter mentions a limitation – the evidence did not strongly emphasize Louv’s writing.
        # - ❌ **Lack of Explanation:** The comment does not explain how the evidence was used, what type of evidence was included, or why it failed to effectively support Louv’s writing. It provides an evaluation but lacks reasoning or elaboration.
        
        # **Conclusion:**
        
        # - ➡️ The comment provides a surface-level evaluation but does not include sufficient reasoning or details to qualify as an explanation.
        
        # **Final Decision:**
        # <answer>Without Explanation</answer>
        
        ------
        
        **Instructions:**
        
        Provide your final decision using one of the following labels:
        
        - <answer>With Explanation</answer> – if the comment includes at least one supporting explanation.
        - <answer>Without Explanation</answer> – if the comment is only an opinion, compliment, or criticism without reasoning.
        
        """


    USR_PROMPT = "USER: Here is the commment: {}."


    def build_dataset(path):
        # data = json.load(open(path, 'r', encoding='utf-8'))
        data = pd.read_csv(path)
        data.dropna(subset=['input'], inplace=True)
        data = data.to_dict('records')

        prompt_list = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": PROMPT_TEMPLATE}]
                +
                [{"role": "user", "content": USR_PROMPT.format(item['comment'])}]
                ,
                tokenize=False,
            )
            # PROMPT_TEMPLATE.format(item['comment'])
            for item in data
        ]

        ground_truths = [LABEL_SEMANTIC[int(str(item['label']))] for item in data]

        dataset = []

        for prompt, ground_truth in zip(prompt_list, ground_truths):
            dataset.append({
                'prompt': prompt,
                'ground_truth': ground_truth,
            })
        return dataset


    dataset = []

    for dataset_path in os.listdir(datadir):
        if dataset_path.endswith(".csv"):
            dataset += (build_dataset(f'{datadir}/{dataset_path}'))

    wrapped_dataset = Dataset.from_list(dataset)

    print(dataset[0])
    print(f"Dataset size: {len(wrapped_dataset)}")

    return wrapped_dataset

# ✅ 导入所需的 PEFT 模块
from peft import LoraConfig, get_peft_model

# ... 保持你之前的其他 import 不变 ...

if __name__ == '__main__':
    args = get_args()
    output_dir = args.output_dir
    device = torch.device(args.device)
    run_name = args.run_name

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    wrapped_dataset = get_dataset(args.datadir)

    reward_object = Rewards(tokenizer)

    # ✅ 创建 PEFT 配置（LoRA）
    peft_params = LoraConfig(
        r=8,  # Rank of LoRA matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # 视具体模型而定
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"  # 自回归语言建模
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=1e-5,
        lr_scheduler_type='cosine',
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=1024,
        num_train_epochs=1,
        save_steps=500,
        max_grad_norm=0.1,
        log_on_each_node=False,
        gradient_checkpointing=True,
        generation_kwargs={
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.6,
            "presence_penalty": 1.5,
        },
        temperature=0.6,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        epsilon_high=0.28,
        report_to=["swanlab"]
    )

    # ✅ 在 Trainer 中加入 peft_config
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[
            reward_object.has_think,
            reward_object.has_judge,
            reward_object.think_before_judge,
            reward_object.acc_r,
        ],
        args=training_args,
        train_dataset=wrapped_dataset,
        peft_config=peft_params  # ✅ 这里新增
    )

    trainer.train()

    output_dir = args.output_dir

    trainer.save_model(output_dir)
    #
    # # Load the saved model
    # model = AutoModelForCausalLM.from_pretrained(output_dir)
    # model.eval()  # Set to evaluation mode
