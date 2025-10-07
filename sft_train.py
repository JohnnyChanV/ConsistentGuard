import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--datadir", type=str, default="data/sft_data/en")
    parser.add_argument("--output_dir", type=str, default="outputs/SFT-Model-7B")
    parser.add_argument("--run_name", type=str, default="SFT-Model-7B")

    args = parser.parse_args()
    return args

def get_dataset(datadir):
    PROMPT_TEMPLATE = """
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
Assistant: \n"""

    def build_dataset(path):
        data = json.load(open(path, 'r', encoding='utf-8'))

        prompt_list = [PROMPT_TEMPLATE.format(item['input']) for item in data]
        completions = [item['completion'] for item in data]

        dataset = []

        for prompt, completion in zip(prompt_list, completions):
            # 合并input和completion作为text字段
            dataset.append({
                'text':prompt + completion,
            })
        return dataset

    dataset = []

    for dataset_path in os.listdir(datadir):
        if dataset_path.endswith(".json"):
            dataset += (build_dataset(f'{datadir}/{dataset_path}'))

    wrapped_dataset = Dataset.from_list(dataset)
    print(f"Dataset size: {len(wrapped_dataset)}")

    return wrapped_dataset

if __name__ == '__main__':
    args = get_args()
    output_dir = args.output_dir
    device = torch.device(args.device)
    run_name = args.run_name
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
        cache_dir=args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    response_template = "<think>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token
    # tokenizer.padding_side = 'right'  # 确保填充在右侧

    wrapped_dataset = get_dataset(args.datadir)

    training_args = SFTConfig(
        max_seq_length=4096,
        output_dir=output_dir,
        evaluation_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=2000,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        # bf16=True,  # 启用 bf16 精度训练
        model=model,
        args=training_args,
        train_dataset=wrapped_dataset,
        data_collator=collator,
    )

    trainer.train()

    # model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")