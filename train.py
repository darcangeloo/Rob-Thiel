from datasets import load_dataset 
from trl import SFTTrainer, SFTConfig
from model import peft_model
from peft import LoraConfig

dataset_raw = load_dataset("json", data_files=["datasets\marketing_dataset_batch01-06_alpaca.json", "datasets\startup_batch.jsonl"], split="train")
dataset = dataset_raw.train_test_split(test_size=0.15, seed=42)

def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["instruction"]}],
        "completion": [
            {"role": "assistant", "content": f"{example['output']}"}
        ],
    }
    
dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "output", "category"])
eval_dataset = dataset["test"]

training_args = SFTConfig(
    packing=False,
    assistant_only_loss=True,
    learning_rate=1e-4,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=12, 
    gradient_checkpointing=True,
    num_train_epochs=5,
    eval_strategy="steps",
    output_dir="model",
    eval_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = SFTTrainer(
    peft_model,
    train_dataset=dataset["train"],
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    ),
    args=training_args,
    eval_dataset=eval_dataset,
)

trainer.train()
