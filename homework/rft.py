from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset
from .data import BenchmarkResult, Dataset, benchmark

from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
import json
import torch


def load() -> BaseLLM:
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


# def format_example(preferred: str, rejected: str) -> dict[str, str]:
#     """
#     Format a preference pair into a binary classification / ranking input.

#     You can optionally add "better answer:" style instructions.
#     """
#     return {
#         "question": f"Which is better? A: {preferred.strip()} B: {rejected.strip()}",
#         "answer": "<answer>A</answer>",
#     }


# class RFTDataset(TorchDataset):
#     def __init__(self, tokenizer, path: str, format_fn):
#         with open(path) as f:
#             self.data = json.load(f)
#         self.tokenizer = tokenizer
#         self.format_fn = format_fn

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         preferred, rejected = self.data[idx]
#         formatted = self.format_fn(preferred, rejected)
#         return tokenize(self.tokenizer, **formatted)


def format_example(prompt: str, reasoning: str) -> dict[str, str]:
    """
    Construct a question / reasoning pair.
    The reasoning should contain the chain-of-thought and final <answer>...</answer> tag.
    """
    return {
        "question": prompt.strip(),
        "answer": reasoning.strip()
    }


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()
    base = BaseLLM()
    model = base.model
    tokenizer = base.tokenizer

    # Add LoRA adapters
    config = LoraConfig(
        r=8,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()

    # Load RFT dataset
    # dataset_path = "homework/rft_data.json"
    # trainset = RFTDataset(tokenizer, dataset_path, format_example)
    
    trainset = TokenizedDataset(tokenizer, Dataset("rft_train"), format_example)
    
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=10,
        learning_rate=2e-4,  # lower learning rate for curated data
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("homework/rft_model")  # or Path(output_dir) / "rft_model"


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
