"""
Module for loading an existing LLM and generating answers.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class ModelConfig:
    model_name: str = "google/flan-t5-base"
    max_input_tokens: int = 512
    max_new_tokens: int = 200


class HuggingFaceModelRunner:
    """
    Runs prompts through an existing Hugging Face instruction model.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Inspired by L1.3-pytorch-basics.ipynb: choose cuda if available, otherwise cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Inspired by L6.0-prompt-engineering.ipynb: load tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        # Inspired by L5.4-gpt.ipynb: autoregressive-style generation from tokenized input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()

    def run_dataframe(self, prompts_df: pd.DataFrame) -> pd.DataFrame:
        outputs: List[str] = []

        # Inspired by L10.2-bias-masking-task.ipynb: tqdm loop over model predictions
        for prompt in tqdm(prompts_df["prompt"], desc="Generating answers"):
            outputs.append(self.generate(prompt))

        result_df = prompts_df.copy()
        result_df["model_name"] = self.config.model_name
        result_df["model_answer"] = outputs
        
        return result_df

