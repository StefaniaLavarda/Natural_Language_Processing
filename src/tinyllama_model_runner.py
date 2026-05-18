"""
TinyLlama model runner.

This module runs prompts through TinyLlama using Hugging Face Transformers.
It also exposes next-token probability inspection for explainability.
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class TinyLlamaModelConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_input_tokens: int = 1024
    max_new_tokens: int = 120
    temperature: float = 0.0
    do_sample: bool = False
    device: str = "auto"


class TinyLlamaModelRunner:
    """
    Runs prompts through TinyLlama.
    """

    def __init__(self, config: TinyLlamaModelConfig):
        self.config = config
        self.device = self._select_device(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, device: str) -> str:
        if device != "auto":
            return device

        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def generate(self, prompt: str) -> str:
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
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return answer.strip()

    def run_dataframe(self, prompts_df: pd.DataFrame) -> pd.DataFrame:
        outputs: List[str] = []

        for prompt in tqdm(prompts_df["prompt"], desc="Generating answers"):
            outputs.append(self.generate(prompt))

        result_df = prompts_df.copy()
        result_df["model_name"] = self.config.model_name
        result_df["model_answer"] = outputs

        return result_df

    def inspect_next_token_probabilities(
        self,
        prompt: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Inspect next-token probabilities after a prompt.
        Useful for probability-based explainability.
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        tokens = [
            self.tokenizer.decode(index.item())
            for index in top_indices[0]
        ]

        return [
            (token, float(prob))
            for token, prob in zip(tokens, top_probs[0].cpu())
        ]