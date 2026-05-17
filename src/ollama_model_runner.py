"""
Ollama model runner.

This module runs prompts through a local Ollama instruction model,
for example llama3.2:3b.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd
from tqdm import tqdm
import ollama


@dataclass
class OllamaModelConfig:
    model_name: str 
    max_input_tokens: int 
    max_new_tokens: int 
    temperature: float 


class OllamaModelRunner:
    """
    Runs prompts through a local Ollama model.
    """

    def __init__(self, config: OllamaModelConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        response = ollama.generate(
            model=self.config.model_name,
            prompt=prompt,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens,
                "num_ctx": self.config.max_input_tokens,
            },
        )

        return response["response"].strip()

    def run_dataframe(self, prompts_df: pd.DataFrame) -> pd.DataFrame:
        outputs: List[str] = []

        for prompt in tqdm(prompts_df["prompt"], desc="Generating answers"):
            outputs.append(self.generate(prompt))

        result_df = prompts_df.copy()
        result_df["model_name"] = self.config.model_name
        result_df["model_answer"] = outputs

        return result_df
