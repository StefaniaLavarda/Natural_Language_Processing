"""
Module for loading and preparing TruthfulQA data and HotPotQA data.
"""

from dataclasses import dataclass
from typing import Any, List
import json

import pandas as pd


@dataclass
class TruthfulQAConfig:
    data_path: str
    sample_size: int
    random_state: int


@dataclass
class HotpotQAConfig:
    data_path: str
    sample_size: int
    random_state: int
    max_context_paragraphs: int


class TruthfulQADataLoader:
    """
    Loads TruthfulQA from a local CSV file.
    """

    def __init__(self, config: TruthfulQAConfig):
        self.config = config

    def _split_semicolon_answers(self, value: Any) -> List[str]:
        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        return [answer.strip() for answer in str(value).split(";") if answer.strip()]

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)

        df = df.sample(
            n=min(self.config.sample_size, len(df)),
            random_state=self.config.random_state,
        ).reset_index(drop=True)

        processed_df = pd.DataFrame({
            "id": [f"truthfulqa_{i}" for i in range(len(df))],
            "dataset": "truthfulqa",
            "source_id": df.index.astype(str),
            "type": df["Type"],
            "category": df["Category"],
            "question": df["Question"],
            "best_answer": df["Best Answer"],
            "best_incorrect_answer": df["Best Incorrect Answer"],
            "correct_answers": df["Correct Answers"].apply(self._split_semicolon_answers),
            "incorrect_answers": df["Incorrect Answers"].apply(self._split_semicolon_answers),
            "source": df["Source"],
        })

        return processed_df


class HotpotQADataLoader:
    """
    Loads HotpotQA from a local JSON file.
    """

    def __init__(self, config: HotpotQAConfig):
        self.config = config

    def _format_context(self, context: Any) -> str:
        """
        Converts HotpotQA context into readable text.
        Keeps a limited number of paragraphs to avoid very long prompts.
        """

        if not isinstance(context, list):
            return ""

        formatted_paragraphs = []

        for item in context[: self.config.max_context_paragraphs]:
            if not isinstance(item, list) or len(item) != 2:
                continue

            title = item[0]
            sentences = item[1]

            if isinstance(sentences, list):
                paragraph = " ".join(sentences)
            else:
                paragraph = str(sentences)

            formatted_paragraphs.append(f"{title}: {paragraph}")

        return "\n".join(formatted_paragraphs)

    def load(self) -> pd.DataFrame:
        with open(self.config.data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        df = pd.DataFrame(data)

        df = df.sample(
            n=min(self.config.sample_size, len(df)),
            random_state=self.config.random_state,
        ).reset_index(drop=True)

        processed_df = pd.DataFrame({
            "id": [f"hotpotqa_{i}" for i in range(len(df))],
            "dataset": "hotpotqa",
            "source_id": df["_id"],
            "question": df["question"],
            "answer": df["answer"],
            "supporting_facts": df["supporting_facts"],
            "context": df["context"],
            "formatted_context": df["context"].apply(self._format_context),
            "type": df["type"],
            "level": df["level"],
        })

        return processed_df
