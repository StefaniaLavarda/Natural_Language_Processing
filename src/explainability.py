"""
Explainability module for TinyLlama.

This module uses top-k next-token probability tracing.

For each prompt, it checks how much probability the model assigns to:
1. words from the truthful/reference answer
2. words from the false answer

"""

from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm


def normalize_token(token: str) -> str:
    """
    Normalize tokens for simple comparison.
    TinyLlama tokens may include spacing markers such as ▁ or Ġ.
    """

    return (
        str(token)
        .replace("▁", "")
        .replace("Ġ", "")
        .strip()
        .lower()
    )


def extract_target_token_probability(
    top_tokens: List[Tuple[str, float]],
    target_words: Iterable[str],
) -> float:
    """
    Sum the probability assigned to target words
    in the model's top-k next-token predictions.
    """

    normalized_targets = {
        normalize_token(word)
        for word in target_words
        if word and str(word).strip()
    }

    probability = 0.0

    for token, prob in top_tokens:
        if normalize_token(token) in normalized_targets:
            probability += float(prob)

    return probability


def get_target_words(answer: str) -> List[str]:
    """
    Extract simple target words from an answer.
    """

    if not answer:
        return []

    cleaned = (
        str(answer)
        .replace(".", "")
        .replace(",", "")
        .replace(";", "")
        .replace(":", "")
        .replace("?", "")
        .replace("!", "")
    )

    words = cleaned.split()

    return words[:3]


class ProbabilityTracer:
    """
    Simple probability tracer based on top-k next-token probabilities.
    """

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def trace_row(
        self,
        row: pd.Series,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Trace next-token probabilities for one evaluated row.
        """

        prompt = row["prompt"]
        reference_answer = row["reference_answer"]
        false_answer = row.get("false_answer", row.get("false_premise", ""))

        answer_prompt = prompt + "\nFinal answer:"

        top_tokens = self.model_runner.inspect_next_token_probabilities(
            prompt=answer_prompt,
            top_k=top_k,
        )
        
        truthful_target_words = get_target_words(reference_answer)
        false_target_words = get_target_words(false_answer)

        truthful_probability = extract_target_token_probability(
            top_tokens,
            truthful_target_words,
        )

        false_probability = extract_target_token_probability(
            top_tokens,
            false_target_words,
        )

        if row.get("condition") == "baseline" or not str(false_answer).strip():
            probability_difference = None
            prefers_truthful = None
        else:
            probability_difference = truthful_probability - false_probability
            prefers_truthful = int(truthful_probability > false_probability)

        traced_row = row.to_dict()

        traced_row["top_next_tokens"] = top_tokens
        traced_row["truthful_target_words"] = truthful_target_words
        traced_row["false_target_words"] = false_target_words

        traced_row["truthful_next_token_probability"] = truthful_probability
        traced_row["false_next_token_probability"] = false_probability
        traced_row["truthful_minus_false_next_token_probability"] = probability_difference
        traced_row["model_prefers_truthful_answer"] = prefers_truthful

        return traced_row

    def trace_dataframe(
        self,
        evaluated_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Apply top-k probability tracing to the full evaluated DataFrame.
        """

        rows: List[Dict[str, Any]] = []

        for _, row in tqdm(
            evaluated_df.iterrows(),
            total=len(evaluated_df),
            desc="Tracing next-token probabilities",
        ):
            rows.append(self.trace_row(row, top_k=top_k))

        return pd.DataFrame(rows)

    def summarize_probability_tracing(
        self,
        traced_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize next-token probability tracing by dataset and condition.
        """

        comparison_df = traced_df.dropna(
            subset=[
                "truthful_minus_false_next_token_probability",
                "model_prefers_truthful_answer",
            ]
        )

        summary = (
            comparison_df
            .groupby(["dataset", "condition"])
            .agg(
                avg_truthful_next_token_probability=(
                    "truthful_next_token_probability",
                    "mean",
                ),
                avg_false_next_token_probability=(
                    "false_next_token_probability",
                    "mean",
                ),
                avg_truthful_minus_false_next_token_probability=(
                    "truthful_minus_false_next_token_probability",
                    "mean",
                ),
                truthful_preference_rate=(
                    "model_prefers_truthful_answer",
                    "mean",
                ),
            )
            .reset_index()
        )

        return summary
