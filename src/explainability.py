"""
Explainability module for TinyLlama.

This module provides lightweight interpretability tools compatible with
TinyLlamaModelRunner:

1. next-token probability tracing
2. target-token probability extraction
3. sequence-level candidate answer scoring

It compares how much the model supports:
- the truthful/reference answer
- the false answer introduced in noisy/adversarial/self-verification prompts
"""

from typing import Dict, Iterable, List, Tuple, Any

import pandas as pd
import torch
from tqdm import tqdm


def normalize_token(token: str) -> str:
    """
    Normalize a token for simple comparison.

    This removes common tokenizer spacing markers and lowercases the token.
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
    Estimate how much probability the model assigns to target words
    in the top-k next-token distribution.

    This is a lightweight interpretability measure.
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


def get_reference_target_words(reference_answer: str) -> List[str]:
    """
    Extract simple candidate target words from the reference answer.

    Works best for short factual answers such as names, places, dates,
    yes/no answers, or short entities.
    """

    if not reference_answer:
        return []

    cleaned = (
        str(reference_answer)
        .replace(".", "")
        .replace(",", "")
        .replace(";", "")
        .replace(":", "")
    )

    words = cleaned.split()

    return words[:3]


def get_false_target_words(false_answer: str) -> List[str]:
    """
    Extract simple candidate target words from the false answer.
    """

    return get_reference_target_words(false_answer)


class ProbabilityTracer:
    """
    Probability-based explainability for TinyLlama.

    This class supports two analyses:

    1. next-token probability:
       Checks whether the next-token distribution gives probability mass
       to words from the truthful or false answer.

    2. sequence-level candidate scoring:
       Computes the average log-probability of a candidate answer
       conditioned on the prompt.
    """

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.model = model_runner.model
        self.tokenizer = model_runner.tokenizer
        self.device = model_runner.device

    def score_candidate_answer(self, prompt: str, candidate_answer: str) -> float:
        """
        Compute the average log-probability of a candidate answer
        conditioned on a prompt.

        Higher score means the model assigns higher probability to the answer.
        """

        if not candidate_answer or len(str(candidate_answer).strip()) == 0:
            return float("-inf")

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_runner.config.max_input_tokens,
        )["input_ids"][0]

        answer_ids = self.tokenizer(
            " " + str(candidate_answer),
            return_tensors="pt",
            truncation=True,
            max_length=self.model_runner.config.max_new_tokens,
            add_special_tokens=False,
        )["input_ids"][0]

        if answer_ids.numel() == 0:
            return float("-inf")

        max_total_length = (
            self.model_runner.config.max_input_tokens
            + self.model_runner.config.max_new_tokens
        )

        total_length = prompt_ids.shape[0] + answer_ids.shape[0]

        if total_length > max_total_length:
            max_prompt_length = max_total_length - answer_ids.shape[0]
            prompt_ids = prompt_ids[-max_prompt_length:]

        input_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0).to(self.device)

        attention_mask = torch.ones_like(input_ids).to(self.device)

        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[0]] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        negative_log_likelihood = outputs.loss.item()

        return -negative_log_likelihood

    def inspect_prompt_next_tokens(
        self,
        prompt: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Inspect the model's top-k next-token probabilities after a prompt.
        """

        return self.model_runner.inspect_next_token_probabilities(
            prompt=prompt,
            top_k=top_k,
        )

    def trace_row(
        self,
        row: pd.Series,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Run explainability tracing for one evaluated row.
        """

        prompt = row["prompt"]
        reference_answer = row["reference_answer"]
        false_answer = row.get("false_answer", row.get("false_premise", ""))

        top_tokens = self.inspect_prompt_next_tokens(prompt, top_k=top_k)

        reference_target_words = get_reference_target_words(reference_answer)
        false_target_words = get_false_target_words(false_answer)

        truthful_next_token_probability = extract_target_token_probability(
            top_tokens,
            reference_target_words,
        )

        false_next_token_probability = extract_target_token_probability(
            top_tokens,
            false_target_words,
        )

        truthful_score = self.score_candidate_answer(
            prompt,
            reference_answer,
        )

        if row.get("condition") == "baseline" or not str(false_answer).strip():
            false_score = None
            score_difference = None
            prefers_truthful = None
            next_token_probability_difference = None
        else:
            false_score = self.score_candidate_answer(
                prompt,
                false_answer,
            )

            score_difference = truthful_score - false_score
            prefers_truthful = int(score_difference > 0)

            next_token_probability_difference = (
                truthful_next_token_probability - false_next_token_probability
            )

        traced_row = row.to_dict()

        traced_row["top_next_tokens"] = top_tokens

        traced_row["reference_target_words"] = reference_target_words
        traced_row["false_target_words"] = false_target_words

        traced_row["truthful_next_token_probability"] = truthful_next_token_probability
        traced_row["false_next_token_probability"] = false_next_token_probability
        traced_row["truthful_minus_false_next_token_probability"] = (
            next_token_probability_difference
        )

        traced_row["truthful_answer_logprob_score"] = truthful_score
        traced_row["false_answer_logprob_score"] = false_score
        traced_row["truthful_minus_false_logprob_score"] = score_difference
        traced_row["model_prefers_truthful_answer"] = prefers_truthful

        return traced_row

    def trace_dataframe(
        self,
        evaluated_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Apply probability tracing to an evaluated DataFrame.
        """

        rows: List[Dict[str, Any]] = []

        for _, row in tqdm(
            evaluated_df.iterrows(),
            total=len(evaluated_df),
            desc="Tracing probabilities",
        ):
            rows.append(self.trace_row(row, top_k=top_k))

        return pd.DataFrame(rows)

    def summarize_probability_tracing(self, traced_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize probability tracing results by dataset and condition.
        """

        comparison_df = traced_df.dropna(
            subset=[
                "false_answer_logprob_score",
                "truthful_minus_false_logprob_score",
                "model_prefers_truthful_answer",
            ]
        )

        summary = (
            comparison_df
            .groupby(["dataset", "condition"])
            .agg(
                avg_truthful_answer_logprob_score=(
                    "truthful_answer_logprob_score",
                    "mean",
                ),
                avg_false_answer_logprob_score=(
                    "false_answer_logprob_score",
                    "mean",
                ),
                avg_truthful_minus_false_logprob_score=(
                    "truthful_minus_false_logprob_score",
                    "mean",
                ),
                truthful_preference_rate=(
                    "model_prefers_truthful_answer",
                    "mean",
                ),
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
            )
            .reset_index()
        )

        return summary