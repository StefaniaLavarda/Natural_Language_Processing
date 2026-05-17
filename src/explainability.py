"""
Module for probability tracing explainability.

This module compares how much the model supports:
1. the truthful answer
2. the false premise

across baseline, noisy, adversarial, and self-verification prompts.
"""

from typing import Dict, List

import torch
import pandas as pd
from tqdm import tqdm


class ProbabilityTracer:
    """
    Computes sequence-level log-probability scores for candidate answers.
    """

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.model = model_runner.model
        self.tokenizer = model_runner.tokenizer
        self.device = model_runner.device

    def score_candidate_answer(self, prompt: str, candidate_answer: str) -> float:
        # Inspired by L5.4-gpt.ipynb: analyze generation probabilities token by token
        # Inspired by L10.2-bias-masking-task.ipynb: inspect model probabilities for specific tokens/answers

        if not candidate_answer or len(str(candidate_answer).strip()) == 0:
            return float("-inf")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_runner.config.max_input_tokens,
        ).to(self.device)

        labels = self.tokenizer(
            candidate_answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_runner.config.max_new_tokens,
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

        negative_log_likelihood = outputs.loss.item()

        return -negative_log_likelihood

    def trace_dataframe(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []

        for _, row in tqdm(
            evaluated_df.iterrows(),
            total=len(evaluated_df),
            desc="Tracing probabilities",
        ):
            truthful_score = self.score_candidate_answer(
                row["prompt"],
                row["reference_answer"],
            )

            false_premise = row.get("false_premise", "")

            if row.get("condition") == "baseline" or not str(false_premise).strip():
                false_score = None
                score_difference = None
                prefers_truthful = None
            else:
                false_score = self.score_candidate_answer(
                    row["prompt"],
                    false_premise,
                )

                score_difference = truthful_score - false_score
                prefers_truthful = int(score_difference > 0)

            traced_row = row.to_dict()
            traced_row["truthful_answer_nll_score"] = truthful_score
            traced_row["false_premise_nll_score"] = false_score
            traced_row["truthful_minus_false_nll_score"] = score_difference
            traced_row["model_prefers_truthful_answer"] = prefers_truthful

            rows.append(traced_row)

        return pd.DataFrame(rows)

    def summarize_probability_tracing(self, traced_df: pd.DataFrame) -> pd.DataFrame:
        comparison_df = traced_df.dropna(
            subset=[
                "false_premise_nll_score",
                "truthful_minus_false_nll_score",
                "model_prefers_truthful_answer",
            ]
        )

        summary = (
            comparison_df
            .groupby(["dataset", "condition"])
            .agg(
                avg_truthful_answer_nll_score=("truthful_answer_nll_score", "mean"),
                avg_false_premise_nll_score=("false_premise_nll_score", "mean"),
                avg_truthful_minus_false_nll_score=("truthful_minus_false_nll_score", "mean"),
                truthful_preference_rate=("model_prefers_truthful_answer", "mean"),
            )
            .reset_index()
        )

        return summary

class AttentionVisualizer:
    """
    Extracts and summarizes encoder attention from FLAN-T5.

    The goal is to inspect whether the model attends strongly to misleading
    tokens in noisy/adversarial prompts.
    """

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.model = model_runner.model
        self.tokenizer = model_runner.tokenizer
        self.device = model_runner.device

    def get_encoder_attention(self, prompt: str):
        # Inspired by L5.1-transformers.ipynb: inspect multi-head attention inside Transformer models
        # Inspired by L9.0-explainability.ipynb: use internal model behavior for interpretability

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_runner.config.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
                return_dict=True,
            )

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        attentions = outputs.attentions

        last_layer_attention = attentions[-1]

        mean_attention = last_layer_attention.mean(dim=1)[0]

        return tokens, mean_attention.cpu().numpy()

    def attention_to_dataframe(self, prompt: str) -> pd.DataFrame:
        tokens, attention_matrix = self.get_encoder_attention(prompt)

        rows = []

        for i, source_token in enumerate(tokens):
            for j, target_token in enumerate(tokens):
                rows.append({
                    "source_token": source_token,
                    "target_token": target_token,
                    "attention_score": attention_matrix[i][j],
                })

        return pd.DataFrame(rows)

    def summarize_token_attention(self, prompt: str) -> pd.DataFrame:
        tokens, attention_matrix = self.get_encoder_attention(prompt)

        token_scores = attention_matrix.mean(axis=0)

        attention_df = pd.DataFrame({
            "token": tokens,
            "mean_attention_received": token_scores,
        })

        attention_df = attention_df.sort_values(
            by="mean_attention_received",
            ascending=False,
        ).reset_index(drop=True)

        return attention_df
