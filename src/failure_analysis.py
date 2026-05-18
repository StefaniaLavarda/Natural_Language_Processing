"""
Module for qualitative failure analysis.

This module automatically selects interesting model outputs and assigns
heuristic failure labels. It also creates empty columns for optional manual
inspection and notes.
"""

from typing import List

import pandas as pd


class FailureAnalyzer:
    """
    Creates a qualitative failure-analysis table from evaluated model outputs.
    """

    def _normalize(self, text: str) -> str:
        return str(text).lower().strip()

    def detect_hallucination_propagation(self, row: pd.Series) -> int:
        # Inspired by L10.3-bias-completion-task.ipynb: false context may propagate into generated completions
        reasoning = self._normalize(row.get("reasoning_chain", ""))
        final_answer = self._normalize(row.get("final_answer", ""))

        is_wrong = row.get("factual_accuracy", 0) == 0
        uses_false_premise = row.get("belief_persistence", 0) == 1
        has_extra_reasoning = len(reasoning.split()) > 12

        return int(is_wrong and uses_false_premise and has_extra_reasoning and len(final_answer) > 0)

    def detect_self_correction_success(self, row: pd.Series) -> int:
        # Inspired by L6.0-prompt-engineering.ipynb: self-verification prompts can trigger correction behavior
        text = self._normalize(
            str(row.get("reasoning_chain", "")) + " " + str(row.get("final_answer", ""))
        )

        correction_markers = [
            "the statement is false",
            "the statement is incorrect",
            "the statement is not true",
            "the premise is false",
            "the premise is incorrect",
            "this is false",
            "this is incorrect",
            "this is not true",
            "the claim is false",
            "the claim is incorrect",
            "the claim is not supported",
            "not supported by the context",
            "contradicted by the context",
            "cannot assume that",
            "there is no evidence that",
        ]

        is_correct = row.get("factual_accuracy", 0) == 1
        corrects_premise = any(marker in text for marker in correction_markers)

        return int(is_correct and corrects_premise)

    def detect_vague_or_evasive(self, row: pd.Series) -> int:
        final_answer = self._normalize(row.get("final_answer", ""))

        # If the evaluator says the answer is factually correct,
        # do not classify it as vague only because it is short.
        if row.get("factual_accuracy", 0) == 1:
            return 0

        vague_markers = [
            "it depends",
            "cannot determine",
            "not enough information",
            "unknown",
            "unclear",
            "i don't know",
            "not specified",
            "not provided",
            "cannot be determined",
        ]

        contains_vague_marker = any(marker in final_answer for marker in vague_markers)

        # Empty or almost empty wrong answers are vague.
        too_short_and_wrong = len(final_answer.split()) <= 2

        return int(too_short_and_wrong or contains_vague_marker)

    def assign_automatic_failure_type(self, row: pd.Series) -> str:
        """
        Assigns one main qualitative label.
        """

        if self.detect_self_correction_success(row) == 1:
            return "self_correction_success"

        if self.detect_hallucination_propagation(row) == 1:
            return "hallucination_propagation"

        if row.get("belief_persistence", 0) == 1:
            return "belief_persistence"

        if row.get("possible_circular_logic", 0) == 1:
            return "circular_logic"

        if self.detect_vague_or_evasive(row) == 1:
            return "vague_or_evasive"

        if row.get("factual_accuracy", 0) == 1 and row.get("logical_consistency", 0) == 1:
            return "correct_reasoning"

        if row.get("factual_accuracy", 0) == 0:
            return "wrong_or_hallucinated_answer"

        return "other"

    def create_failure_analysis_table(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a table for qualitative analysis.

        The examples are selected automatically.
        The manual columns are intentionally empty.
        """

        df = evaluated_df.copy()

        automatic_labels: List[str] = []

        for _, row in df.iterrows():
            automatic_labels.append(self.assign_automatic_failure_type(row))

        df["automatic_failure_type"] = automatic_labels

        selected_df = df[
            (df["automatic_failure_type"] != "correct_reasoning")
            | (df["condition"].isin(["noisy", "adversarial", "self_verification"]))
        ].copy()

        selected_columns = [
            "id",
            "dataset",
            "condition",
            "question",
            "false_premise",
            "false_answer",
            "reasoning_chain",
            "final_answer",
            "model_answer",
            "factual_accuracy",
            "logical_consistency",
            "false_premise_resistance",
            "belief_persistence",
            "possible_circular_logic",
            "accepted_false_premise",
            "error_type",
            "automatic_failure_type",
        ]

        selected_df = selected_df[selected_columns]

        selected_df["manual_failure_type"] = ""
        selected_df["manual_notes"] = ""

        return selected_df.reset_index(drop=True)

    def summarize_failure_types(self, failure_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            failure_df["automatic_failure_type"]
            .value_counts()
            .reset_index()
        )

        summary.columns = ["automatic_failure_type", "count"]

        return summary