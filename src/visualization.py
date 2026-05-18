"""
Module for visualizing experimental results.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


class ResultsVisualizer:
    """
    Creates plots for the project report and presentation.
    """

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metric_by_condition(
        self,
        summary_df: pd.DataFrame,
        metric: str,
        title: str,
        filename: str,
    ) -> None:
        plot_df = summary_df.copy()
        plot_df["label"] = plot_df["dataset"] + " | " + plot_df["condition"]

        plt.figure(figsize=(12, 5))
        plt.bar(plot_df["label"], plot_df[metric])
        plt.ylim(0, 1)
        plt.xlabel("Dataset and prompt condition")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()

    def plot_reasoning_failure_types(
        self,
        evaluated_df: pd.DataFrame,
        filename: str,
    ) -> None:
        # Inspired by L10.0-biases.ipynb: categorize and visualize qualitative model failures
        failure_columns = [
            "belief_persistence",
            "possible_circular_logic",
            "accepted_false_premise",
        ]

        failure_counts = evaluated_df[failure_columns].sum()

        plt.figure(figsize=(8, 5))
        plt.bar(failure_counts.index, failure_counts.values)
        plt.xlabel("Reasoning failure type")
        plt.ylabel("Count")
        plt.title("Reasoning Failure Types")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()

    def plot_truthful_preference_rate(
        self,
        probability_summary_df: pd.DataFrame,
        filename: str,
    ) -> None:
        plot_df = probability_summary_df.copy()
        plot_df["label"] = plot_df["dataset"] + " | " + plot_df["condition"]

        plt.figure(figsize=(12, 5))
        plt.bar(
            plot_df["label"],
            plot_df["truthful_preference_rate"],
        )
        plt.ylim(0, 1)
        plt.xlabel("Dataset and prompt condition")
        plt.ylabel("Truthful Preference Rate")
        plt.title("How Often the Model Prefers the Truthful Answer")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()

    def create_all_plots(
        self,
        summary_df: pd.DataFrame,
        evaluated_df: pd.DataFrame,
        probability_summary_df: pd.DataFrame = None,
    ) -> None:
        self.plot_metric_by_condition(
            summary_df,
            "factual_accuracy",
            "Factual Accuracy by Prompt Condition",
            "factual_accuracy_by_condition.png",
        )

        self.plot_metric_by_condition(
            summary_df,
            "false_premise_resistance",
            "False Premise Resistance by Prompt Condition",
            "false_premise_resistance_by_condition.png",
        )

        self.plot_metric_by_condition(
            summary_df,
            "logical_consistency",
            "Logical Consistency by Prompt Condition",
            "logical_consistency_by_condition.png",
        )

        self.plot_reasoning_failure_types(
            evaluated_df,
            "reasoning_failure_types.png",
        )

        if probability_summary_df is not None and not probability_summary_df.empty:
            self.plot_truthful_preference_rate(
                probability_summary_df,
                "truthful_preference_rate.png",
            )