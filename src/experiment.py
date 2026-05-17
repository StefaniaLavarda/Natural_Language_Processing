"""
Main experiment pipeline.

This module defines the Experiment class, which controls the full
experimental workflow:

1. load data
2. build prompts
3. run model
4. evaluate outputs
5. analyze failures
6. run explainability
7. save results
"""

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.data_loader import (
    TruthfulQAConfig,
    TruthfulQADataLoader,
    HotpotQAConfig,
    HotpotQADataLoader,
)

from src.prompt_builder import (
    TruthfulQAPromptBuilder,
    HotpotQAPromptBuilder,
)

from src.model_runner import ModelConfig, HuggingFaceModelRunner

from src.evaluator import (
    BaseEvaluator,
    TruthfulQAEvaluator,
    HotpotQAEvaluator,
)

from src.visualization import ResultsVisualizer
from src.explainability import ProbabilityTracer, AttentionVisualizer
from src.failure_analysis import FailureAnalyzer
from src.utils import save_dataframe


class Experiment:
    """
    Full experiment pipeline for the project.

    The notebook should only demonstrate or inspect the results.
    The full reproducible experiment is run from this class.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.processed_dir = Path(config["outputs"]["processed_dir"])
        self.outputs_dir = Path(config["outputs"]["outputs_dir"])
        self.plots_dir = Path(config["outputs"]["plots_dir"])

        self.truthfulqa_df = None
        self.hotpotqa_df = None

        self.truthfulqa_prompts_df = None
        self.hotpotqa_prompts_df = None
        self.prompts_df = None

        self.outputs_df = None
        self.truthfulqa_evaluated_df = None
        self.hotpotqa_evaluated_df = None
        self.evaluated_df = None
        self.summary_df = None

        self.failure_analysis_df = None
        self.failure_summary_df = None

        self.traced_df = None
        self.probability_summary_df = None

        self.runner = None
        self.visualizer = ResultsVisualizer(output_dir=str(self.plots_dir))

    def load_data(self) -> None:
        """
        Load TruthfulQA and HotpotQA using separate dataset loaders.
        """

        truthfulqa_config = TruthfulQAConfig(
            data_path=self.config["data"]["truthfulqa_path"],
            sample_size=self.config["data"]["truthfulqa_sample_size"],
            random_state=self.config["project"]["seed"],
        )

        hotpotqa_config = HotpotQAConfig(
            data_path=self.config["data"]["hotpotqa_path"],
            sample_size=self.config["data"]["hotpotqa_sample_size"],
            random_state=self.config["project"]["seed"],
            max_context_paragraphs=self.config["data"]["hotpotqa_max_context_paragraphs"],
        )

        truthfulqa_loader = TruthfulQADataLoader(truthfulqa_config)
        hotpotqa_loader = HotpotQADataLoader(hotpotqa_config)

        self.truthfulqa_df = truthfulqa_loader.load()
        self.hotpotqa_df = hotpotqa_loader.load()

        save_dataframe(
            self.truthfulqa_df,
            self.processed_dir / "truthfulqa_subset.csv",
        )

        save_dataframe(
            self.hotpotqa_df,
            self.processed_dir / "hotpotqa_subset.csv",
        )

        print("Loaded datasets:")
        print(self.truthfulqa_df["dataset"].value_counts())
        print(self.hotpotqa_df["dataset"].value_counts())

    def build_prompts(self) -> None:
        """
        Build prompts separately for TruthfulQA and HotpotQA.
        """

        truthfulqa_prompt_builder = TruthfulQAPromptBuilder()
        hotpotqa_prompt_builder = HotpotQAPromptBuilder()

        self.truthfulqa_prompts_df = truthfulqa_prompt_builder.build_prompt_dataset(
            self.truthfulqa_df
        )

        self.hotpotqa_prompts_df = hotpotqa_prompt_builder.build_prompt_dataset(
            self.hotpotqa_df
        )

        self.prompts_df = pd.concat(
            [self.truthfulqa_prompts_df, self.hotpotqa_prompts_df],
            ignore_index=True,
        )

        save_dataframe(
            self.truthfulqa_prompts_df,
            self.processed_dir / "truthfulqa_prompts.csv",
        )

        save_dataframe(
            self.hotpotqa_prompts_df,
            self.processed_dir / "hotpotqa_prompts.csv",
        )

        save_dataframe(
            self.prompts_df,
            self.processed_dir / "combined_prompts.csv",
        )

        print("Prompt counts:")
        print(self.prompts_df.groupby(["dataset", "condition"]).size())

    def run_model(self) -> None:
        """
        Run the selected language model on all prompts.
        """

        model_config = ModelConfig(
            model_name=self.config["model"]["model_name"],
            max_input_tokens=self.config["model"]["max_input_tokens"],
            max_new_tokens=self.config["model"]["max_new_tokens"],
        )

        self.runner = HuggingFaceModelRunner(model_config)

        self.outputs_df = self.runner.run_dataframe(self.prompts_df)

        save_dataframe(
            self.outputs_df,
            self.outputs_dir / "model_outputs.csv",
        )

        print("Model inference completed.")

    def evaluate(self) -> None:
        """
        Evaluate TruthfulQA and HotpotQA outputs separately, then combine them.
        """

        truthfulqa_outputs_df = self.outputs_df[
            self.outputs_df["dataset"] == "truthfulqa"
        ].copy()

        hotpotqa_outputs_df = self.outputs_df[
            self.outputs_df["dataset"] == "hotpotqa"
        ].copy()

        fuzzy_threshold = self.config["evaluation"]["fuzzy_threshold"]

        truthfulqa_evaluator = TruthfulQAEvaluator(
            fuzzy_threshold=fuzzy_threshold
        )

        hotpotqa_evaluator = HotpotQAEvaluator(
            fuzzy_threshold=fuzzy_threshold
        )

        base_evaluator = BaseEvaluator(
            fuzzy_threshold=fuzzy_threshold
        )

        self.truthfulqa_evaluated_df = truthfulqa_evaluator.evaluate(
            truthfulqa_outputs_df
        )

        self.hotpotqa_evaluated_df = hotpotqa_evaluator.evaluate(
            hotpotqa_outputs_df
        )

        self.evaluated_df = pd.concat(
            [self.truthfulqa_evaluated_df, self.hotpotqa_evaluated_df],
            ignore_index=True,
        )

        self.summary_df = base_evaluator.summarize_metrics(self.evaluated_df)

        save_dataframe(
            self.truthfulqa_evaluated_df,
            self.outputs_dir / "truthfulqa_evaluated_outputs.csv",
        )

        save_dataframe(
            self.hotpotqa_evaluated_df,
            self.outputs_dir / "hotpotqa_evaluated_outputs.csv",
        )

        save_dataframe(
            self.evaluated_df,
            self.outputs_dir / "evaluated_outputs.csv",
        )

        save_dataframe(
            self.summary_df,
            self.outputs_dir / "metrics_summary.csv",
        )

        print("Evaluation completed.")
        print(self.summary_df)

    def analyze_failures(self) -> None:
        """
        Run qualitative failure analysis.
        """

        failure_analyzer = FailureAnalyzer()

        self.failure_analysis_df = failure_analyzer.create_failure_analysis_table(
            self.evaluated_df
        )

        self.failure_summary_df = failure_analyzer.summarize_failure_types(
            self.failure_analysis_df
        )

        save_dataframe(
            self.failure_analysis_df,
            self.outputs_dir / "qualitative_failure_analysis.csv",
        )

        save_dataframe(
            self.failure_summary_df,
            self.outputs_dir / "qualitative_failure_summary.csv",
        )

        print("Failure analysis completed.")
        print(self.failure_summary_df)

    def visualize_results(self) -> None:
        """
        Create main metric plots.
        """

        self.visualizer.create_all_plots(
            self.summary_df,
            self.evaluated_df,
        )

        print("Main plots created.")

    def run_probability_tracing(self) -> None:
        """
        Run probability tracing explainability analysis.
        """

        if not self.config["analysis"]["run_probability_tracing"]:
            print("Probability tracing skipped.")
            return

        probability_tracer = ProbabilityTracer(self.runner)

        self.traced_df = probability_tracer.trace_dataframe(self.evaluated_df)

        self.probability_summary_df = probability_tracer.summarize_probability_tracing(
            self.traced_df
        )

        save_dataframe(
            self.traced_df,
            self.outputs_dir / "probability_tracing.csv",
        )

        save_dataframe(
            self.probability_summary_df,
            self.outputs_dir / "probability_tracing_summary.csv",
        )

        self.visualizer.plot_truthful_preference_rate(
            self.probability_summary_df,
            "truthful_preference_rate.png",
        )

        print("Probability tracing completed.")

    def run_attention_analysis(self) -> None:
        """
        Run attention analysis for a small set of examples.
        Saves attention scores and top attended tokens, but does not create heatmaps.
        """

        if not self.config["analysis"]["run_attention_analysis"]:
            print("Attention analysis skipped.")
            return

        attention_visualizer = AttentionVisualizer(self.runner)

        examples_per_group = self.config["analysis"]["attention_examples_per_group"]

        attention_examples = (
            self.evaluated_df
            .groupby(["dataset", "condition"])
            .head(examples_per_group)
        )

        save_dataframe(
            attention_examples,
            self.outputs_dir / "attention_examples.csv",
        )

        for _, row in attention_examples.iterrows():
            dataset = row["dataset"]
            condition = row["condition"]
            prompt = row["prompt"]

            attention_df = attention_visualizer.attention_to_dataframe(prompt)

            attention_output_path = (
                self.outputs_dir
                / f"attention_scores_{dataset}_{condition}.csv"
            )

            save_dataframe(attention_df, attention_output_path)

            token_attention_df = attention_visualizer.summarize_token_attention(prompt)

            save_dataframe(
                token_attention_df,
                self.outputs_dir / f"top_attention_tokens_{dataset}_{condition}.csv",
            )

        print("Attention analysis completed.")

    def run(self) -> None:
        """
        Execute the complete experiment.
        """

        self.load_data()
        self.build_prompts()
        self.run_model()
        self.evaluate()
        self.visualize_results()
        self.analyze_failures()
        self.run_probability_tracing()
        self.run_attention_analysis()

        print("Full experiment completed.")
