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

from src.tinyllama_model_runner import TinyLlamaModelConfig, TinyLlamaModelRunner

from src.evaluator import (
    BaseEvaluator,
    TruthfulQAEvaluator,
    HotpotQAEvaluator,
)

from src.explainability import ProbabilityTracer
from src.visualization import ResultsVisualizer
from src.failure_analysis import FailureAnalyzer
from src.utils import save_dataframe


class Experiment:
    """
    Full experiment pipeline for the project.

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

        backend = self.config["model"].get("backend", "tinyllama")

        print(f"Using model backend: {backend}")
        print(f"Using model name: {self.config['model']['model_name']}")

        if backend == "tinyllama":
            model_config = TinyLlamaModelConfig(
                model_name=self.config["model"]["model_name"],
                max_input_tokens=self.config["model"]["max_input_tokens"],
                max_new_tokens=self.config["model"]["max_new_tokens"],
                temperature=self.config["model"].get("temperature", 0.0),
                do_sample=self.config["model"].get("do_sample", False),
                device=self.config["model"].get("device", "auto"),
            )

            self.runner = TinyLlamaModelRunner(model_config)

        else:
            raise ValueError(f"Unknown model backend: {backend}")

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
            self.probability_summary_df,
        )

        print("Main plots created.")
    
    def run_explainability(self) -> None:
        """
        Run probability-based explainability for TinyLlama.

        This step compares the model probability assigned to the truthful answer
        and to the false answer.
        """

        if self.runner is None:
            raise ValueError("Model runner is not initialized. Run the model first.")

        if self.evaluated_df is None:
            raise ValueError("Evaluated outputs are not available. Run evaluation first.")

        probability_tracer = ProbabilityTracer(self.runner)

        self.traced_df = probability_tracer.trace_dataframe(
            self.evaluated_df,
            top_k=10,
        )

        self.probability_summary_df = probability_tracer.summarize_probability_tracing(
            self.traced_df
        )

        save_dataframe(
            self.traced_df,
            self.outputs_dir / "probability_tracing_outputs.csv",
        )

        save_dataframe(
            self.probability_summary_df,
            self.outputs_dir / "probability_tracing_summary.csv",
        )

        print("Probability tracing completed.")
        print(self.probability_summary_df)

    def run(self) -> None:
        """
        Execute the complete experiment.
        """

        self.load_data()
        self.build_prompts()
        self.run_model()
        self.evaluate()
        self.run_explainability()
        self.visualize_results()
        self.analyze_failures()

        print("Full experiment completed.")
