# Truth, Lies, and Reasoning Machines

This project investigates how a small Large Language Model behaves when it is exposed to false, misleading, or contradictory information.

The model used is **TinyLlama/TinyLlama-1.1B-Chat-v1.0**. The experiment tests the model on two datasets:

- **TruthfulQA**, to study truthfulness and resistance to common misconceptions.
- **HotpotQA**, to study multi-hop question answering using provided context.

Raw files must be placed in:

* data/raw/TruthfulQA.csv
* data/raw/hotpot_dev_distractor_v1.json

## Goal

The main research question is:

**How does a small instruction-tuned language model behave when exposed to false or misleading information during question answering?**

## Methodology

For each question, the project creates four prompt conditions:

1. **Baseline**: normal question answering.
2. **Noisy**: the prompt includes a possibly wrong suggested answer.
3. **Adversarial**: the false information is presented as reliable.
4. **Self-verification**: the model is asked to check the premise before answering.

## Output Format

The model is asked to answer with:

* Reasoning:
...

* Final Answer:
...

This allows analysis of both the reasoning process and the final answer.

## Evaluation

The outputs are evaluated using:

- fuzzy match score
- factual accuracy
- false-premise acceptance
- false-premise resistance
- logical consistency
- qualitative failure analysis
- next-token probability tracing

## Installation

pip install -r requirements.txt

## Run the Experiment

From the project root:

python run_experiment.py

## Outputs

The experiment saves results in:

* results/outputs/
* results/plots/