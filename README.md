# Truth, Lies, and Reasoning Machines

## Goal

This project investigates how misleading, false, or contradictory information affects the factual accuracy and logical consistency of a Large Language Model (LLM).

---

## Research Question

How does misleading or contradictory information in prompts affect the factual accuracy and logical consistency of an LLM, and can self-verification prompts improve reasoning robustness?

---

## Datasets

The project uses two datasets:

### TruthfulQA

A subset of 25 examples from TruthfulQA is used to test truthfulness and resistance to common false beliefs.

### HotpotQA

A subset of 25 examples from HotpotQA is used to test multi-hop reasoning with context and supporting facts.

Raw files must be placed in:

* data/raw/TruthfulQA.csv
* data/raw/hotpot_dev_distractor_v1.json

---

## Model

The project uses:

* google/flan-t5-base

The model is not trained or fine-tuned. It is only evaluated.

The model can be changed in:

* config.yaml

---

## Experimental Conditions

Each example is tested in four conditions:

1. **Baseline**  
   Original factual question.

2. **Noisy**  
   A false or contradictory statement is added before the question.

3. **Adversarial**  
   The model is instructed to assume a false statement is true.

4. **Self-verification**  
   The model is asked to verify assumptions before answering.

---

## Output Format

The model is asked to answer with:

* Reasoning:
...

* Final Answer:
...

This allows analysis of both the reasoning process and the final answer.

---

## Evaluation

The outputs are evaluated using:

- factual accuracy,
- false premise resistance,
- logical consistency,
- reasoning chain presence,
- belief persistence,
- circular logic detection,
- qualitative failure analysis.

---

## Explainability

The project includes:

- probability tracing,
- attention visualization,
- qualitative failure analysis.

These help understand where the model deviates from truthful reasoning.

---

## Installation

pip install -r requirements.txt

---

## Run the Experiment

From the project root:

python run_experiment.py

---

## Outputs

The experiment saves results in:

* results/outputs/
* results/plots/
