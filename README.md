# Truth, Lies, and Reasoning Machines

## Goal

This project investigates how misleading, false, or contradictory information affects the factual accuracy and logical consistency of a Large Language Model (LLM).

The project focuses on reasoning robustness under "truth distortion" scenarios and studies whether models can resist misinformation and maintain coherent reasoning.

---

## Research Question

How does misleading or contradictory information in prompts affect the factual accuracy and logical consistency of an LLM, and can self-verification prompts improve reasoning robustness?

---

## Datasets

The project uses two datasets:

### TruthfulQA

A subset of 50 examples from TruthfulQA is used to evaluate whether language models generate truthful answers instead of reproducing common misconceptions and false beliefs.

### HotpotQA

A subset of 25 examples from HotpotQA is used to evaluate multi-hop reasoning under misleading information.

TruthfulQA focuses on truthfulness and misinformation resistance.

HotpotQA focuses on reasoning under multi-step inference.

---

## Model

The experiments use the existing instruction-tuned model:

`google/flan-t5-base`

The goal is not to train a new model, but to evaluate the behavior of an existing LLM under different reasoning conditions.

---

## Experimental Conditions

Each example is transformed into four prompt conditions:

1. **Baseline**  
   Original factual question.

2. **Noisy**  
   A false or contradictory statement is added before the question.

3. **Adversarial**  
   The model is instructed to assume a false statement is true.

4. **Self-verification**  
   The model is asked to verify assumptions before answering.

---

## Reasoning Format

The model is instructed to generate outputs using:

```text
Reasoning:
(step-by-step explanation)

Final Answer:
(short answer)