# Efficiency in GPT-2: Parameter Adaptation, Quantization, and Synthetic Data Augmentation

> Ethan Hersch, Ryan D'Cunha, Abhinav Chinta
> 
> CS 224N Default Final Project: Build GPT-2

__Full paper attached to repo.__

# Introduction

Large-scale compute and data has driven unprecedented advancements in natural language processing, full-parameter fine-tuning and data remains costly. We investigate the empirical trade-offs between downstream performance and resource utilization during the fine-tuning of GPT-2. We first establish full-parameter baselines across sentiment classification, cloze-style paraphrase detection, and sonnet generation. Building upon sonnet generation, we evaluate three methods of training efficiency: (1) parameter efficiency, mapping performance using Low-Rank Adaptation (LoRA), (2) memory efficiency, analyzing the memory footprint and performance of precision scaling and Quantization-Aware Fine-Tuning (QAFT), and (3) data efficiency, formulating a cost-versus-performance evaluation and robustness for synthetic data augmentation utilizing the Gemini 2.5 family (Flash Lite, Flash, and Pro).

We focus on three research questions:
- __Parameter Efficiency__: Can we match performance of full fine-tuning using LoRA by investigating the effects of rank, $\alpha$, and modules fine-tuned? We find fine-tuning on attention and MLP layers matches full fine-tuning performance.
- __Quantization Efficiency__: How does memory footprint through model quantization affect performance with respect quantization during fine-tuning or inference? We observe similar performance with inference quantization and QAFT while reducing memory footprint, but evidence of overfitting from testing zero-shot performance on other tasks.
- __Data Efficiency__: How does adding synthetic data change model performance across Gemini-2.5 family? We find improved performance with synthetic data up to a point where GPT-2 capacity limitations prevent improvement with complex models like Gemini 2.5 Pro.

# Chosen Results

<img width="1936" height="842" alt="image" src="https://github.com/user-attachments/assets/e3083936-15ac-45bd-910f-2bd813a90630" />

We distill synthetic data from the Gemini 2.5 Family and perform SFT using subsets of the synthetic data (size 100, 500, 1000) for sonnet generation (143 train examples). Gemini 2.5 Flash Lite data is not of high enough quality to improve baseline performance. Gemini 2.5 Flash data improves performance roughly 10%. Gemini 2.5 Pro data is too complex to improve the performance of GPT-2 Small.

<img width="960" height="504" alt="Screenshot 2026-03-11 at 1 26 46 PM" src="https://github.com/user-attachments/assets/ded5789d-ff58-4b2a-bc93-7200f52a7fec" />

With the INT4 format, post-training quantization significantly degrades performance. However, if we use QAFT, we still see considerable memory and latency reductions (78 to 103 tokens per second on inference). This introduces overfitting and a reduction in zero-shot performance on other tasks.

## A note on the implementation of extensions

> We implemented LoRA from scratch, so we did not use the default Hugging Face package.

> We used `bitsandbytes` for our implementation of quantization and did not do this from scratch.

# Insights
First, we demonstrate that LoRA can match full fine-tuned performance in sonnet generation while reducing trainable parameters and mitigating overfitting. Second, while inference-time quantization degrades performance at lower precisions, QAFT effectively stabilizes performance on sonnet generation, but it risks task-specific overfitting and reduced zero-shot capabilities for paraphrase detection. Finally, our synthetic data evaluation using the Gemini 2.5 family reveals that while higher-quality data improves distillation, GPT-2's capacity limits prevent full knowledge transfer. Notably, augmenting training with synthetic sonnets preserves zero-shot paraphrase accuracy, suggesting this scale of augmentation does not induce catastrophic overfitting.

___

# Technical Instructions

## Testing Instructions
To test Part 1, run:

- `optimizer_test.py`: To test your implementation of `optimizer.py`.
- `sanity_check.py`: To test your implementation of GPT models.
- `classifier.py` : To perform sentiment classification using your models.

In Part 2 of this project, use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of
another as well as generate sonnets via autoregressive language modeling.

To test Part 2, you will run:

- `paraphrase_detection.py`: To perform paraphrase detection.
- `sonnet_generation.py`: To perform sonnet generation.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

### To activate UV venv

`source .venv/bin/activate`

Created `cs224n_gpt` venv with all dependencies

To add a dependency: `(cs224n_gpt) (base) ethanhersch@DN525j91 cs224n_gpt % uv pip install pronouncing`

## To submit `para-dev-output.csv`

You **must** change the labels from `0` and `1` to the proper labels. See Ed post: https://edstem.org/us/courses/90535/discussion/7715611

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).
