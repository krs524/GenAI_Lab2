# GenAI Lab 2 - Qwen3-VL Fine-Tuning and ReVA Evaluation

## Overview

In this lab, I completed the required ReVA evaluation code, fine-tuned Qwen3-VL-4B-Instruct using LoRA, and compared the Qwen baseline, fine-tuned Qwen, and VILA-1.5-3B on the ReVA video question-answering benchmark.

## Work Completed

- Completed the required coding TODOs across four Python files.
- Passed all 7 assignment tests.
- Prepared 200 ReVA video question-answering samples for Qwen fine-tuning.
- Completed LoRA fine-tuning and evaluated all three model configurations on 4,000 ReVA test questions.

## Experimental Setup

- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **Environment:** Ubuntu 22.04 on Windows WSL2
- **Qwen model:** Qwen3-VL-4B-Instruct
- **VILA model:** VILA1.5-3b
- **Video frames per question:** 4
- **Qwen maximum image pixels:** 50,176
- **Fine-tuning method:** LoRA (rank 8, alpha 16)
- **Training:** 200 samples, 1 epoch, batch size 1, 200 steps
- **Learning rate:** 2e-7

## Evaluation Results

| Model | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Qwen baseline | 2,574 | 4,000 | 64.35% |
| Qwen LoRA fine-tuned | 2,575 | 4,000 | 64.38% |
| VILA baseline | 1,948 | 4,000 | 48.70% |

## Key Findings

The Qwen baseline achieved 64.35% accuracy, outperforming VILA by 15.65 percentage points on the same evaluation set.

After LoRA fine-tuning, Qwen's accuracy increased from 64.35% to 64.38%. This represents only one additional correct answer out of 4,000 questions, so the result does not show a meaningful overall improvement from fine-tuning.

Temporal grounding was more challenging than the overall evaluation for all three models. Baseline Qwen achieved 45.78%, fine-tuned Qwen reached 46.72%, and VILA achieved 31.25% on this subcategory.

## Limitations

- Fine-tuning used only 200 training samples and one epoch.
- Qwen and VILA have different architectures and video-processing pipelines.
- Training and test video sources may overlap. This was not fully audited, so the results should not be treated as evidence of performance on entirely unseen videos.
- The small difference between baseline and fine-tuned Qwen is not enough to establish a reliable performance improvement.

## Saved Results

- `outputs/qwen_base/qwen_base/result.csv` - Qwen baseline results
- `outputs/qwen_sft/qwen_sft/result.csv` - Fine-tuned Qwen results
- `outputs/vila_reva_v2_full/metrics.json` - VILA evaluation results
- `outputs/model_comparison.csv` - Comparison of all three models

## Reproducibility

The main commands, software environments, and experiment settings are documented in `commands_used.txt`. The provided `INSTRUCTIONS.md` contains the lab workflow.

The full ReVA annotations, video datasets, and model weights were kept locally and are not included in this GitHub repository.
