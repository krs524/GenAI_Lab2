# GenAI Lab 2 — ReVA Vision-Language Model Evaluation

## Project Overview

This project evaluates vision-language models on the instructor-provided ReVA video question-answering benchmark and investigates the effect of LoRA fine-tuning.

## Completed Work

- Completed all 18 assigned coding TODOs across four Python files.
- Passed all 7 provided tests.
- Prepared 200 video question-answering samples for Qwen fine-tuning.
- Evaluated three model configurations on 4,000 ReVA questions.

## Experimental Setup

- GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- Qwen model: Qwen3-VL-4B-Instruct
- VILA model: VILA1.5-3b
- Video frames per question: 4
- Qwen image pixel limit: 50,176
- Qwen fine-tuning: LoRA, rank 8, alpha 16
- Training: 1 epoch, batch size 1, 200 steps
- Learning rate: 2e-7

## Overall Results

| Model | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Qwen baseline | 2,574 | 4,000 | 64.35% |
| Qwen LoRA fine-tuned | 2,575 | 4,000 | 64.38% |
| VILA baseline | 1,948 | 4,000 | 48.70% |

## Findings

Qwen outperformed VILA on the supplied benchmark. The original Qwen model exceeded VILA by 15.65 percentage points.

LoRA fine-tuning increased Qwen's number of correct answers by only one out of 4,000. This difference is negligible and should not be interpreted as a meaningful improvement.

Temporal grounding remained challenging. The fine-tuned Qwen model achieved 46.72% on this subcategory, compared with 31.25% for VILA.

## Limitations

- Qwen and VILA use different model architectures and video-processing pipelines.
- Fine-tuning was limited to 200 training samples and one epoch.
- Possible overlap between training and test video sources requires further auditing. These results should not be presented as proof of generalization to entirely unseen videos.
- The small fine-tuning improvement does not establish statistical significance.

## Saved Results

- `outputs/qwen_base/qwen_base/result.csv`
- `outputs/qwen_sft/qwen_sft/result.csv`
- `outputs/vila_reva_v2_full/metrics.json`
- `outputs/model_comparison.csv`

## Reproducibility

See `INSTRUCTIONS.md` for the advisor-provided workflow. A separate `commands_used.txt` will document environment setup, execution commands, and compatibility fixes.

Large model weights, video datasets, and instructor-provided annotations are not included in this public repository.
