# AAIPL Hackathon Solution — Qwen3-4B MCQ Agents

## Overview
This repository contains our full solution for the AAIPL (AMD AI Academy Premier League) hackathon, focused on generating and solving hard MCQ questions using Qwen3-4B and Unsloth on AMD MI300X hardware.

- **Q-Agent**: Generates extremely difficult MCQs in JSON format.
- **A-Agent**: Solves MCQs with step-by-step reasoning and outputs answer+explanation in JSON.
- **All code is in the `AAIPL` folder (here).**
- **No model weights are included.**

## Key Technologies Used
- **Qwen3-4B** (base and finetuned)
- **Unsloth** for fast training/inference
- **TRL SFTTrainer** for supervised finetuning
- **LoRA** (r=64, alpha=128, all projection layers)
- **PyTorch** (bfloat16, ROCm/MI300X)
- **Robust JSON extraction** (regex, brace-matching, markdown block parsing)
- **Adaptive verification** (2-way for some topics, none for Syllogisms)
- **Balanced answer distribution** (A/B/C/D)

## Pipeline Summary
1. **Data Generation**
   - Used Qwen3-4B as a teacher with `enable_thinking=False` to avoid <think> tags.
   - Generated 400 MCQs (100 per topic) with adaptive verification:
     - Syllogisms: No verification (model can't self-solve)
     - Seating/Family/Series: 2-way verification (majority voting)
   - Ensured robust JSON extraction and auto-fix for all outputs.
   - Balanced answer hints for A/B/C/D.

2. **Finetuning**
   - Used Unsloth + TRL SFTTrainer for both Q-Agent and A-Agent.
   - LoRA adapters on all projection layers (r=64, alpha=128).
   - 2 epochs, batch size 32, LR=2e-4.
   - Saved to `hf_models/q_agent_finetuned` and `hf_models/a_agent_finetuned`.

3. **Agent Implementation**
   - **Q-Agent**: `python -m agents.question_agent` — generates MCQs to `outputs/questions.json`.
   - **A-Agent**: `python -m agents.answer_agent` — solves MCQs from `outputs/filtered_questions.json` to `outputs/answers.json`.
   - Both agents use robust JSON extraction (handles <think> tags, markdown, concatenated JSONs).
   - Double-prefix choices (e.g., "A) A) Daughter") are auto-fixed.
   - All outputs are filtered for format and validity.

4. **Evaluation**
   - Outputs are saved to `outputs/` as required.
   - `.gitignore` ensures no model weights or large data are pushed.
   - All code is ready for direct evaluation as per AAIPL requirements.

## Techniques Implemented
- **Adaptive Verification**: 2-way for hard topics, none for Syllogisms.
- **Robust JSON Extraction**: Regex, brace-matching, markdown block parsing, auto-fix for common errors.
- **Answer Distribution Balancing**: Ensures A/B/C/D are evenly used.
- **Double-Prefix Choice Fix**: Cleans up model hallucinations in choices.
- **ZeroDivisionError Guards**: All metrics are safe from division by zero.
- **No Hardcoding/RAG**: All data is generated and solved by the model, no external lookups.

## How to Run
1. Clone the repo and install dependencies (see requirements in notebook).
2. Place your finetuned models in `hf_models/` (not included here).
3. Run:
   - `python -m agents.question_agent` to generate questions.
   - `python -m agents.answer_agent` to solve questions.
4. Outputs will be in `outputs/`.

## Notes
- All code is in the AAIPL folder as required.
- No model weights or large data are included in the repo.
- All outputs are in JSON as per competition requirements.

---

**For any questions, contact: aayush110410@gmail.com**
