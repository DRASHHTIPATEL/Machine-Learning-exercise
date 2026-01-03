# Machine Learning Engineer – Technical Exercise

This repository contains my complete submission for the Machine Learning Engineer technical assessment. The work covers model implementation and training, research analysis focused on AI safety and reliability, code review and optimization, and an optional scaling law analysis. The goal of this submission is to demonstrate not only technical correctness, but also structured thinking, awareness of trade-offs, and production-oriented machine learning practices.

---

## Repository Structure
Machine-Learning-exercise/
│
├── src/                      # Core implementation
│   ├── data/                 # Dataset loading and preprocessing
│   ├── models/               # Transformer-based models
│   ├── train/                # Training loop, logging, checkpointing
│   ├── eval/                 # Evaluation, error and edge-case analysis
│   └── viz/                  # Attention visualization utilities
│
├── configs/                  # YAML configuration files
│
├── scripts/                  # Train / evaluate / visualize entry points
│
├── reports/
│   ├── report.md             # Part 1: Model report
│   └── safety_reliability.md # Part 2: Research analysis (exported to PDF)
│
├── scaling_law_analysis.ipynb # Bonus: Scaling law analysis
│
├── artifacts/                # Logs and checkpoints (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Part 1: Model Implementation & Training

### Objective
Implement and train a transformer-based model for a text task while demonstrating modern ML architectures, training practices, interpretability, and systematic evaluation.

### Implementation Overview
- Transformer-based sentiment analysis model
- Config-driven experiments using YAML
- Custom PyTorch training loop with:
  - Validation and early stopping
  - Checkpoint saving and resume support
  - Metric tracking (loss, accuracy / F1)
- Regularization techniques (weight decay, dropout)
- Attention weight visualization for interpretability
- Error analysis and edge-case evaluation
- Ablation study on selected architectural or training choices

### Dataset
- SST-2 (GLUE benchmark)

This dataset was chosen for reproducibility, fast iteration, and well-established evaluation standards.

### Running the Code

Install dependencies:
```bash
pip install -r requirements.txt

Train the model:
python scripts/train.py --config configs/sst2.yaml

Evaluate the model:
python scripts/evaluate.py

Visualize attention weights:
 python scripts/visualize_attention.py

Outputs
	•	Trained model checkpoints
	•	TensorBoard logs
	•	Attention visualizations
	•	Written report discussing design decisions, results, and limitations

Part 2: Research Analysis & Problem Solving

Objective

Analyze safety and reliability challenges in conversational AI systems and propose technically grounded solutions.

Issues Analyzed
	•	Inconsistent responses across conversation turns
	•	Hallucination and overconfident factual errors
	•	Demographic bias
	•	Prompt sensitivity to small wording changes

Key Contributions
	•	Root-cause analysis for each issue
	•	Measurement and evaluation strategies
	•	Prioritization of hallucination and inconsistency as the most critical risks
	•	Detailed solution proposals, including:
	•	Retrieval-augmented generation
	•	Verification and calibration mechanisms
	•	Explicit conversational state tracking
	•	Experimental design with control vs. treatment setups
	•	Discussion of safety–performance trade-offs and user communication

The full technical memo is available in reports/safety_reliability.md and is also exported to PDF.

⸻

Part 3: Code Review & Optimization

Objective

Demonstrate the ability to critically review machine learning code, identify issues, and propose concrete improvements.

What Was Covered
	•	Identification of correctness bugs (tensor shapes, gradient accumulation, masking)
	•	Missing architectural components (positional encoding, causal masking)
	•	Performance and efficiency considerations
	•	ML best practices for training stability and maintainability

Improvements Made
	•	Refactored transformer implementation with:
	•	Correct input shapes
	•	Positional encoding
	•	Causal attention masking
	•	Proper gradient handling
	•	Device management and padding support
	•	Clear explanation of why each change matters

The review is written in a first-person, engineering-focused style and included directly as part of the submission.

⸻

Bonus Challenge: Scaling Law Analysis

Objective

Analyze hypothetical training results using scaling laws to:
	•	Fit a loss–compute relationship
	•	Predict expected loss at larger scales
	•	Recommend optimal allocation under a fixed compute budget

What’s Included
	•	Jupyter notebook (scaling_law_analysis.ipynb) containing:
	•	Log–log regression fits
	•	Visualizations
	•	Loss prediction for a 10B-parameter, 1T-token model
	•	Compute-optimal recommendations under 20 PF-days
	•	Written discussion of assumptions and limitations

Key Insight

Due to strong coupling between model size and dataset size in the provided data, the analysis focuses on compute-scaling behavior and highlights the uncertainty involved in extrapolation.

Reproducibility & Best Practices
	•	Configuration-driven experiments
	•	Modular and documented code
	•	Clear separation between modeling, training, and evaluation
	•	Explicit discussion of limitations and trade-offs
	•	Focus on clarity, robustness, and responsible ML development

Final Notes

This repository reflects how I approach real-world machine learning problems: ensuring correctness first, building toward robustness and interpretability, and carefully considering safety, evaluation, and long-term implications. Where appropriate, I favored clear reasoning and principled design over unnecessary complexity.
