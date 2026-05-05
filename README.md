# Procedural Interpretability: Checklists for AI Control

This repository is a small, reproducible evaluation harness for testing whether checklist-based procedural interpretability changes LLM agent behavior under context perturbation and model degradation.

Core hypothesis:

> Checklist-based procedural interpretability may improve bounded decision reliability by converting implicit human expectations into explicit runtime controls and verifiable decision traces.

The checklist does **not** certify the model. It certifies the **decision trace**.

## Experiment matrix

| Model condition | Context condition | Checklist condition |
|---|---|---|
| clean | clean | off / on |
| clean | problematic | off / on |
| degraded | clean | off / on |
| degraded | problematic | off / on |

## Repository structure

```text
checklists/                 Human-approved checklist specs
configs/                    Reproducible experiment configs
data/tasks/                 Evaluation task set
data/contexts/              Clean and problematic context snippets
perturbations/              Context/model degradation recipes
notebooks/                  Demo notebooks
src/procedural_interp/      Python package for running experiments
results/                    Local result artifacts; ignored except .gitkeep
docs/                       Design notes and white paper drafts
scripts/                    CLI entry points
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# add OPENAI_API_KEY to .env
jupyter lab
```

Start with:

- `notebooks/01_minimal_decision_trace.ipynb`
- `notebooks/02_experiment_matrix.ipynb`

## Design principles

1. Checklists are isolated from prompts.
2. Perturbations are isolated from tasks.
3. Model conditions are explicit and configurable.
4. Every run produces a structured JSON trace.
5. Human labels, LLM judge labels, and deterministic checks are stored separately.
6. The repo should support later Hugging Face Space deployment.

## Current status

Prototype scaffold. Not yet a validated evaluation benchmark.
