# AstroToolBench (working name)

Benchmark and post-training pipeline for **reliable scientific tool use by LLM agents**,
using spacecraft flight dynamics as the experimental domain.

**Research question.** Can an open-source LLM reliably operate numerical spacecraft-dynamics
tools, and how much do tool/API design, post-training and inference optimization improve
its reliability?

## Status
- [x] Core flight-dynamics functions (two-body propagation, Hohmann, eclipses, closest approach)
- [x] Unit tests against analytic results and textbook values (Vallado Ex. 6-1)
- [ ] Benchmark tasks (`benchmark/tasks.jsonl`)
- [ ] Tool exposure to agents (MCP) and evaluation harness
- [ ] Post-training (PyTorch, LoRA) and inference engineering (vLLM)

## Conventions
km, km/s, s, rad, Earth-centered inertial frame. Two-body dynamics, cylindrical shadow.

## Quick start
```bash
pip install -e ".[dev]"
pytest
```
