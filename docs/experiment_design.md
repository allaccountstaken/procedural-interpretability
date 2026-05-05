# Experiment Design Notes

## Research question

Does checklist-based procedural interpretability change LLM agent behavior under context perturbation and model degradation?

## Core comparison

- No checklist vs. checklist
- Clean context vs. problematic context
- Clean model vs. degraded model

## Trace object

The decision trace is the object of evaluation. The model is not certified globally.

## Initial task domains

- Financial compliance RAG
- Enterprise disclosure / authorization
- Legal clause interpretation
- Security policy / prompt injection

## Metrics

- Correct final decision
- Unsafe action rate
- Policy violation rate
- Appropriate escalation rate
- False escalation rate
- Detection of problematic context
- Checklist/action consistency
- Verifier agreement with human labels

## Expected first result

The checklist should help most when failures arise from missing context, wrong attention, prompt injection, ambiguity, policy conflict, or overconfidence. It may help less when the model lacks basic capability or is severely adversarial.
