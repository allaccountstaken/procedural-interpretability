# CONVENTIONS.md
# Procedural Interpretability — Development Conventions

This document defines coding standards, architectural decisions, and naming
conventions for the procedural-interpretability repository. Claude Code and
all contributors should read this before making changes.

---

## Project identity

This is a research project, not a production application. Code quality
standards serve reproducibility and credibility, not scale or performance.
Every structural decision should make the experiment easier to understand,
re-run, and extend — not faster or more clever.

---

## Repository structure

```
checklists/         Human-approved checklist specs (YAML). Never auto-generated.
configs/            Reproducible experiment configs (YAML). One file per experiment run.
data/tasks/         Evaluation task definitions (JSONL). Human-authored or human-reviewed.
data/contexts/      Context snippets, clean and problematic (JSONL). One file per domain.
data/degradation/   Fine-tuning datasets for LoRA degradation. Keep private or gated.
data/labels/        Human annotation labels for verifier calibration (JSONL).
perturbations/      Context perturbation type definitions (YAML).
results/            Experiment outputs (JSONL). Ignored by git except .gitkeep.
docs/               Research documentation. Markdown only.
notebooks/          Demo and exploration notebooks. Not used for final results.
scripts/            CLI entry points. Thin wrappers around src/ only.
src/procedural_interp/  Core Python package. All logic lives here.
tests/              pytest test suite. Covers regression and schema validation.
```

---

## Python conventions

### General

- Python 3.11+. Use `from __future__ import annotations` in all modules.
- Type hints on all function signatures. No untyped public functions.
- Pydantic for all data objects that cross module boundaries. No raw dicts
  between modules.
- No global mutable state. Configuration is passed explicitly, not imported
  as module-level variables.
- All file I/O goes through `src/procedural_interp/io.py`. Do not call
  `open()`, `json.load()`, or `yaml.safe_load()` directly in other modules.

### Naming

- Functions: `snake_case` verbs — `run_one`, `load_checklist`, `build_model`.
- Classes: `PascalCase` nouns — `DecisionTrace`, `OpenAIChatModel`, `ExperimentRun`.
- Constants: `UPPER_SNAKE_CASE` — `BASE_SYSTEM`, `DEFAULT_RESULTS_PATH`.
- Config keys: `snake_case` matching Python names exactly — `checklist_path`,
  `model_condition`, `context_condition`.
- Files: `snake_case` — `experiment_hf_demo.yaml`, `sample_tasks.jsonl`.

### Module responsibilities

Each module has one responsibility. Do not add responsibilities without
updating this document.

| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic data models only. No logic. |
| `io.py` | File reading and writing only. No business logic. |
| `checklists.py` | Load and render checklist YAML to prompt string. |
| `prompts.py` | Prompt templates as module-level string constants only. |
| `models.py` | LLM client wrappers. One class per provider. |
| `runner.py` | Experiment orchestration. Calls models, checklists, evaluator. |
| `evaluator.py` | Verification and scoring logic. No LLM calls except `llm_verify`. |
| `gradio_app.py` | UI only. Calls runner. No business logic. |
| `inspect_adapter.py` | Inspect AI interface only. Wraps runner and evaluator. |

---

## Model backend conventions

All LLM client classes must implement exactly this interface:

```python
class SomeModel:
    def __init__(self, model: str, temperature: float = 0.0): ...
    def json_call(self, system: str, user: str) -> Dict[str, Any]: ...
```

- `json_call` always returns a parsed dictionary, never a raw string.
- `json_call` never raises on malformed JSON — use `_extract_json` from
  `models.py` to handle fences and partial responses.
- Temperature 0.0 is the default for all experiment runs. Non-zero
  temperature is only used for degraded model conditions and must be
  set explicitly in config, never in code.
- Provider routing lives in `build_model` in `runner.py`. Do not add
  provider logic elsewhere.

Current providers and their config `provider` key:

| Provider key | Class | Notes |
|---|---|---|
| `openai` | `OpenAIChatModel` | Requires `OPENAI_API_KEY` in `.env` |
| `huggingface` | `HFChatModel` | Requires `HF_TOKEN` in `.env` |

---

## Schema conventions

- All schemas live in `schemas.py`. No Pydantic models in other modules.
- All schemas use `model_config = ConfigDict(extra="forbid")` to catch
  unexpected fields from LLM responses immediately.
- All schemas carry `schema_version: str` defaulting to the current version
  string. Increment when fields are added or removed.
- Current schema version: `"1.0"`
- Do not change field names in existing schemas without incrementing the
  version. Old result files must remain parseable.

```python
# Correct — version is explicit and default is set
class DecisionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    task_id: str
    ...

# Wrong — no version, no extra="forbid"
class DecisionTrace(BaseModel):
    task_id: str
    ...
```

---

## Checklist conventions

- Checklists are YAML files in `checklists/`. They are the primary research
  artifact. Do not auto-generate them. Every question is a deliberate
  research decision.
- Checklist IDs follow `{name}_v{major}.{minor}` — `core_mmm_v0.1`.
  Increment minor for question edits. Increment major for structural changes.
- Each question must have a unique `id` within its module — `M1`–`M10`,
  `A1`–`A10`, `E1`–`E10`.
- Each module must have a `decision_rule` field. This is what the verifier
  checks against.
- Never modify a checklist that has been used in a published experiment run
  without creating a new version. Result files reference checklist version
  for reproducibility.

---

## Data conventions

### Tasks (`data/tasks/`)

Every task must have these fields and no others:

```json
{
  "task_id": "T001",
  "domain": "finance_compliance",
  "risk_level": "high",
  "user_request": "...",
  "expected_safe_action": "clarify_or_retrieve",
  "label_notes": "Human annotation guidance for verifier calibration."
}
```

- `domain` must be one of: `finance_compliance`, `enterprise_agent`,
  `legal_rag`, `security_policy`.
- `risk_level` must be one of: `high`, `medium`, `low`.
- `expected_safe_action` must be one of: `proceed`, `refuse`,
  `clarify_or_retrieve`, `refuse_or_escalate`, `clarify_or_escalate`.
- `label_notes` is mandatory. It tells human annotators what to look for.
  Do not leave it empty.

### Contexts (`data/contexts/`)

Every context must pair with a task and have both `clean` and `problematic`
variants:

```json
{"context_id": "C001_clean",     "task_id": "T001", "condition": "clean",       "text": "..."}
{"context_id": "C001_problematic","task_id": "T001", "condition": "problematic", "text": "..."}
```

- `condition` must be one of the perturbation types defined in
  `perturbations/context_perturbations.yaml`.
- Problematic contexts must be realistic, not exotic. They should represent
  failure modes an agent would encounter in production RAG systems.

### Human labels (`data/labels/`)

```json
{
  "trace_id": "T001:clean:problematic:true",
  "annotator_id": "human_01",
  "correct_action": "ASK_CLARIFICATION",
  "failure_detected": true,
  "failure_type": "mistake_miss",
  "notes": "..."
}
```

---

## Config conventions

- One YAML file per experiment. Name as `experiment_{description}.yaml`.
- All paths in config are relative to the repository root.
- All model names that come from environment variables use `${VAR_NAME}`
  syntax consistently.
- Never hardcode API keys or model names in config files. Use `.env` and
  `${VAR_NAME}` references.
- Every config file must have an `experiment_id` field that is unique across
  all configs. This is the primary key for experiment tracking.

```yaml
# Correct
experiment_id: hf_lora_degradation_002
checklist_path: checklists/core_mmm_v0.1.yaml
models:
  clean:
    provider: huggingface
    model: mistralai/Mistral-7B-Instruct-v0.3
    temperature: 0.0

# Wrong — no experiment_id, hardcoded model name in wrong place
checklist: checklists/core_mmm_v0.1.yaml
model: gpt-4.1-mini
```

---

## Results conventions

- Results are JSONL files in `results/`. One line per `DecisionTrace`.
- Results files are named `{experiment_id}.jsonl` matching the config.
- Results are never edited after writing. Append only.
- Results are gitignored. Share via HuggingFace Datasets, not git.
- Every trace written to results must include `schema_version`,
  `experiment_id`, and a UTC timestamp in `metadata`.

```python
# Add to run_one before writing trace
trace.metadata["experiment_id"] = cfg["experiment_id"]
trace.metadata["timestamp_utc"] = datetime.utcnow().isoformat()
```

---

## Verifier hierarchy

Five verification mechanisms exist in this project. Their roles are fixed
and must not be conflated in reported results.

| Layer | Mechanism | Role | Report in paper? |
|---|---|---|---|
| 1 | `deterministic_consistency` | Filter only. Catches PROCEED-with-FAIL. | No |
| 2 | `llm_verify` | Primary automated metric. | Yes |
| 3 | Human labels | Ground truth for calibrating layer 2. | Yes (as calibration) |
| 4 | DeepEval judges | Regression testing only. | No |
| 5 | Inspect AI scorers | Reproducibility layer for AI safety community. | As supplement |

Layer 2 is what the paper reports as the primary finding.

---

## Testing conventions

- Tests live in `tests/`. Use pytest.
- At minimum, one schema validation test per Pydantic model.
- At minimum, one regression test per task/expected_action pair in the
  sample task set.
- Tests must run without API calls by default. Use fixtures with saved
  trace objects for unit tests. Mark tests requiring live API calls with
  `@pytest.mark.api` and exclude them from CI.

```python
# Correct — no API call in unit test
def test_deterministic_consistency_flags_proceed_with_fail(sample_trace_fixture):
    sample_trace_fixture.final_action = "PROCEED"
    sample_trace_fixture.misuse_result = "FAIL"
    result = deterministic_consistency(sample_trace_fixture)
    assert result["passed"] is False

# Wrong — makes live API call in unit test
def test_llm_verify_live():
    model = OpenAIChatModel("gpt-4.1-mini")
    ...
```

---

## Documentation conventions

- All research decisions go in `docs/`. Code comments explain how.
  Documentation explains why.
- `experiment_design.md` is the canonical research document. Update it
  when the research question, metrics, or expected results change.
- `research_infrastructure.md` is the tooling roadmap. Update it when
  architectural decisions are made or reversed.
- `CONVENTIONS.md` (this file) is updated when any convention changes.
  Date each update.

---

## What Claude Code should never do

- Write checklist questions. These are research decisions made by the
  investigator.
- Write `expected_safe_action` values for tasks. These require domain
  judgment about financial, legal, and security risk.
- Write `label_notes` for tasks. These guide human annotators and must
  reflect genuine domain expertise.
- Modify results files. Results are append-only research artifacts.
- Change schema field names without being explicitly asked and confirming
  the version increment.
- Add new dependencies to `pyproject.toml` without confirming with the
  investigator that the dependency is necessary and fits the architecture.

---

*Last updated: sprint start. Update this document when conventions change.*
