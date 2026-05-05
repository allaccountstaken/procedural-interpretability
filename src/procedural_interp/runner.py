from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from .io import load_yaml, load_jsonl, append_jsonl
from .checklists import load_checklist, checklist_to_prompt
from .prompts import BASE_SYSTEM, NO_CHECKLIST_PROMPT, WITH_CHECKLIST_PROMPT
from .models import OpenAIChatModel
from .schemas import DecisionTrace
from .evaluator import deterministic_consistency


def build_model(model_cfg: dict) -> OpenAIChatModel:
    model_name = os.path.expandvars(model_cfg["model"])
    # Manual replacement for ${VAR} syntax used in YAML.
    if model_name.startswith("${") and model_name.endswith("}"):
        model_name = os.getenv(model_name[2:-1], "gpt-4.1-mini")
    return OpenAIChatModel(model=model_name, temperature=float(model_cfg.get("temperature", 0.0)))


def select_context(contexts: list[dict], task_id: str, condition: str) -> str:
    for row in contexts:
        if row["task_id"] == task_id and row["condition"] == condition:
            return row["text"]
    raise KeyError(f"No context found for {task_id=} {condition=}")


def run_one(task: dict, context: str, model_condition: str, context_condition: str, checklist_enabled: bool, model, checklist_prompt: str | None) -> DecisionTrace:
    if checklist_enabled:
        prompt = WITH_CHECKLIST_PROMPT.format(task=task["user_request"], context=context, checklist_prompt=checklist_prompt)
    else:
        prompt = NO_CHECKLIST_PROMPT.format(task=task["user_request"], context=context)
    raw = model.json_call(BASE_SYSTEM, prompt)
    trace = DecisionTrace(
        task_id=task["task_id"],
        model_condition=model_condition,
        context_condition=context_condition,
        checklist_enabled=checklist_enabled,
        task=task["user_request"],
        context=context,
        **raw,
    )
    trace.metadata["deterministic_consistency"] = deterministic_consistency(trace)
    trace.metadata["expected_safe_action"] = task.get("expected_safe_action")
    return trace


def run_experiment(config_path: str = "configs/experiment_openai_demo.yaml") -> list[DecisionTrace]:
    load_dotenv()
    cfg = load_yaml(config_path)
    checklist = load_checklist(cfg["checklist_path"])
    checklist_prompt = checklist_to_prompt(checklist)
    tasks = load_jsonl(cfg["tasks_path"])
    contexts = load_jsonl(cfg["contexts_path"])
    traces = []
    for model_condition in cfg["conditions"]["model_condition"]:
        model = build_model(cfg["models"][model_condition])
        for context_condition in cfg["conditions"]["context_condition"]:
            for checklist_enabled in cfg["conditions"]["checklist"]:
                for task in tasks:
                    context = select_context(contexts, task["task_id"], context_condition)
                    trace = run_one(task, context, model_condition, context_condition, checklist_enabled, model, checklist_prompt)
                    traces.append(trace)
                    append_jsonl(cfg["results_path"], [trace.model_dump()])
    return traces


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_openai_demo.yaml")
    args = parser.parse_args()
    traces = run_experiment(args.config)
    print(f"Wrote {len(traces)} traces")

if __name__ == "__main__":
    main()
