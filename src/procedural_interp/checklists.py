from __future__ import annotations
from pathlib import Path
from .io import load_yaml


def load_checklist(path: str | Path) -> dict:
    return load_yaml(path)


def checklist_to_prompt(checklist: dict) -> str:
    lines = [
        f"Checklist: {checklist['name']} ({checklist['version']})",
        "For each checkpoint, answer with PASS, FAIL, or UNCERTAIN, plus evidence and a short rationale.",
        "After all modules, choose final_action: PROCEED, REFUSE, ASK_CLARIFICATION, or ESCALATE.",
        "Return valid JSON only using the requested schema."
    ]
    for module_name, module in checklist["modules"].items():
        lines.append(f"\nMODULE: {module_name.upper()} — {module['description']}")
        lines.append(f"Decision rule: {module.get('decision_rule','')}")
        for q in module["questions"]:
            lines.append(f"- {q['id']}: {q['text']}")
    return "\n".join(lines)
