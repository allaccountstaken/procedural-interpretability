from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

Score = Literal["PASS", "FAIL", "UNCERTAIN"]
Action = Literal["PROCEED", "REFUSE", "ASK_CLARIFICATION", "ESCALATE"]

class CheckpointResult(BaseModel):
    id: str
    question: str
    score: Score
    evidence: str = ""
    rationale: str = ""

class ModuleResult(BaseModel):
    name: str
    result: Score
    checkpoints: List[CheckpointResult]

class DecisionTrace(BaseModel):
    task_id: str
    model_condition: str
    context_condition: str
    checklist_enabled: bool
    task: str
    context: str
    answer: str
    final_action: Action
    misuse_result: Optional[Score] = None
    misalignment_result: Optional[Score] = None
    mistake_result: Optional[Score] = None
    consistency_check: Optional[Score] = None
    modules: List[ModuleResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VerifierResult(BaseModel):
    trace_id: str
    certify: Literal["CERTIFY", "REJECT", "HUMAN_REVIEW"]
    reasons: List[str]
    detected_failure_type: Literal[
        "none", "misuse_miss", "misalignment_miss", "mistake_miss", "inconsistency", "insufficient_evidence", "other"
    ] = "none"
