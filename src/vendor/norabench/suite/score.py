"""Copied from allenai/nora-bench norabench.suite.score"""
from typing import Any
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelUsage
from inspect_ai.solver import SolverSpec
from pydantic import BaseModel


class SolverConfig(BaseModel):
    solver: SolverSpec
    model: str
    model_args: dict[str, Any]

    @classmethod
    def from_eval_log(cls, log: EvalLog) -> "SolverConfig":
        return cls(
            solver=SolverSpec(log.eval.solver or "", log.eval.solver_args or {}),
            model=log.eval.model,
            model_args=log.eval.model_args,
        )


class TaskResult(BaseModel):
    eval_log: EvalLog
    """Evaluation log."""

    metrics: dict[str, float] = {}
    """Flat mapping of metric names to values."""

    model_usages: list[dict[str, ModelUsage]] = []
    """List of model usages per sample."""

    model_costs: list[float] = []
    """List of model costs per sample."""


class ResultSet(BaseModel):
    results: dict[str, TaskResult]
    """Mapping of task to TaskResult."""

    solver_config: SolverConfig
    """Solver configuration for the results."""

    submission_name: str | None = None
    """Leaderboard submission name for the results."""