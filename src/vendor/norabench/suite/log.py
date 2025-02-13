"""Copied from allenai/nora-bench norabench.suite.log"""
from logging import getLogger

from inspect_ai.model import ModelUsage
from litellm import cost_per_token

logger = getLogger(__name__)


def compute_model_cost(model_usage: dict[str, ModelUsage]) -> float:
    """
    Compute aggregate cost for a given dictionary of ModelUsage.
    """
    total_cost = 0.0
    for model, usage in model_usage.items():
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
            total_cost += prompt_cost + completion_cost
        except Exception as e:
            total_cost = float("nan")
            logger.warning(f"Problem calculating cost for model {model}: {e}")
            break
    return total_cost