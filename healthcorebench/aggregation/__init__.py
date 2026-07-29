"""Aggregation: recompute summary.json purely from results.jsonl + judgments.jsonl.

No aggregation ever reads in-memory run state; every metric is derived from the persisted
per-sample logs so a summary can be regenerated and verified independently.
"""

from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.aggregation.confidence_interval import wilson_interval
from healthcorebench.aggregation.grouping import group_scores

__all__ = ["summarize_run", "wilson_interval", "group_scores"]
