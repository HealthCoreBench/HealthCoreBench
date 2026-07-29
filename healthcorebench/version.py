"""Central version constants for the framework and its on-disk schemas.

These are recorded into every run's manifest and into individual JSONL records so that
logs remain interpretable across framework upgrades. Bump the relevant constant whenever
the corresponding behaviour or on-disk layout changes.
"""

# Version of the HealthCoreBench framework code as a whole.
FRAMEWORK_VERSION = "2.5.1"

# Version of the JSONL / manifest / summary record schemas. Bump on any
# backward-incompatible change to persisted record layouts.
SCHEMA_VERSION = "1.3"

# Version of the aggregation (summary) code. Recorded in summary.json so a summary can be
# traced to the exact aggregation logic that produced it. Bump when grouping / metric
# formulas change.
SUMMARY_CODE_VERSION = "1.7"

# Default version tags for pluggable components. Individual adapters / parsers /
# evaluators may override these with their own values.
DEFAULT_ADAPTER_VERSION = "1.0"
DEFAULT_PARSER_VERSION = "1.3"
DEFAULT_EVALUATOR_VERSION = "1.0"
DEFAULT_PROMPT_TEMPLATE_VERSION = "1.0"
