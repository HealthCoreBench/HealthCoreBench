# AGENTS.md

This file is the repository-level operating guide for coding assistants working on
HealthCoreBench. It applies to every file in this repository. Follow a user's explicit
instructions first, then this guide. Read the relevant implementation and tests before making
changes; do not infer behavior from filenames or old run artifacts alone.

## Project Overview

HealthCoreBench is a resumable, auditable evaluation framework for medical language models
and vision-language models. It communicates with models through OpenAI-compatible Chat
Completions APIs. Model serving is external to this repository: the framework does not load,
train, or deploy model checkpoints.

The Python distribution and import package are both named `healthcorebench`.

Core invariants:

- Inference, parsing, scoring, and aggregation are separate stages.
- Raw model responses remain available after parsing and rescoring.
- A failed API request is recorded as an error and is never scored as an incorrect answer.
- Unknown provider metadata remains `null`; do not invent token counts, model versions,
  probabilities, or medical metrics.
- Run records are append-only. Reparse and rescore operations append new records rather than
  rewriting the original inference response.
- Stable sample identities make results comparable across compatible runs.
- Summaries are derived artifacts and must be reproducible from persisted run records.
- Benchmark discovery is registry-based and uses fixed local data directories.

## Working Rules

Before editing:

1. Run `git status --short`.
2. Read the target code, its schemas, and its tests.
3. Check whether the same file already contains user changes.
4. Define the smallest change that satisfies the request.

While editing:

- Preserve unrelated staged, unstaged, and untracked user work.
- Do not use destructive Git commands or rewrite history unless the user explicitly requests it.
- Do not create commits, push branches, open pull requests, or delete data unless requested.
- Do not modify historical files under `runs/` unless the task explicitly targets run artifacts.
- Do not reformat or rewrite third-party benchmark trees as part of a framework change.
- Do not add large benchmark files, run outputs, caches, virtual environments, or build products
  to Git.
- Treat stability as the default: fix demonstrated behavior, avoid opportunistic refactors, and
  keep diffs focused.
- Prefer deterministic tests and small synthetic fixtures over calls to real model endpoints.
- If an unexpected concurrent change appears while working, stop and ask before overwriting it.

After editing:

1. Run tests appropriate to the changed area.
2. Run `git diff --check`.
3. Review both `git diff` and `git diff --cached`.
4. Report what changed, what was verified, what was not verified, and the final Git state.

## Repository Layout

```text
healthcorebench/
  aggregation/       Run summaries, grouped metrics, confidence intervals, batch reports
  benchmarks/        Registry, base adapter, answer parsing, text and VLM adapter code
  clients/           OpenAI-compatible client, messages, response normalization, errors
  evaluators/        Rule-based, text, structured, grounding, and LLM-judge evaluators
  media/             Image encoding and video handling
  runtime/           Orchestration, execution, retries, recording, resume, rate limiting
  schemas/           Pydantic contracts for config and persisted records
  tools/             Validation, reparse, rescore, export, and legacy migration utilities
  utils/             JSONL, hashing, timestamps, environment, and shared helpers

configs/             Example and full-suite YAML operation files
tests/               Unit, integration, mock HTTP, regression, and end-to-end tests
benchmarks/          Local benchmark data plus its version-controlled inventory README
runs/                Generated evaluation outputs; not version-controlled

README.md            User-facing overview and quick start
benchmarks/README.md Benchmark catalog, source links, descriptions, and selected counts
pyproject.toml       Package metadata, dependencies, and optional dependency groups
requirements.txt     Base dependency installation list
pytest.ini           Pytest configuration
```

Do not confuse these two locations:

- `healthcorebench/benchmarks/` contains first-party Python adapter code.
- `benchmarks/` contains local benchmark datasets and third-party source material.

Only `benchmarks/README.md` is intended for normal version control. The dataset subtrees are
local inputs and are ignored by Git.

## Environment Setup

HealthCoreBench requires Python 3.11 or later. Reuse an active project environment when one is
already available. Otherwise, a standard local setup is:

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Install video support when working on video benchmarks:

```bash
python -m pip install -e '.[test,video]'
```

Installing from the pinned project list is also supported:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use `python -m pip` and `python -m pytest` so commands run in the same interpreter. Tests use
mock clients and a local standard-library HTTP server; they do not require a GPU or an external
model service.

For an actual evaluation, start an OpenAI-compatible server separately and configure its base
URL and served model name in YAML. HealthCoreBench should not acquire GPU resources or launch a
model server as a hidden side effect of a code change.

## Configuration

Run configuration is validated by `healthcorebench/schemas/config.py`. Unknown keys are rejected.
The top-level sections are:

```yaml
experiment:
benchmark:
model:
generation:
runtime:
media:
output:
evaluation:
hardware:
```

Important conventions:

- YAML files are first-class operation files. Preserve the user's chosen configuration style.
- `model.api_key` and `evaluation.judge.api_key` may be supplied directly in YAML.
- `model.api_key_env` and `evaluation.judge.api_key_env` are also supported. A direct key takes
  precedence when both forms are present.
- Do not silently migrate direct YAML values to environment-only configuration or the reverse.
- CLI flags override only the explicitly supported fields: benchmark, split, model, base URL,
  concurrency, and maximum sample count.
- `benchmark.debug_data_path_override` is for non-standard debugging only; such a run must not be
  treated as a normal comparable evaluation.
- Operational runtime values such as concurrency and timeout may change during a compatible
  resume, while output-affecting values participate in run identity checks.
- `output.root_dir` defaults to `runs`.
- `hardware` describes externally managed execution facts; it does not provision hardware.

Before changing a formal YAML file, inspect all related text and multimodal configs and the
configuration regression tests. Do not normalize model names, URLs, evaluator choices, or output
directories unless the requested change requires it.

## Benchmark Data

The fixed local trees are:

```text
benchmarks/medical_llm_benchmarks/<N>_<Name>/
benchmarks/medical_vlm_benchmarks/<N>_<Name>/
```

The registry in `healthcorebench/benchmarks/registry.py` maps task keys to these directories and
adapter classes. The normal execution path does not download datasets and does not accept an
arbitrary data path.

Rules for working with data:

- Keep source datasets read-only unless the user explicitly requests a data correction.
- Do not create placeholder data to make an adapter pass.
- Report missing files and malformed records precisely.
- Do not treat third-party scripts inside a dataset directory as framework runtime code unless
  an adapter explicitly imports or executes them.
- Preserve data order unless the dataset protocol requires deterministic normalization.
- Filter invalid samples only with a deterministic, documented condition.
- A missing reference is a data/adapter issue, not a model error.
- Keep registry keys, directory numbering, configs, tests, and `benchmarks/README.md` aligned when
  adding a benchmark.

## Architecture and Data Flow

The normal pipeline is:

```text
YAML configuration
  -> Pydantic validation
  -> benchmark registry resolution
  -> source-file discovery and hashing
  -> adapter loading and sample normalization
  -> logical message construction
  -> request-time media encoding
  -> API execution and attempt recording
  -> result persistence
  -> response parsing
  -> evaluator judgments
  -> summary aggregation
  -> experiment-level JSON, CSV, and Markdown reports
```

Keep responsibilities separated:

- Adapters discover and normalize data; they do not call evaluated models.
- Message construction produces logical messages without persisting request-only media objects.
- The executor performs API calls and records attempts/results; it does not own benchmark scoring.
- Parsers extract answers without destroying raw output.
- Evaluators produce judgments without altering inference records.
- Aggregation reads persisted records and never calls the evaluated model.
- CLI code orchestrates components and should not accumulate benchmark-specific parsing logic.
- Pydantic schemas define persisted contracts and compatibility expectations.

## CLI Commands

Use the module entry point from the repository root:

```bash
python -m healthcorebench --help
```

Common commands:

```bash
python -m healthcorebench list-benchmarks
python -m healthcorebench list-benchmarks --json
python -m healthcorebench inspect-benchmark --benchmark MMLU

python -m healthcorebench run --config configs/example_text.yaml
python -m healthcorebench run --config configs/example_multimodal.yaml
python -m healthcorebench run --config <config.yaml> --run-dir <directory>

python -m healthcorebench summarize --run-dir <run-directory>
python -m healthcorebench reparse --run-dir <run-directory>
python -m healthcorebench reparse --run-dir <run-directory> --rescore-and-summarize
python -m healthcorebench rescore --run-dir <run-directory>
python -m healthcorebench rescore --run-dir <run-directory> --secondary-only
python -m healthcorebench validate-run --run-dir <run-directory>
python -m healthcorebench export-parquet --run-dir <run-directory>
python -m healthcorebench migrate-legacy --legacy-dir <old> --output-dir <new>
```

Command effects:

| Command | Calls evaluated model | Mutates run data |
|---|:---:|:---:|
| `list-benchmarks` | No | No |
| `inspect-benchmark` | No | No |
| `run` | Yes | Yes, append-oriented |
| `summarize` | No | Rewrites derived `summary.json` |
| `reparse` | No | Appends reparsed records; optional rule-based judgments/summary |
| `rescore` | No | Appends rule-based judgments and rebuilds summary; it does not rerun an LLM judge |
| `validate-run` | No | No |
| `export-parquet` | No | Writes derived Parquet files |
| `migrate-legacy` | No | Writes a new migrated run layout |

Use a small `benchmark.max_samples` or the CLI `--max-samples` override for smoke tests. Never
launch a full suite, a paid API run, or a large re-evaluation merely to validate a code edit
unless the user has requested it.

## Run Artifacts

A task run normally contains:

```text
manifest.json
samples.jsonl
attempts.jsonl
results.jsonl
judgments.jsonl
summary.json
events.jsonl
```

A multi-task experiment also produces:

```text
all_tasks_results.json
all_tasks_results.csv
all_tasks_results.md
```

Interpretation:

- `manifest.json` records run identity, configuration, source hashes, software information, and
  lifecycle status.
- `samples.jsonl` records normalized selected samples.
- `attempts.jsonl` records every real API attempt, including retries and failures.
- `results.jsonl` records final logical inference outcomes.
- `judgments.jsonl` records rule-based, LLM-judge, or human evaluations.
- `summary.json` is a derived aggregation.
- `events.jsonl` records lifecycle events.

Logical inference state is keyed by `(sample_id, sample_repeat_index)`. Judgment replacement is
resolved per `(result_id, evaluator_name)`. Records are append-only, and aggregation uses the
latest applicable records. A successful result must not be displaced by a later failure during
resume indexing.

When diagnosing a run, inspect `attempts.jsonl` as well as `results.jsonl`; the final result alone
does not show all request attempts. A missing metric is unavailable, not zero. Preserve valid
partial metrics when another evaluator or submetric is missing.

## Resume Behavior

`runtime.resume: true` scans existing records and skips completed inference. Missing successful
judgments may be backfilled without requesting the evaluated model again. Failed samples are
retried only when the configured retry-failed behavior allows it.

Resume compatibility depends on the persisted execution identity, including source data,
selected samples, output-affecting configuration, and adapter/parser/prompt behavior. Do not
bypass a resume mismatch by editing hashes or manifest fields.

Operational guidance:

- Keep the benchmark tree unchanged between an interrupted run and its resume.
- Use one writer per run directory; the runtime uses a lease to enforce this.
- Treat `runtime.resume: false` as a request for a new run directory, not as permission to append
  a clean run over an existing directory.
- Do not remove individual JSONL lines to force reruns.
- Run `validate-run` before and after any explicitly requested artifact migration.
- Back up material run artifacts before an explicitly requested manual repair.

## Debugging Guide

### Many metrics are `N/A`

Check in this order:

1. Read `summary.json` counts and `metrics_by_evaluator`.
2. Confirm that logical results have `status="success"`.
3. Check `attempts.jsonl` for transport, provider, timeout, or request errors.
4. Check `parsing_status`, `parsed_answer`, and `normalized_answer` in results.
5. Check whether a successful primary judgment exists in `judgments.jsonl`.
6. Check `evaluation_status` and evaluator error details.
7. Check `finish_reason`; length-truncated output may be intentionally unscored.
8. Confirm that tasks requiring an LLM judge have a usable judge configuration.
9. Distinguish an unavailable headline score from valid secondary or partial metrics.
10. Rebuild the summary only after confirming that source JSONL records are valid.

### A request appears stuck

Inspect the latest attempt timestamps and error fields, then check:

- request timeout and provider response time;
- retry/backoff behavior and provider `Retry-After` guidance;
- RPM/TPM limiter settings;
- configured concurrency;
- large image encoding or video frame extraction;
- judge requests, which are separate from evaluated-model requests.

Do not classify a long wait as a deadlock without checking the persisted attempts and process
state.

### A benchmark does not load

Check:

- the exact registry task key and requested split;
- `BenchmarkRegistryEntry` directory and adapter mapping;
- `discover_source_files()` and `validate_source_files()`;
- file encoding and JSON/JSONL structure;
- deterministic sample IDs and duplicate normalized identities;
- non-empty inputs and references;
- local image/video path resolution;
- optional video dependencies when applicable.

Use `inspect-benchmark` before writing ad hoc filesystem probes.

### Resume is rejected

Compare the existing manifest with the new configuration and inspect:

- source-file combined hash;
- selected sample identity;
- benchmark split and adapter version;
- generation, media, evaluator, prompt, and relevant model identity fields;
- whether the directory belongs to a different experiment or task.

Do not weaken resume validation just to reuse an incompatible directory.

### A score looks wrong

Trace one sample end to end:

1. normalized sample and reference;
2. logical/formatted prompt;
3. raw model response;
4. parsed and normalized answer;
5. evaluator name and version;
6. primary-metric marker;
7. latest judgment for that evaluator;
8. denominator policy and sample weight;
9. grouped and batch aggregation.

Primary evaluator failure must remain visible; a secondary evaluator must not silently become the
headline metric.

### A JSONL file may be damaged

Run:

```bash
python -m healthcorebench validate-run --run-dir <run-directory>
```

Normal interruption may leave a torn final line, but malformed middle records require explicit
investigation. Do not silently delete damaged records.

## Adding a Text Benchmark or Task

1. Confirm the local dataset directory, test split, source provenance, and license constraints.
2. Add or update the registry entry in `healthcorebench/benchmarks/registry.py`.
3. Implement a focused adapter under `healthcorebench/benchmarks/adapters/`.
4. Discover only files required for the registered task.
5. Normalize records into `EvaluationSample` with stable source identity and metadata.
6. Reject or deterministically filter malformed inputs; never guess a reference answer.
7. Build logical messages using shared prompt helpers where appropriate.
8. Select an existing parser and evaluator, or add a narrowly scoped implementation.
9. Add unit tests for discovery, loading, normalization, parsing, and scoring.
10. Test a real first sample read-only when local data is available.
11. Add the task to appropriate configs only after its behavior is verified.
12. Update `benchmarks/README.md` without adding dataset files to Git.

Adapters should declare behavior through their task metadata rather than special-casing the CLI or
aggregation layer.

## Adding a Multimodal Benchmark or Task

Follow the text workflow, plus these requirements:

- Preserve deterministic media order.
- Store request-only image/video sources in `runtime_media`; do not put non-serializable objects
  into persisted sample content.
- Resolve missing media as a deterministic data error.
- Encode images and sample video frames at request time.
- Make `media.max_images` compatible with the maximum final image/frame parts.
- Test single-image, multi-image, and applicable video cases.
- Use task-appropriate evaluators for classification, free text, multi-label output, structured
  document fields, multi-stage decisions, or grounding.
- Do not report probability-based metrics when real probabilities are unavailable.
- Do not report ranked grounding metrics when predictions do not contain ranked confidence.
- Treat an LLM judge as a textual assessment unless the judge request explicitly includes and
  supports the required visual evidence.

## Parsers and Evaluators

Reuse `healthcorebench/benchmarks/answer_parsing.py` and existing evaluators before introducing a
new parser or metric. New behavior must be deterministic where the benchmark has a deterministic
answer protocol.

Evaluator rules:

- Keep benchmark-native detail in `raw_score` or `parsed_judgment`.
- Use `normalized_score` for comparable aggregation, normally in `[0, 1]`.
- Use `is_correct=None` when binary correctness is not meaningful.
- Tag the intended headline judgment with `provider_metadata.primary_metric=true`.
- Secondary metrics supplement the primary metric and never replace it implicitly.
- A primary judgment error must surface as an evaluation error.
- Support alternate valid references through the sample's reference aliases rather than by
  weakening normalization globally.
- Add aggregation tests for empty, partial, error, rescore, and repeated-sample cases.

Avoid `eval()` for model output. Parse constrained JSON with `json.loads()` and validate its
shape explicitly.

## Schemas and Persistence

Files under `healthcorebench/schemas/` define durable run contracts. Schema edits require more
caution than ordinary internal refactors.

- Prefer optional fields with defaults for backward-compatible additions.
- Do not remove or rename persisted fields without an explicit migration plan.
- Preserve `schema_version` semantics.
- Keep JSON serialization finite: no `NaN` or infinity.
- Do not persist request-only media objects or encoded media bytes.
- Keep append-only attempts, results, judgments, and events.
- Use atomic writes for replaceable derived JSON artifacts.
- Ensure reparse, rescore, resume, validation, aggregation, and legacy migration still understand
  the record shape.
- Update schema tests and representative end-to-end tests together with contract changes.

## Test and Verification Matrix

Run the smallest relevant set first, then expand according to risk.

Full suite:

```bash
python -m pytest -q
```

Cache-minimizing full suite:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider --basetemp=/tmp/healthcorebench-pytest
```

Suggested minimum coverage:

| Changed area | Minimum verification |
|---|---|
| Documentation or ignore rules | `git diff --check`, link/path checks, `git status` |
| Config schema or YAML | Config-secret tests and benchmark/config registry tests |
| Client or executor | Client, executor, runtime-hardening, and mock-server tests |
| Recorder or resume | Recorder/resume, runner, validation, and E2E tests |
| Text adapter | Target adapter tests plus parser/evaluator regression tests |
| VLM adapter or media | VLM adapter, media encoder, metrics, and relevant E2E tests |
| Evaluator or aggregation | Evaluator, result-regression, batch-results, and reporting tests |
| Persisted schema | Schema consumers, tools, resume, aggregation, and E2E tests |
| CLI orchestration | CLI progress/reporting and at least one E2E path |
| Broad or cross-cutting change | Full test suite |

Useful focused commands include:

```bash
python -m pytest -q tests/test_executor.py tests/test_runtime_hardening.py
python -m pytest -q tests/test_recorder_resume.py tests/test_runner.py tests/test_tools.py
python -m pytest -q tests/test_vlm_adapters.py tests/test_media_encoder.py tests/test_vlm_metrics.py
python -m pytest -q tests/test_batch_results.py tests/test_reporting.py
python -m pytest -q tests/test_benchmark_registry_configs.py
```

Do not make a failing test pass by deleting assertions, weakening validation, or converting a real
error into `N/A`. If a test expectation is obsolete, explain and verify the intended replacement
behavior.

## Code Style

- Follow the surrounding Python style; avoid broad formatting-only diffs.
- Use type hints for public interfaces and persisted structures where practical.
- Prefer small functions with explicit inputs over hidden global state.
- Keep comments focused on invariants or non-obvious reasoning.
- Use `pathlib.Path` for filesystem paths.
- Use shared JSONL, hashing, timestamp, and reporting utilities rather than duplicating them.
- Keep user-facing terminal messages concise and consistent with existing reporting helpers.
- Preserve ASCII by default unless a file or benchmark requires another script or language.
- Do not introduce benchmark-specific branches into generic code when the behavior belongs in an
  adapter or evaluator.

## Pull Requests and Contributions

Keep each pull request focused on one coherent behavior. Before proposing a PR, run:

```bash
git status --short
git diff --check
git diff
git diff --cached
python -m pytest -q
```

### Commit Messages

Use the exact format `[type] concise description` for every commit. Keep the description
specific, imperative, and limited to the change in that commit. For example, a commit that adds a
new capability should look like:

```text
[feat] add support for the ExampleQA benchmark
```

Choose `type` from this table:

| Type | Meaning | Use for |
|---|---|---|
| `feat` | New feature | New functionality, capabilities, or dataset support |
| `fix` | Bug fix | Corrections for errors or unintended behavior |
| `docs` | Documentation | README files, guides, documentation, or comment-only changes |
| `style` | Code style | Whitespace, indentation, or formatting changes that do not alter behavior |
| `refactor` | Refactoring | Code reorganization that neither adds a feature nor fixes a bug |
| `perf` | Performance | Speed improvements, memory reductions, or other performance work |
| `test` | Tests | Adding or modifying test code |
| `build` | Build system | Dependencies, packaging, compilation, or build configuration |
| `ci` | CI/CD | GitHub Actions and other automation workflows |
| `chore` | Maintenance | Routine maintenance that does not fit another type |
| `revert` | Revert | Reverting an earlier commit |

Do not combine unrelated change types in one commit. Split them into independently understandable
commits and assign each commit its own type. After the commits are ready and verified, create the
pull request from the working branch. Do not open a pull request before the intended commits have
been created and pushed. If the task does not authorize pushing or opening a PR, prepare the
commits and report the remaining PR step instead of performing it.

The PR description should state:

- the problem and user-visible behavior;
- the implementation approach and affected modules;
- tests run and their outcomes;
- whether benchmark selection or source data changed;
- whether persisted schemas or report formats changed;
- resume and backward-compatibility implications;
- whether model or judge request behavior changed;
- whether scoring, metric meaning, or denominator policy changed;
- known limitations and unverified scenarios.

Contribution rules:

- Do not include unrelated refactors or repository-wide formatting.
- Do not commit `runs/`, benchmark datasets, caches, virtual environments, or build outputs.
- Do not modify third-party licenses or source attribution casually.
- New adapters require tests and an entry in the benchmark inventory.
- Changes to persisted behavior require regression coverage.
- Changes that can increase API requests or alter scoring must be called out explicitly.
- Use descriptive commit messages rather than generic messages such as `update` or `fix`.

## Known Operational Boundaries

- Keep external image and video files immutable during a run and its resume.
- Use a new directory for a deliberately fresh non-resume execution.
- When rate limiting is enabled, provider failures and uncertain timeout usage can affect when the
  next request is admitted.
- Use `validate-run` when JSONL integrity is uncertain.
- Direct YAML credential fields are supported configuration; do not replace them solely to impose
  a different configuration convention.
- Benchmark subtrees may contain upstream utilities with their own assumptions. They are not part
  of the framework execution path unless explicitly integrated.

## Definition of Done

A coding task is complete only when:

- the requested behavior is implemented without unrelated changes;
- affected tests pass, or unrun/failed tests are reported with a reason;
- persisted and resume behavior remains compatible or the incompatibility is explicit;
- benchmark data and run artifacts remain untouched unless they were in scope;
- no unintended files are staged;
- the final response identifies changed files, verification evidence, remaining risks, and Git
  status.
