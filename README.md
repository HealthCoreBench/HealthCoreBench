English | [中文](README_zh.md)

![](./assets/healthcorebench.png)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=https%3A%2F%2Fgithub.com%2FFreedomIntelligence&left_text=visitors)
[![Project Page](https://img.shields.io/badge/Project-Website-2ea44f)](https://pku-yuangroup.github.io/Helios-Page)
[![arXiv](https://img.shields.io/badge/Arxiv-2606.07962-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.07962)
[![hf_space](https://img.shields.io/badge/🤗-HealthCoreBench-blue.svg)](https://huggingface.co/datasets/Kohsin/ChronoPhyBench)

[![][github-release-shield]][github-release-link]
[![][github-contributors-shield]][github-contributors-link]
[![][github-forks-shield]][github-forks-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-link]

**HealthCoreBench** aims to advance medical evaluation through *two complementary directions*: building a unified evaluation infrastructure and curating a compact, reliable benchmark for measuring medical intelligence.

* **Unified medical evaluation infrastructure.** HealthCoreBench provides a comprehensive evaluation suite covering **71 language tasks and 36 multimodal tasks**, together with a unified open-source evaluation framework that enables efficient and standardized assessment of medical capabilities across diverse models.

* **Compact and reliable benchmark curation.** HealthCoreBench identifies a high-quality subset from a broad and fragmented benchmark pool, rather than simply reducing benchmark size or selecting questions with the lowest model accuracy. The resulting benchmark is **compact, diverse, challenging, and reliable**, ensuring that model failures more meaningfully reflect limitations in medical intelligence.

![](https://github.com/WangRongsheng/CareGPT/blob/main/assets/images/hx.png?raw=true)

## 💡 News

* **`July 20, 2026`** 🤗

## ⚙️ Quick Start

A complete list of all supported evaluation benchmarks can be found [here](benchmarks/README.md). You can use our project in two ways:

1. **Use the datasets independently:** Download the curated medical evaluation datasets and implement your own evaluation pipeline.
2. **Use the HealthCoreBench framework:** Directly run standardized medical capability evaluations through the unified HealthCoreBench evaluation framework.

#### Option 1: Use the datasets independently

<details>
<summary>Click to expand</summary>

You can download all evaluation datasets directly from [Hugging Face (link 1)](https://huggingface.co/datasets/FreedomIntelligence/HealthCoreBench), [Hugging Face (link 2)](https://huggingface.co/datasets/wangrongsheng/HealthCoreBench) or download any individual dataset separately.

Accelerated Downloads in China:

```bash
# download hfd
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

# set mirror
# for linux/mac
export HF_ENDPOINT=https://hf-mirror.com
# for win
# $env:HF_ENDPOINT = "https://hf-mirror.com"

# download data
./hfd.sh FreedomIntelligence/HealthCoreBench --dataset
# ./hfd.sh wangrongsheng/HealthCoreBench --dataset
```

</details>

#### Option 2: Use the HealthCoreBench framework

<details>
<summary>Click to expand</summary>

HealthCoreBench is a **resumable, auditable** evaluation framework for medical LLMs and VLMs. It communicates with models through an OpenAI-compatible API, with first-class support for vLLM and compatibility with services implementing OpenAI Chat Completions. The framework does not load or deploy models itself: serve the model out of process and point HealthCoreBench to its `base_url`.

##### Core principles

* **Inference, parsing, scoring, and aggregation are decoupled.** Each is a separate stage with its own versioned records.
* **Raw responses are never overwritten by parsing.** Re-parsing appends a new record.
* **Failed API calls are never scored as wrong answers.** They are recorded as errors and excluded from the score denominator, with the policy recorded in the summary.
* **Every aggregate is recomputable from per-sample JSONL.** `summary.json` is derived rather than authoritative.
* **Unknown provider data is `null`, never guessed**, including token counts and model versions.
* **Stable `sample_id` values** align the same question across models and runs.
* **Runs are resume-safe.** You can interrupt a run without losing or repeating completed work.
* **Credentials can come from YAML or environment variables** but are never written to run artifacts. Header values, provider-specific request bodies, and URL user information are redacted as well.

##### Installation

Python **3.11 or later** is required. Install the project dependencies with:

```bash
pip install -r requirements.txt
```

##### Run an evaluation

Serve a model, for example with vLLM:

```bash
vllm serve /models/Qwen3-32B --port 8000 --served-model-name Qwen3-32B
export OPENAI_API_KEY=EMPTY   # vLLM ignores the key; any non-empty value works
```

Run an evaluation:

```bash
python -m healthcorebench run --config configs/example_text.yaml

# Or override fields on the CLI:
python -m healthcorebench run --config configs/example_text.yaml \
    --benchmark MMLU --model Qwen3-32B \
    --base-url http://127.0.0.1:8000/v1 --concurrency 32
```

##### Commands

| Command | Purpose |
|---|---|
| `run` | Run one benchmark against one model using configuration. |
| `summarize` | Rebuild `summary.json` from `results.jsonl` and `judgments.jsonl`. |
| `reparse` | Re-extract answers from stored raw responses without model calls. |
| `rescore` | Re-run rule-based evaluators over stored results without evaluated-model calls. |
| `validate-run` | Check the integrity of a run directory. |
| `export-parquet` | Convert JSONL logs to Parquet. |
| `list-benchmarks` | List registered benchmarks and whether a parser is implemented. |
| `inspect-benchmark` | Show a benchmark's fixed directory, files, hashes, and sample count. |
| `migrate-legacy` | Convert old-format results into the current layout. |

##### Benchmark data

Benchmark data is read **only** from the fixed in-project trees `benchmarks/medical_llm_benchmarks/<N>_<Name>/` and `benchmarks/medical_vlm_benchmarks/<N>_<Name>/`, resolved through the registry in `healthcorebench/benchmarks/registry.py`. There is no network download during evaluation and no arbitrary `--data-path`. The manifest records the files actually read, their SHA256 hashes, and a deterministic combined hash called the *effective benchmark revision*, allowing two runs to be checked for identical data.

The benchmark catalog covers **71 language benchmarks and 36 multimodal benchmarks**. Because some benchmarks expose multiple evaluation subsets, the registry currently provides **98 language tasks and 56 multimodal tasks**. Existing `--benchmark ALL` remains language-only; use `ALL_VLM` for multimodal tasks or `ALL_BENCHMARKS` for both. Run the following command to inspect task keys and implementation status:

```bash
python -m healthcorebench list-benchmarks
```

##### Output layout

```text
runs/<experiment_id>/<benchmark>/<task>/
  manifest.json      run config, environment, versions, model identity, source-file hashes, status
  samples.jsonl      normalized selected samples
  attempts.jsonl     one line per real API attempt, including retries and failures
  results.jsonl      one line per final logical response: raw, parsed, normalized, and reference
  judgments.jsonl    one line per scoring or judge result
  summary.json       recomputed from results and judgments
  events.jsonl       run lifecycle events
```

When `run` expands to one or more benchmark tasks, it writes three experiment-level exports beside the task directories:

```text
all_tasks_results.json   complete rows, including metrics_by_evaluator
all_tasks_results.csv    spreadsheet-friendly core columns
all_tasks_results.md     Markdown tables grouped by metric profile
```

The Markdown tables are not printed to the terminal. Tasks with the same metric columns share one table; its first column is the benchmark or task key, and the remaining columns contain only that metric profile, such as Accuracy, Judge, EM/Token-F1, ROUGE, or BLEU. At the end of a run, the terminal prints only the absolute Markdown path and the batch run directory. These paths, task progress, and other human-facing run messages are emitted in green English text.

The detailed JSON and CSV files retain one row per benchmark or task, including status, counts, primary score, and all applicable native metrics. A missing value means that the metric was unavailable, not that its score was zero.

Each run has a `run_id` in the form `YYYYMMDDTHHMMSSZ_<short_uuid>`, recorded inside its artifacts. Resuming a directory requires matching configuration, source data, adapter, parser, prompt versions, and selected samples; this prevents different code versions from mixing results.

##### Configuration

See `configs/example_text.yaml` and `configs/example_multimodal.yaml`. Key points:

* `model.api_key_env` and `evaluation.judge.api_key_env` name environment variables holding credentials. A direct `api_key` is also accepted and takes precedence when both are set. Direct keys are excluded from configuration dumps, hashes, manifests, and logs. They remain in the YAML file and are resolved only in memory for requests.
* `configs/example_text.yaml` supports both native-metric tasks and open tasks. Its judge block is used automatically only when the selected benchmark requires an LLM judge.
* Token usage is recorded directly from provider responses. The framework does not calculate monetary prices, which vary independently of benchmark results.
* One run represents one model, one benchmark, and one split. An external orchestrator can launch multiple runs for batch matrices.
* Use `configs/run_all_benchmarks_text.yaml` for the established full text suite and `configs/run_all_benchmarks_multimodal.yaml` for the explicit list of all implemented VLM tasks.

##### Tests

```bash
python -m pytest tests/ -q
```

Tests use an in-process mock OpenAI-compatible client and a standard-library mock HTTP server; no network access or GPU is required.

</details>

## 🌍 Who Uses It?

## 📝 Citation

If you find this work helpful, please consider to **star🌟 this repo** or **cite us**. Thanks for your support!

```bib

```

If you have any suggestions or questions, please feel free to [submit an issue](), [open a pull request (PR)](), or contact us via 📮 email: `rongshengwang@link.cuhk.edu.cn`.

## 👍 Acknowledgement

We thank the contributors of [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit), [Medmarks](https://github.com/MedARC-AI/Medmarks), [Awesome-AI4Med](https://github.com/FreedomIntelligence/Awesome-AI4Med), and [Awesome-Medical-Agents](https://github.com/zhcz328/Awesome-Medical-Agents) for their valuable contributions to the medical AI community. These open-source projects provide important foundations for medical benchmark evaluation, resource organization, and the development of medical AI systems. Their efforts have greatly facilitated reproducible research and inspired the construction of HealthCoreBench.

> [!NOTE]
> HealthCoreBench serves as a practical extension of [Awesome-AI4Med](https://github.com/FreedomIntelligence/Awesome-AI4Med), transforming its broad resource curation into a curated, executable benchmark evaluation framework for medical AI systems. This is just the beginning of this journey, and we will keep expanding it in the future!

## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/OpenS2V-Nexus/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/OpenS2V-Nexus&anon=true" />
</a>

[github-contributors-link]: https://github.com/open-compass/opencompass/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/opencompass?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/opencompass/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/opencompass?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/opencompass/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/opencompass?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/opencompass/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/opencompass?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/open-compass/opencompass/releases
[github-release-shield]: https://img.shields.io/github/v/release/open-compass/opencompass?color=369eff&labelColor=black&logo=github&style=flat-square
[github-releasedate-link]: https://github.com/open-compass/opencompass/releases
[github-releasedate-shield]: https://img.shields.io/github/release-date/open-compass/opencompass?labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/opencompass/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/opencompass?color=ffcb47&labelColor=black&style=flat-square
[github-trending-shield]: https://trendshift.io/api/badge/repositories/6630
[github-trending-url]: https://trendshift.io/repositories/6630
