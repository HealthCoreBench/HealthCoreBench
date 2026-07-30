[English](README.md) | 中文

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

**HealthCoreBench** 旨在从*两个互补方向*推动医学评测发展：构建统一的评测基础设施，以及打造用于衡量医学智能的紧凑、可靠 benchmark。

* **统一的医学评测基础设施。** HealthCoreBench 提供覆盖 **71 个语言任务和 36 个多模态任务**的综合评测套件，以及统一的开源评测框架，从而能够高效、标准化地评估不同模型的医学能力。

* **紧凑且可靠的 benchmark 筛选。** HealthCoreBench 从广泛而分散的 benchmark 池中识别高质量子集，而不是简单缩减 benchmark 规模，或只选择模型准确率最低的问题。最终得到的 benchmark **紧凑、多样、具有挑战性且可靠**，使模型失败能够更有意义地反映其医学智能局限。

![](https://github.com/WangRongsheng/CareGPT/blob/main/assets/images/hx.png?raw=true)

## 💡 最新消息

* **`2026 年 7 月 20 日`** 🤗

## ⚙️ 快速开始

所有已支持评测 benchmark 的完整列表见[此处](benchmarks/README.md)。你可以通过两种方式使用本项目：

1. **独立使用数据集：** 下载经过筛选的医学评测数据集，并实现自己的评测流程。
2. **使用 HealthCoreBench 框架：** 通过统一的 HealthCoreBench 评测框架，直接运行标准化的医学能力评测。

#### 方式一：独立使用数据集

<details>
<summary>点击展开</summary>

你可以直接从 [Hugging Face（链接 1）](https://huggingface.co/datasets/FreedomIntelligence/HealthCoreBench)、[Hugging Face（链接 2）](https://huggingface.co/datasets/wangrongsheng/HealthCoreBench) 下载全部评测数据集，也可以分别下载任意单个数据集。

中国大陆加速下载：

```bash
# 下载 hfd
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

# 设置镜像
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows
# $env:HF_ENDPOINT = "https://hf-mirror.com"

# 下载数据
./hfd.sh FreedomIntelligence/HealthCoreBench --dataset
# ./hfd.sh wangrongsheng/HealthCoreBench --dataset
```

</details>

#### 方式二：使用 HealthCoreBench 框架

<details>
<summary>点击展开</summary>

HealthCoreBench 是一个面向医学 LLM 和 VLM 的**可断点续跑、可审计**评测框架。它通过 OpenAI-compatible API 与模型通信，优先支持 vLLM，同时兼容实现 OpenAI Chat Completions 的服务。框架本身不加载或部署模型：请在框架进程外启动模型服务，并将 HealthCoreBench 指向其 `base_url`。

##### 核心原则

* **推理、解析、评分和聚合相互解耦。** 每个阶段彼此独立，并拥有各自的版本化记录。
* **解析永远不会覆盖原始响应。** 重新解析会追加一条新记录。
* **失败的 API 调用永远不会被计为错误答案。** 它们会被记录为错误并排除在分数分母之外，相关策略会记录在汇总结果中。
* **所有聚合结果都可以根据逐样本 JSONL 重新计算。** `summary.json` 是派生结果，而不是权威数据源。
* **未知的服务提供方数据使用 `null`，绝不猜测**，包括 token 数量和模型版本。
* **稳定的 `sample_id`** 用于对齐不同模型和不同运行中的同一道问题。
* **运行支持安全续跑。** 你可以中断运行，而不会丢失或重复已经完成的工作。
* **凭据可以来自 YAML 或环境变量**，但绝不会写入运行产物。请求头字段、服务提供方特定的请求体以及 URL 用户信息同样会被脱敏。

##### 安装

需要 Python **3.11 或更高版本**。使用以下命令安装项目依赖：

```bash
pip install -r requirements.txt
```

##### 运行评测

首先启动模型服务，例如使用 vLLM：

```bash
vllm serve /models/Qwen3-32B --port 8000 --served-model-name Qwen3-32B
export OPENAI_API_KEY=EMPTY   # vLLM 会忽略该密钥；任意非空值均可
```

运行评测：

```bash
python -m healthcorebench run --config configs/example_text.yaml

# 也可以通过 CLI 覆盖配置字段：
python -m healthcorebench run --config configs/example_text.yaml \
    --benchmark MMLU --model Qwen3-32B \
    --base-url http://127.0.0.1:8000/v1 --concurrency 32
```

被评估模型的请求并发数由 YAML 中的 `runtime.concurrency` 控制，也可以用
`--concurrency` 临时覆盖。默认值以及仓库自带的所有配置均为 `1`，此时严格串行请求；
设置为 `2` 或更大的整数时，最多允许相同数量的独立 API 请求同时进行。具体数值应根据
模型服务吞吐和限流能力调整。LLM judge 使用独立的
`evaluation.judge.concurrency` 配置，不受该字段控制。

##### 命令

| 命令 | 用途 |
|---|---|
| `run` | 使用配置针对一个模型运行一个 benchmark。 |
| `summarize` | 根据 `results.jsonl` 和 `judgments.jsonl` 重新生成 `summary.json`。 |
| `reparse` | 从已存储的原始响应中重新提取答案，不调用模型。 |
| `rescore` | 对已存储结果重新运行规则评估器，不调用被评测模型。 |
| `validate-run` | 检查运行目录的完整性。 |
| `export-parquet` | 将 JSONL 日志转换为 Parquet。 |
| `list-benchmarks` | 列出已注册的 benchmark，以及是否已实现相应 parser。 |
| `inspect-benchmark` | 显示 benchmark 的固定目录、文件、哈希值和样本数量。 |
| `migrate-legacy` | 将旧格式结果转换为当前布局。 |

##### Benchmark 数据

Benchmark 数据**只会**从项目内固定目录 `benchmarks/medical_llm_benchmarks/<N>_<Name>/` 和 `benchmarks/medical_vlm_benchmarks/<N>_<Name>/` 读取，并通过 `healthcorebench/benchmarks/registry.py` 中的注册表进行解析。评测过程中不会通过网络下载数据，也不支持任意的 `--data-path`。manifest 会记录实际读取的文件、这些文件的 SHA256 哈希，以及一个称为*有效 benchmark 版本*的确定性组合哈希，从而可以确认两次运行是否使用了完全相同的数据。

Benchmark 目录覆盖 **71 个语言 benchmark 和 36 个多模态 benchmark**。由于部分 benchmark 包含多个评测子集，当前注册表共提供 **98 个语言任务和 56 个多模态任务**。现有的 `--benchmark ALL` 仍然只包含语言任务；多模态任务请使用 `ALL_VLM`，全部任务请使用 `ALL_BENCHMARKS`。运行以下命令可以查看任务 key 和实现状态：

```bash
python -m healthcorebench list-benchmarks
```

##### 输出布局

```text
runs/<experiment_id>/<benchmark>/<task>/
  manifest.json      运行配置、环境、版本、模型身份、源文件哈希和状态
  samples.jsonl      标准化后的已选样本
  attempts.jsonl     每次真实 API 请求一行，包括重试和失败
  results.jsonl      每个最终逻辑响应一行：原始、解析、标准化结果和参考答案
  judgments.jsonl    每个评分结果或 judge 结果一行
  summary.json       根据结果和评分重新计算的汇总
  events.jsonl       运行生命周期事件
```

当 `run` 展开为一个或多个 benchmark 任务时，会在任务目录旁写入三个实验级导出文件：

```text
all_tasks_results.json   完整结果行，包括 metrics_by_evaluator
all_tasks_results.csv    适合电子表格查看的核心列
all_tasks_results.md     按指标结构分组的 Markdown 表格
```

Markdown 表格不会直接打印到终端。拥有相同指标列的任务会共用一张表；第一列是 benchmark 或任务 key，其余列只包含该指标结构，例如 Accuracy、Judge、EM/Token-F1、ROUGE 或 BLEU。运行结束时，终端只会打印 Markdown 文件的绝对路径和批量运行目录。上述路径、任务进度及其他面向用户的运行消息会以绿色英文文本输出。

详细的 JSON 和 CSV 文件会为每个 benchmark 或任务保留一行，其中包括状态、数量、主要分数和所有适用的原生指标。缺失值表示该指标不可用，并不代表得分为零。

每次运行都有一个格式为 `YYYYMMDDTHHMMSSZ_<short_uuid>` 的 `run_id`，并记录在运行产物中。续跑目录要求配置、源数据、adapter、parser、prompt 版本和所选样本全部匹配，从而避免不同代码版本的结果混合在一起。

##### 配置

配置示例见 `configs/example_text.yaml` 和 `configs/example_multimodal.yaml`。要点如下：

* `model.api_key_env` 和 `evaluation.judge.api_key_env` 用于指定保存凭据的环境变量名称。也可以直接提供 `api_key`；当两者同时存在时，直接提供的密钥优先。直接提供的密钥会从配置转储、哈希、manifest 和日志中排除。它们保留在 YAML 文件中，仅在发送请求时于内存中解析。
* `configs/example_text.yaml` 同时支持原生指标任务和开放式任务。只有当所选 benchmark 需要 LLM judge 时，才会自动使用其中的 judge 配置块。
* Token 使用量直接记录服务提供方返回的结果。框架不计算费用，因为价格会独立于 benchmark 结果发生变化。
* 一次运行代表一个模型、一个 benchmark 和一个 split。批量矩阵可以由外部编排器启动多次运行来完成。
* 已建立的完整文本套件使用 `configs/run_all_benchmarks_text.yaml`；所有已实现 VLM 任务的明确列表使用 `configs/run_all_benchmarks_multimodal.yaml`。

##### 测试

```bash
python -m pytest tests/ -q
```

测试使用进程内模拟 OpenAI-compatible 客户端和 Python 标准库模拟 HTTP 服务器；不需要网络访问或 GPU。

</details>

## 🔍 关键洞察和结果

## 📖 Benchmark 支持

HealthCoreBench 当前支持 71 个医学语言 benchmark 和 36 个医学多模态 benchmark。每个 benchmark 的说明、论文、下载链接、年份和样本数量请参阅[完整 benchmark 目录](benchmarks/README.md)。

<table align="center" width="100%">
  <tbody>
    <tr align="center" valign="bottom">
      <td colspan="2" width="50%">
        <b>医学 LLM Benchmarks（71）</b>
      </td>
      <td colspan="2" width="50%">
        <b>医学 VLM Benchmarks（36）</b>
      </td>
    </tr>
    <tr valign="top">
      <td width="25%">

- [MMLU (Medical)](benchmarks/medical_llm_benchmarks/1_MMLU/)
- [PubMedQA](benchmarks/medical_llm_benchmarks/2_PubMedQA/)
- [MedQA-USMLE](benchmarks/medical_llm_benchmarks/69_MedQA-USMLE/)
- [MedQA-MCMLE](benchmarks/medical_llm_benchmarks/70_MedQA-MCMLE/)
- [MedMCQA](benchmarks/medical_llm_benchmarks/3_MedMCQA/)
- [Medbullets](benchmarks/medical_llm_benchmarks/4_Medbullets/)

</td>
<td width="25%">

- [MedXpertQA (Text)](benchmarks/medical_llm_benchmarks/5_MedXpertQA_Text/)
- [SuperGPQA (Medicine & Biology)](benchmarks/medical_llm_benchmarks/6_SuperGPQA/)
- [CMB](benchmarks/medical_llm_benchmarks/8_CMB/)
- [HLE (Medicine, Text)](benchmarks/medical_llm_benchmarks/11_HLE_med/)
- [MedR-Bench](benchmarks/medical_llm_benchmarks/27_MedR-Bench/)
- ...

</td>
<td width="25%">

- [VQA-RAD](benchmarks/medical_vlm_benchmarks/1_VQA-RAD/)
- [SLAKE](benchmarks/medical_vlm_benchmarks/2_SLAKE/)
- [PathVQA](benchmarks/medical_vlm_benchmarks/3_PathVQA/)
- [PMC-VQA](benchmarks/medical_vlm_benchmarks/4_PMC-VQA/)
- [OmniMedVQA](benchmarks/medical_vlm_benchmarks/5_OmniMedVQA/)
- [MedXpertQA](benchmarks/medical_vlm_benchmarks/6_MedXpertQA/)
- [IU-Xray](benchmarks/medical_vlm_benchmarks/9_IU-Xray/)
- [MedFrameQA](benchmarks/medical_vlm_benchmarks/11_MedFrameQA/)

</td>
<td width="25%">

- [Quilt-VQA](benchmarks/medical_vlm_benchmarks/21_Quilt-VQA/)
- [PathMMU](benchmarks/medical_vlm_benchmarks/22_PathMMU/)
- [MMMU (Health & Medicine)](benchmarks/medical_vlm_benchmarks/23_MMMU-Health-Medicine/)
- [MIMIC-Ext-MIMIC-CXR-VQA](benchmarks/medical_vlm_benchmarks/30_MIMIC-Ext-MIMIC-CXR-VQA/)
- [HLE (Medicine, MM)](benchmarks/medical_vlm_benchmarks/32_HLM/)
- [GMAI-MMBench (Val)](benchmarks/medical_vlm_benchmarks/36_GMAI-MMBench/)
- [LiveClin](benchmarks/medical_vlm_benchmarks/34_LiveClin/)
- ...

</td>

</tr>
  </tbody>
</table>

## 🗂️ 仓库结构

```text
HealthCoreBench/
├── assets/                         README 图片和项目视觉资源
├── benchmarks/                     框架使用的 benchmark 数据集
│   ├── README.md                   完整 benchmark 目录和元数据
│   ├── medical_llm_benchmarks/     71 个文本医学 benchmark
│   │   ├── 1_MMLU/
│   │   │   ├── anatomy_test.json
│   │   │   ├── clinical_knowledge_test.json
│   │   │   └── ...                 每个已选医学学科对应一个文件
│   │   ├── 2_PubMedQA/
│   │   ├── 3_MedMCQA/
│   │   ├── ...
│   │   └── 71_GlobalDentBench/
│   └── medical_vlm_benchmarks/     36 个多模态医学 benchmark
│       ├── 1_VQA-RAD/
│       │   ├── images/             样本引用的图像资源
│       │   ├── vqa_rad_test.json   评测标注
│       │   └── vqa_rad_test.parquet
│       ├── 2_SLAKE/
│       ├── 3_PathVQA/
│       ├── ...
│       └── 36_GMAI-MMBench/
├── configs/                        示例和完整套件 YAML 配置
├── healthcorebench/                评测框架源代码包
│   ├── benchmarks/                 注册表以及文本和 VLM 数据 adapter
│   ├── clients/                    OpenAI-compatible 模型客户端
│   ├── evaluators/                 规则指标和 LLM judge 支持
│   ├── media/                      图像和多模态输入处理
│   ├── runtime/                    执行、记录、续跑和报告
│   ├── schemas/                    持久化数据模型
│   ├── tools/                      汇总、重新解析、重新评分、校验和导出
│   ├── aggregation/                任务和实验结果聚合
│   ├── utils/                      共享工具
│   ├── cli.py                      命令行接口
│   └── config.py                   配置加载和校验
├── tests/                          单元、回归、集成和 E2E 测试
├── runs/                           生成的运行产物，不属于源数据
├── AGENTS.md                       面向编程助手的使用说明
├── CODE_OF_CONDUCT.md              贡献者行为准则
├── pyproject.toml                  包和工具配置
├── requirements.txt                Python 依赖
└── README.md                       项目概览和使用指南
```

Benchmark 目录会保留各自来源数据集的特定布局，因此不同数据集的具体文件名可能不同。
框架通过 `healthcorebench/benchmarks/` 中的 adapter 读取这些数据；请勿将这些 adapter
模块与 `benchmarks/` 下存储的 benchmark 数据混淆。每个已支持 benchmark 及其源文件
请参阅 [benchmark 目录](benchmarks/README.md)。

## 🌍 谁在使用？

## 📝 引用

如果这项工作对你有所帮助，请考虑为本仓库点亮 **Star🌟** 或**引用我们的工作**。感谢你的支持！

```bib

```

如有任何建议或问题，欢迎[提交 Issue]()、[发起 Pull Request（PR）]()，或通过 📮 邮箱联系我们：`rongshengwang@link.cuhk.edu.cn`。

## 👍 致谢

感谢 [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit)、[Medmarks](https://github.com/MedARC-AI/Medmarks)、[Awesome-AI4Med](https://github.com/FreedomIntelligence/Awesome-AI4Med) 和 [Awesome-Medical-Agents](https://github.com/zhcz328/Awesome-Medical-Agents) 的贡献者为医学 AI 社区作出的宝贵贡献。这些开源项目为医学 benchmark 评测、资源组织和医学 AI 系统开发提供了重要基础，极大推动了可复现研究，并为 HealthCoreBench 的构建带来了启发。

> [!NOTE]
> HealthCoreBench 是 [Awesome-AI4Med](https://github.com/FreedomIntelligence/Awesome-AI4Med) 的实践延伸，将其广泛的资源整理转化为一个经过筛选、可直接执行的医学 AI benchmark 评测框架。这只是我们旅程的起点，未来还会持续扩展！

## 🤝 贡献者

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
