[English](README.md) | [中文](README_zh.md)

![](./assets/healthcorebench.png)

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

## ⚙️ Quick Start

You can use our project in two ways:

1. **Use the datasets independently:** Download the curated medical evaluation datasets and implement your own evaluation pipeline.
2. **Use the HealthCoreBench framework:** Directly run standardized medical capability evaluations through the unified HealthCoreBench evaluation framework.

### Option 1: Use the datasets independently

<details>
<summary>Click to expand</summary>

You can download all evaluation datasets directly from [Hugging Face]() or download any individual dataset separately.

Accelerated Downloads in China:
```python
# download hfd
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

# for linux/mac
export HF_ENDPOINT=https://hf-mirror.com
# for win
#$env:HF_ENDPOINT = "https://hf-mirror.com"

# download data
./hfd.sh FreedomIntelligence/HealthCoreBench --dataset
```

</details>

### Option 2: Use the HealthCoreBench framework

## 🌍 Who Uses It?

## 📝 Citation

If you find HealthCoreBench useful, please cite us:
```bib

```

If you have any suggestions or questions, please feel free to submit an issue, open a pull request (PR), or contact us via 📮 email: `rongshengwang@link.cuhk.edu.cn`.

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
