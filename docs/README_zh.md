<h1 align="center">PyWPEM</h1>

> [!TIP]
> **访问 PyWPEM 产品主页：**[https://bin-cao.github.io/PyWPEM/](https://bin-cao.github.io/PyWPEM/)

<p align="center">
  <strong>面向复杂晶体结构分析的新一代全谱精修框架。</strong>
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md"><strong>简体中文</strong></a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_es.md">Español</a> |
  <a href="README_fr.md">Français</a> |
  <a href="README_de.md">Deutsch</a>
</p>

<p align="center">
  <a href="https://bin-cao.github.io/PyWPEM/">
    <img src="https://img.shields.io/badge/Homepage-项目-black?style=for-the-badge" alt="项目主页" />
  </a>
  <a href="https://pyxplore.netlify.app/">
    <img src="https://img.shields.io/badge/Docs-文档-grey?style=for-the-badge" alt="文档" />
  </a>
  <a href="https://arxiv.org/abs/2602.16372">
    <img src="https://img.shields.io/badge/arXiv-论文-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv 论文" />
  </a>
  <a href="https://www.researchgate.net/publication/400962382">
    <img src="https://img.shields.io/badge/Supplementary-补充材料-teal?style=for-the-badge" alt="补充材料" />
  </a>
  <a href="https://www.pepy.tech/projects/PyXplore">
    <img src="https://img.shields.io/badge/Downloads-下载统计-4CAF50?style=for-the-badge" alt="下载统计" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/Bin-Cao/PyWPEM">
    <img src="https://img.shields.io/github/stars/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="GitHub Stars" />
  </a>
  <a href="https://github.com/Bin-Cao/PyWPEM/forks">
    <img src="https://img.shields.io/github/forks/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="GitHub Forks" />
  </a>
  <a href="https://github.com/Bin-Cao/PyWPEM/issues">
    <img src="https://img.shields.io/github/issues/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="Open Issues" />
  </a>
  <a href="https://github.com/Bin-Cao/PyWPEM/issues?q=is%3Aissue+is%3Aclosed">
    <img src="https://img.shields.io/github/issues-closed/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="Closed Issues" />
  </a>
  <a href="https://github.com/Bin-Cao/PyWPEM/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Bin-Cao/PyWPEM?style=for-the-badge" alt="License" />
  </a>
  <a href="https://github.com/Bin-Cao/PyWPEM/commits/main">
    <img src="https://img.shields.io/github/last-commit/Bin-Cao/PyWPEM?style=for-the-badge" alt="Last Commit" />
  </a>
</p>

> [!NOTE]
> **WPEM 提出了一种不同于传统 Rietveld 方法的 XRD 精修新范式。**  
> WPEM 不再通过传统最小二乘峰形匹配来拟合衍射峰，而是将完整衍射谱表示为受物理约束的概率混合分布，并通过期望最大化框架执行全谱分解。通过在优化过程中显式嵌入 Bragg 一致性，PyWPEM 可以在严重峰重叠、混合相、非晶背景和复杂实验条件下实现稳定的分相精修。本工作是将 AI 驱动结构分析与物理可容许衍射精修统一起来的早期尝试之一，有望重新定义下一代自动化 XRD 精修流程。

**我们已在哔哩哔哩发布中文教程视频：[链接](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

其他工具包括用于初始结构推断的 [XQueryer](https://github.com/Bin-Cao/XQueryer)，以及用于晶体性质预测的 [PRDNet](https://github.com/Bin-Cao/PRDNet)。

**我正在开发一个用户界面以支持 PyXplore 的全部功能**（参见 [UI 仓库](https://github.com/WPEM/PyxploreUI)）。通过该界面，用户只需输入数据与参数，即可自动生成对应的执行代码。敬请关注后续更新。

我们欢迎社区贡献。贡献者将在当前论文中**获得致谢**。对关键功能的重要贡献可能在后续 WPEM 版本论文中获得**共同作者**资格。

---

<p align="center">
  <a href="https://star-history.com/#Bin-Cao/PyWPEM&Date">
    <img
      src="https://api.star-history.com/svg?repos=Bin-Cao/PyWPEM&type=Date"
      width="650"
      alt="Star History"
      style="border: 1px solid #d0d7de; border-radius: 12px; padding: 8px; background: #ffffff;"
    />
  </a>
</p>

---

## 概述

**[PyXplore](https://pyxplore.netlify.app/)** 是一个模块化 Python 框架，用于 **X 射线衍射（XRD）模拟、分解、定量分析以及 AI 辅助结构精修**。

它集成了：

* 基于物理的衍射建模
* 基于 EM 的 Bragg 优化
* 结构图构建
* 消光与 Wyckoff 分析
* 非晶相定量分析
* AI 驱动结构精修

该工具包旨在支持材料表征与 AI for Science 研究中的可复现科学工作流。

---

## 安装

从 PyPI 安装，并参考[依赖安装说明](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md)：

```bash
pip install PyXplore
```

升级到最新版本：

```bash
pip install --upgrade PyXplore
```

---

## 核心功能

* **XRD 模拟**  
  基于晶体学信息生成精确的衍射图谱。

* **峰分解与定量分析**  
  基于 WPEM 的分解与体积分数计算。

* **Bragg 定律优化（EM 框架）**  
  基于期望最大化算法的参数求解。

* **消光与 Wyckoff 处理**  
  支持对称性约束下的预处理与结构筛选。

* **基于图的结构表示**  
  构建晶体图结构，用于下游机器学习任务。

* **非晶结构分析**  
  基于 RDF 的定量评估。

* **多模态扩展**  
  集成 XAS 与 XPS 分析模块。

---

## 架构概览

```text
PyWPEM/
├── WPEM.py
├── XRDSimulation/
├── EMBraggOpt/
├── Refinement/
├── StructureOpt/
├── GraphStructure/
├── Extinction/
├── Amorphous/
├── Background/
├── Plot/
├── DecomposePlot/
├── WPEMXAS/
├── WPEMXPS/
└── refs/
```

该设计遵循**物理一致的模块化架构**，支持独立执行或流水线式组合。

---

## 图表展示

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM 表格" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM 图示" />
</p>

---

## 科学引用

如果您在研究中使用 **PyWPEM**，请引用：

```bibtex
@article{cao2026wpem,
  title={AI-Driven Structure Refinement of X-ray Diffraction},
  author={Bin Cao, Qian Zhang, Zhenjie Feng, Taolue Zhang, Jiaqiang Huang, Lu-Tao Weng, Tong-Yi Zhang},
  journal={arXiv preprint},
  year={2026},
  url={https://arxiv.org/abs/2602.16372v1}
}
```

---

## 许可证

本项目基于 MIT License 发布。
