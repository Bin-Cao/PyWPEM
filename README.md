<h1 align="center">PyWPEM</h1>

<p align="center">
  <strong>A next-generation whole-pattern refinement framework for complex crystal structure analysis.</strong>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> |
  <a href="docs/README_zh.md">简体中文</a> |
  <a href="docs/README_ja.md">日本語</a> |
  <a href="docs/README_ko.md">한국어</a> |
  <a href="docs/README_es.md">Español</a> |
  <a href="docs/README_fr.md">Français</a> |
  <a href="docs/README_de.md">Deutsch</a>
</p>

<p align="center">
  <a href="https://bin-cao.github.io/PyWPEM/">
    <img src="https://img.shields.io/badge/Homepage-Project-black?style=for-the-badge" alt="Homepage" />
  </a>
  <a href="https://pyxplore.netlify.app/">
    <img src="https://img.shields.io/badge/Docs-Documentation-grey?style=for-the-badge" alt="Documentation" />
  </a>
  <a href="https://arxiv.org/abs/2602.16372">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" />
  </a>
  <a href="https://www.researchgate.net/publication/400962382">
    <img src="https://img.shields.io/badge/Supplementary-Materials-teal?style=for-the-badge" alt="Supplementary Materials" />
  </a>
  <a href="https://www.pepy.tech/projects/PyXplore">
    <img src="https://img.shields.io/badge/Downloads-Stats-4CAF50?style=for-the-badge" alt="Download Stats" />
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

<p align="center">
  <img width="1536" height="1024" alt="WPEM Image" src="https://github.com/user-attachments/assets/93076f8d-3c28-4b2b-b72c-82e1328d93f9" />
</p>

> [!NOTE]
> **WPEM introduces a fundamentally new paradigm for XRD refinement beyond conventional Rietveld methods.**  
> Instead of fitting diffraction peaks through traditional least-squares profile matching, WPEM formulates the entire diffraction pattern as a physics-constrained probabilistic mixture distribution and performs whole-pattern decomposition through an expectation-maximization framework. By explicitly embedding Bragg consistency into the optimization process, PyWPEM enables stable phase-resolved refinement under severe peak overlap, mixed phases, amorphous backgrounds, and complex experimental conditions. This work represents one of the first attempts to unify AI-driven structure analysis with physically admissible diffraction refinement, potentially redefining the next generation of automated XRD refinement workflows.

**We have released the tutorial videos in Chinese on BiliBili: [Link](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

Other tools include [XQueryer](https://github.com/Bin-Cao/XQueryer) for initial structure inference and [PRDNet](https://github.com/Bin-Cao/PRDNet) for crystal property prediction.

**I am developing a user interface to support all functionalities of PyXplore** (see the [UI repository](https://github.com/WPEM/PyxploreUI)). Through the UI, users only need to input their data and parameters, and the corresponding execution code is generated automatically. Please stay tuned for updates.

We welcome contributions from the community. Contributors will **be acknowledged** in the current paper. Substantial contributions to key functionalities may lead to **co-authorship** in future publications for the next version of WPEM.

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

## Overview

**[PyXplore](https://pyxplore.netlify.app/)** is a modular Python framework for **X-ray diffraction (XRD) simulation, decomposition, quantitative analysis, and AI-assisted structure refinement**.

It integrates:

* Physics-based diffraction modeling
* EM-based Bragg optimization
* Structure graph construction
* Extinction and Wyckoff analysis
* Amorphous phase quantification
* AI-driven structural refinement

The toolkit is designed for reproducible scientific workflows in materials characterization and AI for Science research.

---

## Installation

Install from PyPI and [install the dependencies](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md):

```bash
pip install PyXplore
```

Upgrade to the latest version:

```bash
pip install --upgrade PyXplore
```

---

## Key Features

* **XRD Simulation**  
  Accurate diffraction pattern generation from crystallographic information.

* **Peak Decomposition & Quantitative Analysis**  
  WPEM-based decomposition and volume fraction determination.

* **Bragg Law Optimization (EM Framework)**  
  Expectation-Maximization-based parameter solving.

* **Extinction & Wyckoff Handling**  
  Symmetry-aware preprocessing and structural filtering.

* **Graph-Based Structure Representation**  
  Crystal graph construction for downstream machine learning tasks.

* **Amorphous Structure Analysis**  
  RDF-based quantitative evaluation.

* **Multi-modal Extension**  
  Integrated modules for XAS and XPS analysis.

---

## Architecture Overview

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

The design follows a **physics-consistent, modular architecture**, enabling independent or pipeline-based execution.

---

## Tables & Figures

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM table" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM figure" />
</p>

---

## Scientific Reference

If you use **PyWPEM** in your research, please cite:

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

## License

This project is released under the MIT License.
