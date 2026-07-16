<h1 align="center">PyWPEM</h1>

> [!TIP]
> **Entdecken Sie die PyWPEM-Produktseite:** [https://bin-cao.github.io/PyWPEM/](https://bin-cao.github.io/PyWPEM/)

<p align="center">
  <strong>Ein Whole-Pattern-Refinement-Framework der nächsten Generation für die Analyse komplexer Kristallstrukturen.</strong>
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_es.md">Español</a> |
  <a href="README_fr.md">Français</a> |
  <a href="README_de.md"><strong>Deutsch</strong></a>
</p>

<p align="center">
  <a href="https://bin-cao.github.io/PyWPEM/"><img src="https://img.shields.io/badge/Homepage-Project-black?style=for-the-badge" alt="Homepage" /></a>
  <a href="https://pyxplore.netlify.app/"><img src="https://img.shields.io/badge/Docs-Documentation-grey?style=for-the-badge" alt="Documentation" /></a>
  <a href="https://arxiv.org/abs/2602.16372"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" /></a>
  <a href="https://www.researchgate.net/publication/400962382"><img src="https://img.shields.io/badge/Supplementary-Materials-teal?style=for-the-badge" alt="Supplementary Materials" /></a>
  <a href="https://www.pepy.tech/projects/PyXplore"><img src="https://img.shields.io/badge/Downloads-Stats-4CAF50?style=for-the-badge" alt="Download Stats" /></a>
</p>

<p align="center">
  <a href="https://github.com/Bin-Cao/PyWPEM"><img src="https://img.shields.io/github/stars/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="GitHub Stars" /></a>
  <a href="https://github.com/Bin-Cao/PyWPEM/forks"><img src="https://img.shields.io/github/forks/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="GitHub Forks" /></a>
  <a href="https://github.com/Bin-Cao/PyWPEM/issues"><img src="https://img.shields.io/github/issues/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="Open Issues" /></a>
  <a href="https://github.com/Bin-Cao/PyWPEM/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed/Bin-Cao/PyWPEM?style=for-the-badge&logo=github" alt="Closed Issues" /></a>
  <a href="https://github.com/Bin-Cao/PyWPEM/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Bin-Cao/PyWPEM?style=for-the-badge" alt="License" /></a>
  <a href="https://github.com/Bin-Cao/PyWPEM/commits/main"><img src="https://img.shields.io/github/last-commit/Bin-Cao/PyWPEM?style=for-the-badge" alt="Last Commit" /></a>
</p>

<p align="center">
  <img width="1536" height="1024" alt="WPEM Image" src="https://github.com/user-attachments/assets/93076f8d-3c28-4b2b-b72c-82e1328d93f9" />
</p>

> [!NOTE]
> **WPEM führt ein grundlegend neues Paradigma für die XRD-Verfeinerung jenseits konventioneller Rietveld-Methoden ein.**  
> Anstatt Beugungspeaks über klassisches Least-Squares-Profilmatching anzupassen, formuliert WPEM das gesamte Beugungsmuster als physikalisch eingeschränkte probabilistische Mischverteilung und führt die Whole-Pattern-Dekomposition mit einem Expectation-Maximization-Framework durch. Durch die explizite Einbettung der Bragg-Konsistenz in den Optimierungsprozess ermöglicht PyWPEM eine stabile phasenaufgelöste Verfeinerung bei starker Peak-Überlappung, Mischphasen, amorphen Hintergründen und komplexen experimentellen Bedingungen. Diese Arbeit ist einer der ersten Versuche, KI-gestützte Strukturanalyse mit physikalisch zulässiger Beugungsverfeinerung zu vereinen, und kann die nächste Generation automatisierter XRD-Refinement-Workflows neu definieren.

**Wir haben chinesische Tutorial-Videos auf BiliBili veröffentlicht: [Link](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

Weitere Werkzeuge sind [XQueryer](https://github.com/Bin-Cao/XQueryer) für die initiale Strukturinferenz und [PRDNet](https://github.com/Bin-Cao/PRDNet) für die Vorhersage von Kristalleigenschaften.

**Ich entwickle eine Benutzeroberfläche, die alle Funktionen von PyXplore unterstützt** (siehe das [UI-Repository](https://github.com/WPEM/PyxploreUI)). Über diese UI müssen Nutzer nur ihre Daten und Parameter eingeben, und der entsprechende Ausführungscode wird automatisch generiert. Bitte verfolgen Sie die weiteren Updates.

Beiträge aus der Community sind willkommen. Mitwirkende werden im aktuellen Paper **gewürdigt**. Wesentliche Beiträge zu Kernfunktionen können in zukünftigen Veröffentlichungen zur nächsten WPEM-Version zu **Koautorenschaft** führen.

---

<p align="center">
  <a href="https://star-history.com/#Bin-Cao/PyWPEM&Date">
    <img src="https://api.star-history.com/svg?repos=Bin-Cao/PyWPEM&type=Date" width="650" alt="Star History" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 8px; background: #ffffff;" />
  </a>
</p>

---

## Überblick

**[PyXplore](https://pyxplore.netlify.app/)** ist ein modulares Python-Framework für **Röntgenbeugungs-Simulation (XRD), Dekomposition, quantitative Analyse und KI-unterstützte Strukturverfeinerung**.

Es integriert:

* Physikbasierte Beugungsmodellierung
* EM-basierte Bragg-Optimierung
* Konstruktion von Strukturgraphen
* Extinktions- und Wyckoff-Analyse
* Quantifizierung amorpher Phasen
* KI-gestützte Strukturverfeinerung

Das Toolkit ist für reproduzierbare wissenschaftliche Workflows in der Materialcharakterisierung und AI-for-Science-Forschung konzipiert.

---

## Installation

Installation über PyPI und [Einrichtung der Abhängigkeiten](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md):

```bash
pip install PyXplore
```

Upgrade auf die neueste Version:

```bash
pip install --upgrade PyXplore
```

---

## Kernfunktionen

* **XRD-Simulation**  
  Präzise Erzeugung von Beugungsmustern aus kristallographischen Informationen.

* **Peak-Dekomposition und Quantitative Analyse**  
  WPEM-basierte Dekomposition und Bestimmung von Volumenanteilen.

* **Bragg-Gesetz-Optimierung (EM-Framework)**  
  Parameterlösung auf Basis von Expectation-Maximization.

* **Extinktions- und Wyckoff-Behandlung**  
  Symmetriebezogene Vorverarbeitung und Strukturfilterung.

* **Graphbasierte Strukturdarstellung**  
  Aufbau von Kristallgraphen für nachgelagerte Machine-Learning-Aufgaben.

* **Analyse Amorphen Strukturen**  
  RDF-basierte quantitative Bewertung.

* **Multimodale Erweiterung**  
  Integrierte Module für XAS- und XPS-Analyse.

---

## Architektur

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

Das Design folgt einer **physikalisch konsistenten, modularen Architektur** und ermöglicht unabhängige oder pipelinebasierte Ausführung.

---

## Tabellen und Abbildungen

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM table" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM figure" />
</p>

---

## Wissenschaftliche Referenz

Wenn Sie **PyWPEM** in Ihrer Forschung verwenden, zitieren Sie bitte:

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

## Lizenz

Dieses Projekt wird unter der MIT License veröffentlicht.
