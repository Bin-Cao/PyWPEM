<h1 align="center">PyWPEM</h1>

<p align="center">
  <strong>Un framework de raffinement par motif complet de nouvelle génération pour l'analyse de structures cristallines complexes.</strong>
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_es.md">Español</a> |
  <a href="README_fr.md"><strong>Français</strong></a> |
  <a href="README_de.md">Deutsch</a>
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
> **WPEM introduit un nouveau paradigme pour le raffinement XRD au-delà des méthodes Rietveld conventionnelles.**  
> Au lieu d'ajuster les pics de diffraction par appariement traditionnel de profils aux moindres carrés, WPEM formule l'ensemble du motif de diffraction comme une distribution de mélange probabiliste contrainte par la physique et réalise une décomposition du motif complet au moyen d'un framework d'espérance-maximisation. En intégrant explicitement la cohérence de Bragg dans le processus d'optimisation, PyWPEM permet un raffinement stable par phase en présence de fort chevauchement de pics, de phases mixtes, de fonds amorphes et de conditions expérimentales complexes. Ce travail représente l'une des premières tentatives d'unifier l'analyse structurale pilotée par l'IA avec un raffinement de diffraction physiquement admissible, avec le potentiel de redéfinir les workflows automatisés de raffinement XRD de prochaine génération.

**Nous avons publié les vidéos de tutoriel en chinois sur BiliBili : [Link](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

Les autres outils incluent [XQueryer](https://github.com/Bin-Cao/XQueryer) pour l'inférence de structure initiale et [PRDNet](https://github.com/Bin-Cao/PRDNet) pour la prédiction des propriétés cristallines.

**Je développe une interface utilisateur pour prendre en charge toutes les fonctionnalités de PyXplore** (voir le [dépôt UI](https://github.com/WPEM/PyxploreUI)). Grâce à cette interface, les utilisateurs n'auront qu'à saisir leurs données et paramètres, et le code d'exécution correspondant sera généré automatiquement. Restez attentifs aux prochaines mises à jour.

Les contributions de la communauté sont les bienvenues. Les contributeurs seront **remerciés** dans l'article actuel. Des contributions substantielles aux fonctionnalités clés peuvent conduire à une **co-signature** dans de futures publications de la prochaine version de WPEM.

---

<p align="center">
  <a href="https://star-history.com/#Bin-Cao/PyWPEM&Date">
    <img src="https://api.star-history.com/svg?repos=Bin-Cao/PyWPEM&type=Date" width="650" alt="Star History" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 8px; background: #ffffff;" />
  </a>
</p>

---

## Vue d'ensemble

**[PyXplore](https://pyxplore.netlify.app/)** est un framework Python modulaire pour la **simulation de diffraction des rayons X (XRD), la décomposition, l'analyse quantitative et le raffinement structural assisté par IA**.

Il intègre :

* Modélisation de diffraction fondée sur la physique
* Optimisation de Bragg basée sur EM
* Construction de graphes de structure
* Analyse des extinctions et de Wyckoff
* Quantification des phases amorphes
* Raffinement structural piloté par l'IA

La boîte à outils est conçue pour des workflows scientifiques reproductibles en caractérisation des matériaux et en recherche AI for Science.

---

## Installation

Installez depuis PyPI et [configurez les dépendances](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md) :

```bash
pip install PyXplore
```

Mettez à jour vers la dernière version :

```bash
pip install --upgrade PyXplore
```

---

## Fonctionnalités Clés

* **Simulation XRD**  
  Génération précise de motifs de diffraction à partir d'informations cristallographiques.

* **Décomposition de Pics et Analyse Quantitative**  
  Décomposition basée sur WPEM et détermination des fractions volumiques.

* **Optimisation de la Loi de Bragg (Framework EM)**  
  Résolution de paramètres basée sur l'espérance-maximisation.

* **Gestion des Extinctions et de Wyckoff**  
  Prétraitement et filtrage structural tenant compte de la symétrie.

* **Représentation Structurale Basée sur les Graphes**  
  Construction de graphes cristallins pour les tâches d'apprentissage automatique en aval.

* **Analyse des Structures Amorphes**  
  Évaluation quantitative basée sur RDF.

* **Extension Multimodale**  
  Modules intégrés pour l'analyse XAS et XPS.

---

## Architecture

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

La conception suit une **architecture modulaire et cohérente avec la physique**, permettant une exécution indépendante ou sous forme de pipeline.

---

## Tableaux et Figures

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM table" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM figure" />
</p>

---

## Référence Scientifique

Si vous utilisez **PyWPEM** dans vos recherches, veuillez citer :

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

## Licence

Ce projet est publié sous MIT License.
