<h1 align="center">PyWPEM</h1>

<p align="center">
  <strong>Un framework de refinamiento de patrón completo de próxima generación para el análisis de estructuras cristalinas complejas.</strong>
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_es.md"><strong>Español</strong></a> |
  <a href="README_fr.md">Français</a> |
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
> **WPEM introduce un nuevo paradigma para el refinamiento XRD más allá de los métodos Rietveld convencionales.**  
> En lugar de ajustar picos de difracción mediante el ajuste tradicional de perfiles por mínimos cuadrados, WPEM formula todo el patrón de difracción como una distribución de mezcla probabilística con restricciones físicas y realiza la descomposición de patrón completo mediante un framework de expectativa-maximización. Al incorporar explícitamente la consistencia de Bragg en la optimización, PyWPEM permite un refinamiento estable por fases bajo solapamiento severo de picos, fases mixtas, fondos amorfos y condiciones experimentales complejas. Este trabajo representa uno de los primeros intentos de unificar el análisis estructural impulsado por IA con el refinamiento de difracción físicamente admisible, con potencial para redefinir los flujos de trabajo automatizados de refinamiento XRD de próxima generación.

**Hemos publicado los videos tutoriales en chino en BiliBili: [Link](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

Otras herramientas incluyen [XQueryer](https://github.com/Bin-Cao/XQueryer) para inferencia estructural inicial y [PRDNet](https://github.com/Bin-Cao/PRDNet) para predicción de propiedades cristalinas.

**Estoy desarrollando una interfaz de usuario para soportar todas las funcionalidades de PyXplore** (ver el [repositorio de la UI](https://github.com/WPEM/PyxploreUI)). Con esta interfaz, los usuarios solo tendrán que introducir sus datos y parámetros, y el código de ejecución correspondiente se generará automáticamente. Mantente atento a las próximas actualizaciones.

Damos la bienvenida a las contribuciones de la comunidad. Los colaboradores serán **reconocidos** en el artículo actual. Las contribuciones sustanciales a funcionalidades clave pueden conducir a **coautoría** en futuras publicaciones de la próxima versión de WPEM.

---

<p align="center">
  <a href="https://star-history.com/#Bin-Cao/PyWPEM&Date">
    <img src="https://api.star-history.com/svg?repos=Bin-Cao/PyWPEM&type=Date" width="650" alt="Star History" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 8px; background: #ffffff;" />
  </a>
</p>

---

## Descripción General

**[PyXplore](https://pyxplore.netlify.app/)** es un framework modular de Python para **simulación de difracción de rayos X (XRD), descomposición, análisis cuantitativo y refinamiento estructural asistido por IA**.

Integra:

* Modelado de difracción basado en física
* Optimización de Bragg basada en EM
* Construcción de grafos estructurales
* Análisis de extinción y Wyckoff
* Cuantificación de fases amorfas
* Refinamiento estructural impulsado por IA

El toolkit está diseñado para flujos de trabajo científicos reproducibles en caracterización de materiales e investigación de AI for Science.

---

## Instalación

Instala desde PyPI y [configura las dependencias](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md):

```bash
pip install PyXplore
```

Actualiza a la última versión:

```bash
pip install --upgrade PyXplore
```

---

## Funciones Principales

* **Simulación XRD**  
  Generación precisa de patrones de difracción a partir de información cristalográfica.

* **Descomposición de Picos y Análisis Cuantitativo**  
  Descomposición basada en WPEM y determinación de fracciones de volumen.

* **Optimización de la Ley de Bragg (Framework EM)**  
  Resolución de parámetros basada en expectativa-maximización.

* **Extinción y Tratamiento Wyckoff**  
  Preprocesamiento y filtrado estructural conscientes de la simetría.

* **Representación Estructural Basada en Grafos**  
  Construcción de grafos cristalinos para tareas posteriores de aprendizaje automático.

* **Análisis de Estructuras Amorfas**  
  Evaluación cuantitativa basada en RDF.

* **Extensión Multimodal**  
  Módulos integrados para análisis XAS y XPS.

---

## Arquitectura

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

El diseño sigue una **arquitectura modular y físicamente consistente**, que permite ejecución independiente o basada en pipelines.

---

## Tablas y Figuras

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM table" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM figure" />
</p>

---

## Referencia Científica

Si usas **PyWPEM** en tu investigación, cita:

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

## Licencia

Este proyecto se publica bajo la MIT License.
