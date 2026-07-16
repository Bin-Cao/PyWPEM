<h1 align="center">PyWPEM</h1>

> [!TIP]
> **PyWPEM 제품 홈페이지:** [https://bin-cao.github.io/PyWPEM/](https://bin-cao.github.io/PyWPEM/)

<p align="center">
  <strong>복잡한 결정 구조 분석을 위한 차세대 전체 패턴 정련 프레임워크.</strong>
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md">日本語</a> |
  <a href="README_ko.md"><strong>한국어</strong></a> |
  <a href="README_es.md">Español</a> |
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
> **WPEM은 기존 Rietveld 방법을 넘어서는 XRD 정련의 새로운 패러다임을 제시합니다.**  
> WPEM은 전통적인 최소제곱 피크 프로파일 매칭으로 회절 피크를 맞추는 대신, 전체 회절 패턴을 물리 제약을 갖는 확률적 혼합 분포로 정식화하고 기대값-최대화 프레임워크를 통해 전체 패턴 분해를 수행합니다. 최적화 과정에 Bragg 일관성을 명시적으로 포함함으로써 PyWPEM은 심한 피크 중첩, 혼합상, 비정질 배경, 복잡한 실험 조건에서도 안정적인 상별 정련을 가능하게 합니다. 이 작업은 AI 기반 구조 분석과 물리적으로 허용 가능한 회절 정련을 통합하려는 초기 시도 중 하나이며, 차세대 자동 XRD 정련 워크플로를 재정의할 잠재력이 있습니다.

**중국어 튜토리얼 영상을 BiliBili에 공개했습니다: [Link](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

관련 도구로는 초기 구조 추론을 위한 [XQueryer](https://github.com/Bin-Cao/XQueryer)와 결정 물성 예측을 위한 [PRDNet](https://github.com/Bin-Cao/PRDNet)이 있습니다.

**PyXplore의 모든 기능을 지원하는 사용자 인터페이스를 개발 중입니다**([UI repository](https://github.com/WPEM/PyxploreUI) 참조). 이 UI에서는 사용자가 데이터와 매개변수를 입력하면 해당 실행 코드가 자동으로 생성됩니다. 향후 업데이트를 기대해 주세요.

커뮤니티의 기여를 환영합니다. 기여자는 현재 논문에서 **감사의 글에 포함**됩니다. 핵심 기능에 대한 중요한 기여는 향후 WPEM 다음 버전 논문에서 **공동 저자**로 이어질 수 있습니다.

---

<p align="center">
  <a href="https://star-history.com/#Bin-Cao/PyWPEM&Date">
    <img src="https://api.star-history.com/svg?repos=Bin-Cao/PyWPEM&type=Date" width="650" alt="Star History" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 8px; background: #ffffff;" />
  </a>
</p>

---

## 개요

**[PyXplore](https://pyxplore.netlify.app/)** 는 **X선 회절(XRD) 시뮬레이션, 분해, 정량 분석, AI 보조 구조 정련**을 위한 모듈형 Python 프레임워크입니다.

다음 기능을 통합합니다.

* 물리 기반 회절 모델링
* EM 기반 Bragg 최적화
* 구조 그래프 구성
* 소멸 규칙 및 Wyckoff 분석
* 비정질 상 정량화
* AI 기반 구조 정련

이 도구는 재료 특성 분석과 AI for Science 연구에서 재현 가능한 과학 워크플로를 지원하도록 설계되었습니다.

---

## 설치

PyPI에서 설치하고 [의존성 설치 안내](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md)를 확인하세요.

```bash
pip install PyXplore
```

최신 버전으로 업그레이드합니다.

```bash
pip install --upgrade PyXplore
```

---

## 주요 기능

* **XRD 시뮬레이션**  
  결정학 정보를 기반으로 정확한 회절 패턴을 생성합니다.

* **피크 분해 및 정량 분석**  
  WPEM 기반 분해와 부피 분율 산정을 수행합니다.

* **Bragg 법칙 최적화(EM 프레임워크)**  
  기대값-최대화 기반 매개변수 풀이를 제공합니다.

* **소멸 규칙 및 Wyckoff 처리**  
  대칭성을 고려한 전처리와 구조 필터링을 지원합니다.

* **그래프 기반 구조 표현**  
  후속 머신러닝 작업을 위한 결정 그래프를 구성합니다.

* **비정질 구조 분석**  
  RDF 기반 정량 평가를 제공합니다.

* **멀티모달 확장**  
  XAS 및 XPS 분석 모듈을 통합합니다.

---

## 아키텍처 개요

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

이 설계는 **물리적으로 일관된 모듈형 아키텍처**를 따르며, 독립 실행과 파이프라인 실행을 모두 지원합니다.

---

## 표와 그림

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM table" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM figure" />
</p>

---

## 학술 인용

연구에서 **PyWPEM**을 사용한다면 다음을 인용해 주세요.

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

## 라이선스

이 프로젝트는 MIT License로 배포됩니다.
