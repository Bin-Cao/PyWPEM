<h1 align="center">PyWPEM</h1>

<p align="center">
  <strong>複雑な結晶構造解析のための次世代全パターン精密化フレームワーク。</strong>
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md">简体中文</a> |
  <a href="README_ja.md"><strong>日本語</strong></a> |
  <a href="README_ko.md">한국어</a> |
  <a href="README_es.md">Español</a> |
  <a href="README_fr.md">Français</a> |
  <a href="README_de.md">Deutsch</a>
</p>

<p align="center">
  <a href="https://bin-cao.github.io/PyWPEM/">
    <img src="https://img.shields.io/badge/Homepage-プロジェクト-black?style=for-the-badge" alt="Homepage" />
  </a>
  <a href="https://pyxplore.netlify.app/">
    <img src="https://img.shields.io/badge/Docs-ドキュメント-grey?style=for-the-badge" alt="Documentation" />
  </a>
  <a href="https://arxiv.org/abs/2602.16372">
    <img src="https://img.shields.io/badge/arXiv-論文-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" />
  </a>
  <a href="https://www.researchgate.net/publication/400962382">
    <img src="https://img.shields.io/badge/Supplementary-資料-teal?style=for-the-badge" alt="Supplementary Materials" />
  </a>
  <a href="https://www.pepy.tech/projects/PyXplore">
    <img src="https://img.shields.io/badge/Downloads-統計-4CAF50?style=for-the-badge" alt="Download Stats" />
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
> **WPEM は、従来の Rietveld 法を超える XRD 精密化の新しいパラダイムを提示します。**  
> 従来の最小二乗ピークプロファイル照合で回折ピークをフィットするのではなく、WPEM は回折パターン全体を物理制約付きの確率混合分布として定式化し、期待値最大化フレームワークによって全パターン分解を行います。最適化過程に Bragg 整合性を明示的に組み込むことで、PyWPEM は深刻なピーク重なり、混合相、アモルファス背景、複雑な実験条件下でも安定した相分解精密化を可能にします。本研究は、AI 駆動構造解析と物理的に許容される回折精密化を統合する初期の試みの一つであり、次世代の自動 XRD 精密化ワークフローを再定義する可能性があります。

**BiliBili で中国語チュートリアル動画を公開しました：[リンク](https://www.bilibili.com/video/BV1xfRVBQEFv/?spm_id_from=333.337.search-card.all.click&vd_source=6b9872e6d30ffcbac3baf8965e05bab4)**

関連ツールとして、初期構造推定のための [XQueryer](https://github.com/Bin-Cao/XQueryer) と、結晶物性予測のための [PRDNet](https://github.com/Bin-Cao/PRDNet) があります。

**現在、PyXplore の全機能をサポートするユーザーインターフェースを開発中です**（[UI リポジトリ](https://github.com/WPEM/PyxploreUI) を参照）。この UI では、ユーザーがデータとパラメータを入力するだけで、対応する実行コードが自動生成されます。今後の更新にご期待ください。

コミュニティからの貢献を歓迎します。貢献者は現在の論文で**謝辞に記載**されます。主要機能への大きな貢献は、将来の WPEM 次期バージョンの論文で**共著者**となる可能性があります。

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

## 概要

**[PyXplore](https://pyxplore.netlify.app/)** は、**X 線回折（XRD）のシミュレーション、分解、定量解析、および AI 支援構造精密化**のためのモジュール型 Python フレームワークです。

以下を統合しています。

* 物理ベースの回折モデリング
* EM ベースの Bragg 最適化
* 構造グラフの構築
* 消滅則および Wyckoff 解析
* アモルファス相の定量評価
* AI 駆動の構造精密化

本ツールキットは、材料解析および AI for Science 研究における再現可能な科学ワークフローのために設計されています。

---

## インストール

PyPI からインストールし、[依存関係を設定](https://github.com/Bin-Cao/PyWPEM/blob/main/INSTALL.md)してください。

```bash
pip install PyXplore
```

最新版へアップグレードします。

```bash
pip install --upgrade PyXplore
```

---

## 主な機能

* **XRD シミュレーション**  
  結晶学情報に基づく正確な回折パターン生成。

* **ピーク分解と定量解析**  
  WPEM に基づく分解と体積分率の決定。

* **Bragg 則最適化（EM フレームワーク）**  
  期待値最大化法に基づくパラメータ求解。

* **消滅則と Wyckoff 処理**  
  対称性を考慮した前処理と構造フィルタリング。

* **グラフベース構造表現**  
  下流の機械学習タスクに向けた結晶グラフ構築。

* **アモルファス構造解析**  
  RDF に基づく定量評価。

* **マルチモーダル拡張**  
  XAS および XPS 解析モジュールを統合。

---

## アーキテクチャ概要

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

本設計は、**物理整合的なモジュールアーキテクチャ**に従っており、単独実行とパイプライン実行の両方に対応します。

---

## 表と図

<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/da5bd320-3651-4223-b862-06fb5ce1f96a" alt="PyWPEM table" />
</p>

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/50b1aacc-6a4f-4b58-95fb-a4094da60055" alt="PyWPEM figure" />
</p>

---

## 参考文献

研究で **PyWPEM** を使用する場合は、以下を引用してください。

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

## ライセンス

本プロジェクトは MIT License の下で公開されています。
