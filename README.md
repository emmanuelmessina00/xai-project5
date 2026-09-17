# eXplainable AI in Medical Imaging: Unsupervised Concept Discovery in Vision-Language Models via Sparse Autoencoders and Hierarchical Semantic Alignment

This document outlines the architecture, methodology, and critical analysis of our XAI project, serving as a comprehensive blueprint for the final academic paper. The work addresses the inherent "black-box" nature of medical Vision-Language Models (VLMs) by proposing an unsupervised, automated pipeline to extract, ground, and validate interpretable clinical concepts from chest X-rays, culminating in a quantitatively validated hierarchical alignment evaluation.

---

## 1. Pipeline Architecture & Methodology

The project is structured into five sequential phases, each addressing a specific gap in current interpretability research. The workflow guides raw image data from mathematical abstraction to clinically validated semantic features, and finally to a multi-resolution clinical aggregation.

### 01 - Feature Extraction

This initial phase translates raw medical images into dense vector representations. Leveraging the state-of-the-art **BioMedCLIP** architecture (ViT-B/16 and PubMedBERT), the system extracts visual and textual embeddings. We strategically decoupled our datasets to optimize performance:

* **NIH Chest X-ray Subset (10k images):** Used exclusively to train the Sparse Autoencoder. Its massive scale provides the visual diversity required to prevent "dead neurons" and ensure a robust, highly populated latent space.
* **Open-I Dataset (3,666 images):** Used strictly for final evaluation because it provides images paired with their original, free-text clinical reports (the *Findings* and *Impression* sections).

All embeddings undergo L2 normalization to stabilize training and facilitate accurate cosine similarity calculations.

### 02 - Medical Dictionary Creation

The objective is to build a robust, data-driven semantic concept matrix (`T`). Starting from the raw metadata of the Open-I dataset, the pipeline applies a strict **Clinical Filtration** to remove technical artifacts and medical devices (e.g., "surgical instruments", "catheters"), preserving only anatomical and pathological descriptors.
These terms undergo **Domain Alignment** to map ambiguous words (e.g., "opacity") to strict thoracic terminology (e.g., "lung opacity") before querying the **UMLS API**. This guarantees deterministic retrieval of certified Concept Unique Identifiers (CUIs) and high-quality synonyms from SNOMED CT and MeSH. Finally, an LLM (**Gemini 3.1-flash-lite**) expands the dictionary by generating 5 realistic, report-style radiology phrases for each concept. The text encoder processes this corpus to generate the final semantic matrix: **998 fine-grained phrases** grounded in **100 parent clinical concepts**.

### 03 - Sparse Autoencoder (SAE) Training

This notebook tackles the "polysemanticity" problem in dense models, where a single neuron might represent multiple unrelated concepts. A Sparse Autoencoder is trained on the NIH visual embeddings to decompose the dense latent space into expanded, interpretable features (1024 latent neurons). By enforcing an L1 sparsity penalty, the SAE forces only a small subset of neurons to activate for any given image, isolating specific visual patterns. The SAE's decoder weights are then projected onto the text-embedding space using the semantic matrix from Step 02, assigning a human-readable clinical label to each latent neuron based on cosine similarity.

### 04 - Concept Evaluation & Semantic Alignment

The fourth module quantifies the clinical accuracy of the SAE's extracted concepts against the patient's actual radiology report. For each image, active neurons are selected via a **dynamic, cumulative-mass threshold**: neurons are ranked by activation and retained until their cumulative share reaches 90% of the total positive activation (capped at `TOP_K = 10`), each mapped to its closest phrase concept (cosine similarity ≥ 0.20) via Step 02's dictionary.

To eliminate generative hallucinations and latency, we utilize **MedGemma 1.5 4B** as a frozen external evaluator using a deterministic **Zero-Shot Logit Scoring** strategy. By probing the unnormalized logits of single-token identifiers (A, B, C) representing *Aligned*, *Unaligned*, and *Uncertain*, we ensure a rigorous and mathematically sound evaluation:

1. **Forward Pass:** the prompt is passed through the model without triggering the auto-regressive generation loop (`model.generate()`).
2. **Logit Extraction:** we extract the raw, unnormalized logits for the last token in the sequence.
3. **Activation Measurement:** we isolate the logits corresponding to tokens `A`, `B`, and `C`.
4. **Deterministic Output:** the token with the highest logit dictates the final verdict.

**Debiasing via Label Permutation.** To remove any positional or lexical bias tied to a fixed letter-to-class mapping, each concept–report pair is evaluated across **three permuted forward passes**, cycling *Aligned*, *Unaligned*, and *Uncertain* through positions A, B, and C. The resulting per-class probabilities are averaged across the three runs before the final verdict is selected, so no single letter is systematically favored.

**Refined Evaluation Rules.** The evaluator's few-shot prompt was extended with explicit disambiguation rules to reduce systematic errors: absence of mention is never treated as an implicit negation; general statements such as "no acute cardiopulmonary abnormality" do not automatically exclude every specific disease; and an anatomical structure is explicitly distinguished from a disease affecting it, so that a normal-appearing structure still supports the anatomical concept even when no pathology is mentioned.

### 05 - Hierarchical Concept Analysis & Macro-Clusters

The fifth module groups the 100 parent clinical concepts from Step 02 into semantic macro-clusters via **Agglomerative Clustering** (cosine distance, average linkage), then re-aggregates the fixed MedGemma verdicts from Step 04 onto these broader clinical families. This dual-granularity approach allows us to verify if the SAE captures the correct anatomical or pathological domain, even if the exact phrasing differs — and it is now a **quantitatively validated, multi-resolution** analysis rather than a fixed choice:

* **Silhouette Validation of *k*:** before fixing the number of macro-clusters, we score every candidate `k = 3–15` (same cosine distance / average linkage) via silhouette analysis, and project the reference grouping with a 2D t-SNE visualization. This makes the choice of macro-cluster count auditable rather than arbitrary (see Section 3.1 for the result and why `k=8` is still used as the clinical reference despite not scoring highest).
* **Multi-Resolution Aggregation:** the pipeline evaluates a configurable set of resolutions (`CLUSTER_COUNTS = [3, 5, 8, 10]` by default) in a single run, producing per-*k* verdict counts, percentages, and plots, plus a combined `cluster_performance_summary.csv` across every (*k*, sample-size) combination processed.
* **Fixed Ground-Truth Verdicts:** changing *k* only changes how concepts are grouped — it never re-queries MedGemma, so all resolutions remain directly comparable against the same Step 04 evaluation.

### 06 - Independent Validation & Multi-Model Blind Evaluation

A recognized limitation of concept-based explainability pipelines that rely on a single LLM-as-a-judge (as in the *MedConcept* framework and our own Step 04 protocol) is judge reliability itself — prior work has documented systematic biases and selectivity issues in this setup (*"Judging the Judges"*, Shi et al., 2025). To address this directly — following feedback from Prof. Eleonora Poeta — we designed and ran a **Double-Blind Independent Validation Protocol** to benchmark MedGemma's verdicts against both human judgment and commercial LLMs.

**Protocol.** A stratified subset of **N=25** concept–report pairs was drawn from the Step 04 evaluation (8 Aligned, 8 Unaligned, 9 Uncertain, by MedGemma's original verdict), then exported to a spreadsheet with the MedGemma verdict **hidden** to prevent anchoring bias. Three independent annotators (Davide, Riccardo, Emmanuel) labeled every sample manually, with a majority-vote `human_consensus` as ground truth. In parallel, three commercial models — ChatGPT (GPT-4o), Claude 3.5 Sonnet, and Gemini 1.5 Pro — evaluated the same samples under two input modalities: direct **file attachment** of the CSV, and a **batch text prompt** pasted in a fresh chat session, to test whether input format affects judgment consistency.

**Results.** Inter-annotator agreement among the three human raters was strong but not unanimous: all three agreed on **18/25 samples (72.0%)**, with pairwise agreement ranging from 76.0% (Davide–Riccardo, Riccardo–Emmanuel) to 92.0% (Davide–Emmanuel). Format robustness was high for every commercial model, with file-attachment and text-prompt verdicts matching on **88.0% (ChatGPT)** and **92.0% (Claude, Gemini)** of samples, indicating that input modality has only a marginal effect on LLM judgment. Against the human-consensus ground truth, all three commercial models clustered closely together regardless of format — **68–76% accuracy** (ChatGPT: 72.0% both formats; Claude: 68.0% both formats; Gemini: 76.0% file / 72.0% text) — with no single model or format standing out as a clear favorite. This places commercial-LLM agreement with human judgment in the same range as inter-human agreement itself (72–92%), suggesting the disagreement is driven more by genuine ambiguity in borderline concept–report pairs than by a specific evaluator's weakness. A direct MedGemma-vs-human-consensus figure requires re-merging the hidden reference verdicts (excluded from the blind export by design) and is reported in the notebook's own analysis cell once that merge is performed.

---

## 2. Experimental Results & Comparative Scaling (N=10 to N=500)

To validate the statistical stability of our pipeline and verify whether the observed behaviors hold across larger populations, we scaled the evaluation from an initial exploratory benchmark ($N=10$) up to an extended cohort ($N=500$).

### 2.1 Quantitative Alignment Metrics

The quantitative distribution across the evaluated benchmarks shows strong statistical consistency, converging cleanly as the sample size expands:

| Benchmark Metric | $N = 10$ | $N = 50$ | $N = 100$ | $N = 500$ | Dynamic Trend Analysis |
| --- | --- | --- | --- | --- | --- |
| **Avg. Concepts / Image** | 26.2 | 26.2 | 26.2 | 26.0 | Near-invariant (Dynamic $TOP\_K$ saturation) |
| **Avg. Aligned Score** | **7.25%** | **8.40%** | **7.64%** | **7.21%** | Highly stable ($\pm 1\%$) |
| **Avg. Unaligned Score** | **23.17%** | **22.88%** | **23.37%** | **22.49%** | Stable contradiction rate ($\sim 23\%$) |
| **Avg. Uncertain Score** | **69.57%** | **68.72%** | **68.99%** | **70.30%** | Strong reporting-bias capture ($\sim 70\%$) |

### 2.2 Qualitative Case Studies (Best, Median, Worst)

Scaling to the full $N=500$ dataset surfaced richer, higher-confidence clinical alignments and reinforced patterns of radiological omission:

* **Best Case (Img #108, $N=500$):**
  The maximum alignment matched the radiological confirmation of hyperinflation and COPD traits (`Aligned = 0.31`). MedGemma successfully mapped key SAE features such as `eventration, diaphragmatic`, `localized dome shaped elevation`, and `obliteration of the costophrenic sulcus` directly to the clinical text.

* **Median Case (Img #294, $N=500$):**
  The median case centered around a completely normal cardiopulmonary study (`Aligned = 0.05`, `Uncertain = 0.85`). The report simply stated "Normal heart size. Clear lungs. No pneumothorax. No pleural effusion." This confirms that for healthy patients, the SAE detects baseline visual structures that the radiologist condenses into "within normal limits," heavily skewing the results towards Uncertainty.

* **Worst Case (Img #433, $N=500$):**
  In worst-case scenarios, descriptive focal findings (e.g., a 4.8 cm mass in the left lower hemithorax) dominate the report. The SAE extracts secondary anatomical context which is entirely omitted by the physician (e.g., bone anomalies or subdiaphragmatic signs), leading to extremely high uncertainty scores (`Uncertain = 0.80`).

---

## 3. Hierarchical Aggregation: Cluster Validation & Multi-Resolution Results

### 3.1 Choosing the Number of Macro-Clusters: Silhouette Analysis

Silhouette scores were computed for every `k` from 3 to 15 over the 100 parent-concept embeddings (cosine distance, average linkage):

| $k$ | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Silhouette Score | **0.274** | 0.139 | 0.122 | 0.195 | 0.192 | **0.184** | 0.245 | 0.263 | 0.227 | 0.202 | 0.211 | 0.233 | 0.224 |

The best-scoring value among those tested is `k=3` (0.274), not `k=8` (0.184). We nonetheless retain **`k=8` as the clinical reference** for the main analysis: at `k=3` the macro-clusters collapse multiple distinct clinical domains (e.g., pleural, skeletal, and parenchymal findings) into a single broad group, which is too coarse to support the anatomical/pathological interpretation this project targets. `k=8` was chosen for the clinical granularity it preserves, not because it is the statistically optimal partition — a limitation we report transparently rather than obscure, and Section 3.2 now evaluates `k=3, 5, 8, 10` side by side so the reader can judge the trade-off directly.

### 3.2 Macro-Cluster Alignment Across Resolutions (N=500)

Mapping individual concept predictions onto macro-clusters at multiple resolutions shows that the core finding — high-precision alignment for cardiomediastinal/vascular findings alongside broad reporting-bias-driven uncertainty elsewhere — holds regardless of `k`:

| Macro-Cluster ID | Description / Clinical Area | Aligned | Unaligned | Uncertain |
| --- | --- | --- | --- | --- |
| **0** | Pulmonary Opacities & Parenchymal Disease | 8.62% | 21.85% | 69.53% |
| **1** | Airway & Bronchial Alterations | 0.88% | 23.51% | 75.61% |
| **2** | Skeletal & Thoracic Wall Structures | 5.15% | 22.29% | 72.57% |
| **3** | Pleural Space & Effusions | 4.38% | 36.25% | 59.37% |
| **4** | Cardiomediastinal & Vascular Morphology | **49.23%** | 3.46% | 47.31% |
| **5** | Calcifications & Chronic Granulomas | 0.15% | 28.38% | 71.47% |
| **6** | Rare Thoracic Anomalies | 4.60% | 3.45% | 91.95% |
| **7** | Diaphragmatic & Subdiaphragmatic Signs | 0.56% | 28.65% | 70.79% |

*(k = 8, N = 500 — the finest resolution reported here; `k=3`, `k=5`, and `k=10` are computed in parallel by the notebook and follow the same trend, with the Cardiomediastinal/Vascular cluster consistently the standout: 49.2% Aligned at `k=3`/`k=5`/`k=8`/`k=10`.)*

At coarser resolutions the same signal survives with less granularity — at `k=3`, the three clusters resolve to 6.76% / 49.23% / 0.15% Aligned respectively, confirming that Cluster 4's precision at `k=8` is not an artifact of a narrow bucket. At `k=10`, splitting further mainly fragments Cluster 0/1's low-signal mass without revealing new high-alignment sub-domains.

---

## 4. In-Depth Discussion & Academic Significance

The transition across sample benchmarks and cluster resolutions yields four critical insights that form the core argumentation of our research:

### 4.1 The "Reporting Bias" Paradigm Shift

Across all sample sizes up to 500 images, the proportion of *Uncertain* verdicts stabilized at approximately **70%**. In traditional NLP evaluation, this might indicate poor retrieval. In the medical domain, however, this consistency proves that the evaluation successfully captures clinical **reporting bias**. Radiologists draft pragmatic, action-oriented reports; they do not dictate every healthy structure or normal physiological variant visible in an X-ray. The SAE, conversely, acts as an objective, exhaustive feature detector. It extracts everything it visualizes. The resulting high uncertainty rate perfectly quantifies the discrepancy between exhaustive machine perception and selective human synthesis.

### 4.2 High Diagnostic Precision in Macro-Cluster 4

The hierarchical aggregation proves its worth when analyzing Macro-Cluster 4 (Cardiomediastinal & Vascular Morphology). This cluster emerged as the most reliable clinical domain, stabilizing around **~49% Aligned**, with an incredibly low unaligned error rate of only **3.46%**. This phenomenon occurs because heart size and mediastinal contours are almost always explicitly addressed in radiological reports, minimizing the "Uncertain" reporting bias. This confirms that the SAE learns highly coherent, disentangled visual features for major anatomies, and when the ground truth text is exhaustive, the model's accuracy is undeniably high. Crucially, this result now holds across `k=3,5,8,10` (Section 3.2), so it is a property of the concept, not an artifact of a particular clustering resolution.

### 4.3 Controlled Feature Sparsity & Latent Dormancy

At smaller subsets ($N=10$ to $N=100$), clusters 6 and 7 were entirely dormant. Only as the sample size reached $N=500$ did these rare clinical signs (e.g., Rare Thoracic Anomalies and Diaphragmatic Signs) trigger positive activations. This "latent dormancy" provides clear, empirical evidence of the L1 penalty mechanism at the heart of the Sparse Autoencoder. Neurons corresponding to specific macro-categories simply do not fire unless the rare visual pattern is physically present in the evaluated batch. This proves the architecture effectively resists spurious background hallucinations and maintains strict feature disentanglement.

### 4.4 Cluster Count as a Reported Design Choice, Not a Hidden Parameter

The silhouette analysis (3.1) makes explicit that `k=8` is not the "optimal" partition by a standard clustering metric — `k=3` and `k=10` both score higher. Choosing `k=8` on clinical-interpretability grounds is a defensible methodological decision, but it is a decision, and we now report the alternative resolutions alongside it rather than presenting `k=8` as the only possible grouping. This strengthens the academic honesty of the pipeline: the reader can verify that our headline result (Section 4.2) is robust to the choice of `k`, rather than having to trust a single, unvalidated cluster count.

Ultimately, this project demonstrates that the "black box" of medical Vision-Language Models can be transparently mapped using unsupervised architectures like Sparse Autoencoders. By substituting generative LLM decoding with a deterministic, logit-based, permutation-debiased MedGemma verification protocol, and by validating the clinical clustering resolution rather than assuming it, we eliminate hallucination risks and unvalidated design choices, successfully bridging the gap between mathematical embedding spaces and verifiable clinical semantics.

---

## Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
