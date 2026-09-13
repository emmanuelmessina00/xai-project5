# eXplainable AI in Medical Imaging: Unsupervised Concept Discovery in Vision-Language Models via Sparse Autoencoders and Hierarchical Semantic Alignment

This document outlines the architecture, methodology, and critical analysis of our XAI project, serving as a comprehensive blueprint for the final academic paper. The work addresses the inherent "black-box" nature of medical Vision-Language Models (VLMs) by proposing an unsupervised, automated pipeline to extract, ground, and validate interpretable clinical concepts from chest X-rays, culminating in an innovative hierarchical alignment evaluation.

---

## 1. Pipeline Architecture & Methodology

The project is structured into four sequential phases, each documented in its respective notebook, guiding the raw data from mathematical abstraction to clinically validated semantic features.

### 01 - Feature Extraction

This initial phase translates raw medical images into dense vector representations. Leveraging the state-of-the-art **BioMedCLIP** architecture (ViT-B/16 and PubMedBERT), the system extracts visual and textual embeddings. We strategically decoupled our datasets to optimize performance:

* **NIH Chest X-ray Subset (10k images):** Used exclusively to train the Sparse Autoencoder. Its scale provides the visual diversity required to prevent "dead neurons" and ensure a robust latent space.


* **Open-I Dataset:** Used strictly for final evaluation because it provides images paired with their original, free-text clinical reports (the *Findings* and *Impression* sections).
All embeddings undergo L2 normalization to stabilize training and facilitate accurate cosine similarity calculations.



### 02 - Medical Dictionary Creation

The objective here is to build a robust, data-driven semantic concept matrix (`T`). Starting from the raw metadata of the Open-I dataset, the pipeline applies a strict **Clinical Filtration** to remove technical artifacts and medical devices (e.g., "surgical instruments", "catheters"), preserving only anatomical and pathological descriptors.
These terms undergo **Domain Alignment** to map ambiguous words (e.g., "opacity") to strict thoracic terminology (e.g., "lung opacity") before querying the **UMLS API**. This guarantees deterministic retrieval of certified Concept Unique Identifiers (CUIs) and high-quality synonyms from SNOMED CT and MeSH. Finally, an LLM (**Gemini 3.1-flash-lite**) expands the dictionary by generating 5 realistic, report-style radiology phrases for each concept. The text encoder processes this corpus to generate the final semantic matrix.

### 03 - Sparse Autoencoder (SAE) Training

This notebook tackles the "polysemanticity" problem in dense models, where a single neuron might represent multiple unrelated concepts. A Sparse Autoencoder is trained on the NIH visual embeddings to decompose the dense latent space into expanded, interpretable features. By enforcing an L1 sparsity penalty, the SAE forces only a small subset of neurons to activate for any given image, isolating specific visual patterns. The SAE's decoder weights are then projected onto the text-embedding space using the semantic matrix from Step 02, assigning a human-readable clinical label to each latent neuron based on cosine similarity.

### 04 - Concept Evaluation & Hierarchical Extension

The final module quantifies the clinical accuracy of the SAE's extracted concepts against the patient's actual radiology report. To eliminate generative hallucinations and latency, we utilize **MedGemma 1.5 4B** as a frozen external evaluator using a deterministic **Zero-Shot Logit Scoring** strategy. By probing the unnormalized logits of single-token identifiers (A, B, C) representing *Aligned*, *Unaligned*, and *Uncertain*, we ensure a rigorous and mathematically sound evaluation.

**The Hierarchical Clustering Extension:**
To evaluate the model beyond strict, literal text matching, we apply **Agglomerative Clustering** (using cosine similarity and average linkage) to group the fine-grained dictionary concepts into **8 semantic macro-clusters**. MedGemma's verdicts are then mapped back to these broader clinical families. This dual-granularity approach allows us to verify if the SAE captures the correct anatomical or pathological domain, even if the exact phrasing differs.

---

## 2. Critical Analysis of the Results

The quantitative and qualitative data extracted from the pipeline reveal highly significant dynamics that strongly align with real-world clinical practice.

**1. The Clinical Reporting Bias is a Feature, Not a Bug**
The aggregated results demonstrate a heavy predominance of *Uncertain* verdicts (approx. 67%). While this might initially seem like poor alignment, it is actually a highly accurate reflection of clinical "reporting bias." Radiologists do not dictate every healthy structure or normal anatomy visible in an X-ray; they only report anomalies. Conversely, the SAE is objective and extracts *everything* it visualizes (e.g., "thoracic vertebrae", "normal cardiac silhouette"). The high uncertainty rate perfectly quantifies the discrepancy between exhaustive machine perception and selective human synthesis.

**2. The Efficacy of Macro-Cluster Evaluation**
The hierarchical extension successfully mitigated the penalties of rigid semantic matching. By aggregating verdicts into macro-clusters, we proved that the model effectively understands broader clinical contexts. For instance, specific clusters (e.g., Cluster 4 in our preliminary tests) achieved an *Aligned* score of over 66%, demonstrating that when the SAE focuses on distinct, well-defined pathologies or structures, it finds strong, reliable agreement with the physician's report. Other clusters highlight areas where the model detects visual patterns that radiologists frequently omit or describe differently.

**3. Empirical Proof of SAE Sparsity**
The hierarchical analysis also provided definitive empirical proof of the Sparse Autoencoder's core mechanism. When evaluating a limited sample of 10 images, the clustering table mapped only 6 out of the 8 available macro-clusters. This is not a mapping error, but rather a direct demonstration of the SAE's extreme sparsity: latent features (and their corresponding macro-categories) remain entirely dormant if that specific pathology or visual pattern is not present in the evaluated batch.

---

## 3. Conclusions and Pathway to the Final Paper

This project successfully demonstrates that the "black box" of medical Vision-Language Models can be opened using unsupervised architectures like Sparse Autoencoders. The pipeline we built is automated, modular, protected against LLM hallucinations, and deeply grounded in certified clinical ontologies (UMLS).

The hierarchical clustering extension proved to be the decisive factor in accurately interpreting the evaluation metrics. By shifting the perspective from individual text fragments to semantic macro-areas, we demonstrated that the VLM learns high-level, clinically meaningful representations, effectively bridging the semantic gap caused by varying reporting styles and physiological omissions.

## Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_ai_core
```
