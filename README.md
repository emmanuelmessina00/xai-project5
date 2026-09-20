# Unsupervised Concept Discovery and Evaluation for Medical Vision–Language Models

## Introduction

This project investigates how medical **Vision–Language Models (VLMs)** represent clinical concepts and how those concepts can be evaluated against radiology reports. It was developed for the **Explainable and Trustworthy AI** course at **Politecnico di Torino**, academic year **2025/2026**.

Medical VLMs encode images and text in a shared embedding space, but the meaning of their internal features is often difficult to interpret. Building on **MedConcept**, we adapt unsupervised concept discovery to **2D chest X-rays**: a Sparse Autoencoder (SAE) decomposes frozen **BiomedCLIP** image embeddings into sparse features, which are assigned textual labels from a medical dictionary. We then use **MedGemma 1.5 4B** to assess whether the extracted concepts are supported by the corresponding radiology reports.

Our work combines a literature review and analysis of research gaps with an implemented pipeline, hierarchical concept clustering, and a blinded audit of evaluator reliability. The full methodology, experiments, and discussion are available in the [final report](report/main.pdf). The [project brief](Project5.pdf) describes the original assignment, and the [presentation](presentation/beamerpolito.pdf) summarizes the work.

## Methodology

The implementation is organized into six notebooks, following the order below.

| Step | Notebook | Purpose |
| --- | --- | --- |
| 01 | [Feature extraction](src/notebooks/01_feature_extraction.ipynb) | Extract 512-dimensional BiomedCLIP embeddings from NIH and Open-I chest X-rays, retaining Open-I reports for evaluation. |
| 02 | [Dictionary creation](src/notebooks/02_dictionary_creation.ipynb) | Filter Open-I metadata, retrieve UMLS terminology, expand concepts with Gemini phrases, and encode the dictionary with BiomedCLIP. |
| 03 | [SAE training](src/notebooks/03_sae_training.ipynb) | Train a 1,024-feature SAE with reconstruction and L1 sparsity losses; inspect reconstruction, activations, and concept assignments. |
| 04 | [Concept evaluation](src/notebooks/04_evaluation.ipynb) | Extract concepts from Open-I images and evaluate their agreement with reports using MedGemma. |
| 05 | [Hierarchical clustering](src/notebooks/05_hierarchical_clustering.ipynb) | Group parent concepts using agglomerative clustering, assess silhouette scores, and aggregate existing verdicts at several resolutions. |
| 06 | [Independent validation](src/notebooks/06_independent_validation.ipynb) | Analyze a blinded audit of 25 concept–report pairs, comparing author consensus with MedGemma and commercial evaluators. |

### Datasets and concept dictionary

| Dataset | Size used | Role |
| --- | --- | --- |
| NIH Chest X-ray 10k subset (`g-ronimo/NIH-Chest-X-ray-dataset_10k`) | 7,500 training images; 2,500 validation images | Fit the SAE and select its checkpoint. The dataset's provided test split serves as the validation set. |
| Open-I (`ykumards/open-i`) | 3,666 frontal image–report pairs after preprocessing | Supply the evaluation pool and the metadata used to construct the concept vocabulary. Evaluation covers subsets of 10, 50, 100, and 500 images. |

Open-I reference reports combine the available **Findings** and **Impression** sections. Its images and reports are not used to train the SAE, although its **Problems** metadata supplies the candidate dictionary terms.

The final dictionary contains **998 entries associated with 100 parent concepts**: 498 base-term and synonym entries, plus 500 phrases generated with Gemini 3.1 Flash-Lite. UMLS provides Concept Unique Identifiers for 90 parent concepts; the remaining ten use their base terms as a fallback. BiomedCLIP's text encoder maps each entry to the shared 512-dimensional embedding space.

### Concept extraction and evaluation

Each SAE decoder feature is named using its nearest dictionary entry by cosine similarity. For each image, features are ranked by activation and retained until their cumulative positive activation exceeds 90%; repeated phrase labels are then deduplicated. This produces approximately 26 unique phrases per image in the reported evaluation.

MedGemma receives each phrase and its corresponding report text and assigns one of three verdicts:

- **Aligned:** the report explicitly supports the concept or a clear synonym.
- **Unaligned:** the report explicitly negates or contradicts the concept.
- **Uncertain:** the concept is unmentioned, ambiguous, or tentative.

The evaluator scores single-token answers A, B, and C across three permutations of the class–letter mapping. It averages the resulting class probabilities and selects the highest-scoring verdict, without generating a free-text response. This reduces letter bias but does not guarantee correct judgments.

Hierarchical clustering uses cosine distance and average linkage over parent-concept embeddings. We assess silhouette scores for `k = 3–15` and aggregate the same MedGemma verdicts for `k = 3, 5, 8, 10`. The final report uses **`k = 10`** as the reference resolution, balancing separation and granularity; `k = 3` has the highest silhouette score.

## Results

The selected SAE uses an L1 coefficient of `5e-5`. On 256 validation embeddings, it achieves a reconstruction **MSE of 3.32 × 10⁻⁵**, a mean cosine similarity of **0.9916**, and an average of **44 active features out of 1,024** per image.

The report-based evaluation gives the following mean per-image verdict percentages:

| Images | Phrases per image | Aligned | Unaligned | Uncertain |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 26.20 | 7.25% | 23.17% | 69.57% |
| 50 | 26.22 | 8.40% | 22.88% | 68.72% |
| 100 | 26.23 | 7.64% | 23.37% | 68.99% |
| 500 | 25.99 | 7.21% | 22.49% | 70.30% |

At 500 images, the evaluation includes **12,996 phrase–image pairs**. Alignment varies substantially across the ten reference clusters: Cluster 9 reaches **49.23% Aligned**, while Cluster 5 reaches **0.15%**. Cluster percentages count phrase–image pairs within each group; the overall table gives each image equal weight.

In the blinded audit, the three authors independently assessed **25 concept–report pairs**, with **72% unanimous agreement**. Against their majority verdict, MedGemma achieved **56% agreement**. GPT-5.6 Luna, Claude Sonnet 5, and Gemini 3.1 Pro achieved **68–76%**, depending on the evaluator and whether the input was supplied as a file or pasted text.

These results expose limits of report-based evaluation. Uncertainty may reflect omitted findings, ambiguous dictionary entries, incorrect concept assignments, or evaluator errors. High reconstruction fidelity alone does not establish that a feature has a single clinical meaning. The audit measures agreement on report interpretation in a small, stratified sample, without expert review of the images. See the [final report](report/main.pdf) for the detailed analysis and limitations.

## Project Structure

```text
.
├── src/
│   ├── notebooks/
│   │   ├── 01_feature_extraction.ipynb
│   │   ├── 02_dictionary_creation.ipynb
│   │   ├── 03_sae_training.ipynb
│   │   ├── 04_evaluation.ipynb
│   │   ├── 05_hierarchical_clustering.ipynb
│   │   └── 06_independent_validation.ipynb
│   ├── scripts/
│   │   ├── sae.py                       # SAE architecture and loss functions
│   │   └── clustering.py                # Parent-concept embeddings and clustering
│   └── results/
│       ├── 01_feature_extraction/        # Saved image embeddings and reports
│       ├── 02_dictionary_creation/       # Dictionary, metadata, and API checkpoints
│       ├── 03_sae_training/              # SAE weights and training checkpoints
│       ├── 04_evaluation/                # Concept–report verdicts for each subset
│       ├── 05_hierarchical_clustering/   # Cluster statistics and plots at each k
│       └── 06_independent_validation/   # Audit annotations, prompts, and plots
├── papers/                              # Reference papers
├── report/                              # Final report PDF, LaTeX sources, and figures
├── presentation/                        # Presentation PDF, LaTeX sources, and assets
├── Project5.pdf                         # Professor's project brief
├── requirements.txt                     # Python dependencies
└── README.md
```

Saved embeddings, dictionary artifacts, trained checkpoints, evaluation CSVs, and audit annotations are included under `src/results/`, so individual stages can be inspected or reused without rerunning the full pipeline.

## Requirements and Setup

### Environment

The notebooks record **Python 3.12** environments. The main dependencies are PyTorch, OpenCLIP, Transformers, Hugging Face Datasets, NumPy, pandas, scikit-learn, Matplotlib, and Seaborn. Dictionary construction additionally uses Requests, python-dotenv, and the Google Gen AI SDK. The complete dependency list is in [requirements.txt](requirements.txt); versions are not pinned.

For model execution, use hardware with sufficient memory for BiomedCLIP and MedGemma. The notebooks detect CUDA, Apple MPS, or CPU for feature extraction and SAE operations. Notebook 04 loads MedGemma with `bfloat16` and automatic device placement; compatibility and memory requirements depend on the runtime. Analysis of the saved evaluation and audit CSVs does not require loading MedGemma.

### Local installation

Run the following commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows, activate with: .venv\Scripts\activate

python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel

cd src/notebooks
python -m jupyterlab
```

Execute notebooks with **`src/notebooks/` as the working directory** so their relative paths resolve correctly. JupyterLab and ipykernel are installed separately because they are not listed in `requirements.txt`.

### Credentials

To rebuild the dictionary, create a local `.env` file in the repository root with your credentials, or configure the same names in Google Colab Secrets:

```dotenv
UMLS_API_KEY=your_umls_api_key
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_hugging_face_token
```

Notebook 02 reads the local `.env` file. Notebook 04 checks the `HF_TOKEN` environment variable or saved Hugging Face credentials and otherwise prompts for a token using hidden input. The account must have access to `google/medgemma-1.5-4b-it` to run the evaluator. UMLS and Gemini credentials are needed to regenerate dictionary entries; they are not needed to reuse the saved dictionary.

### Running the experiments

Run notebooks **01 through 06** in order for the complete workflow. The saved artifacts also support starting from later stages:

- **Train or evaluate the SAE:** reuse the embeddings and dictionary from Steps 01–02. The checkpoint used in the report is `sae_model_best_e1000_l5e-05_h1024.pt` in `src/results/03_sae_training/`.
- **Inspect clustering:** run Notebook 05 with the saved dictionary and Step 04 evaluation CSVs. It reuses existing verdicts without querying MedGemma again.
- **Inspect evaluator agreement:** run Notebook 06 with the completed `final_validation_subset.csv`, the 500-image evaluation CSV, and `k_10/df_details_500_samples.csv`. It analyzes recorded human and commercial-model judgments; it does not call those commercial models automatically.

## Contributors

- Riccardo Marconi (riccardo.marconi@studenti.polito.it)
- Davide Candela (davide.candela@studenti.polito.it)
- Emmanuel Messina (s333951@studenti.polito.it)

Politecnico di Torino — Explainable and Trustworthy AI, 2025/2026.
