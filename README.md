# CS439 Final Project: Hybrid Risk Phenotyping for Breast Cancer Diagnosis

This repository contains a complete, reproducible data science final project prepared from the CS439 final project instructions and rubric.

## Research Question

Can unsupervised morphology-based risk phenotypes learned with K-Means provide an interpretable signal for predicting whether a breast mass is malignant or benign from the Wisconsin Diagnostic Breast Cancer dataset?

## Project Contents

```text
.
├── data/
│   └── wdbc_clean.csv                    # Generated clean dataset
├── notebooks/
│   └── final_project_analysis.ipynb      # Reproducibility notebook
├── outputs/
│   ├── figures/                          # Generated figures for report
│   └── tables/                           # Generated metrics and metadata
├── report/
│   ├── final_report.tex                  # LaTeX source
│   ├── references.bib                    # Bibliography
│   └── final_report.pdf                  # Compiled paper
├── src/
│   └── train.py                          # Main experiment pipeline
├── requirements.txt
└── README.md
```

## How to Reproduce

1. Create and activate a Python environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the full experiment:

```bash
python src/train.py
```

4. Outputs will be written to:

```text
outputs/tables/
outputs/figures/
```

5. To compile the report from source:

```bash
cd report
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

If `pdflatex` is not working for me on Windows. Tex file is available and all related files in report folder.

## Code Repository

GitHub repository: https://github.com/daniyalfs/cs439-final-project.git

## Main Results

On a stratified 20% held-out test set:

- RBF SVM achieved the highest accuracy: **98.25%**.
- Random Forest achieved the highest ROC-AUC: **0.9970**.
- The proposed hybrid K-Means + Logistic Regression model achieved **96.49% accuracy** and **0.9927 ROC-AUC**.

The hybrid model did not outperform the best supervised baseline, but it provided an interpretable clustering layer that separated morphology-based phenotypes and supported the paper's error analysis.
