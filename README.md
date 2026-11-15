# Reproducibility code for the PMH psychiatric dataset

This package provides a script to reproduce key data-quality, descriptive, scale-reliability, and illustrative logistic-regression results for the dataset described in the mda Data Report, *A Multimodal Clinical-Psychosocial and Speech-Linked Dataset of Psychiatric Patients in Bangladesh*.

**The package contains code only. It does not contain participant-level data, identifiers, or audio recordings.** Obtain the released dataset from Mendeley Data: https://doi.org/10.17632/wr8n8kz4vm.3

## Contents

- `PMH.py` — reads the dataset, validates required columns, calculates missingness and categorical frequency tables, summarizes PHQ-9/GAD-7 total scores, calculates Cronbach's alpha from item responses, calculates the PHQ-9/GAD-7 Spearman correlation, summarizes audio-file availability, and runs the fixed logistic-regression examples and apparent/five-fold cross-validated performance summaries.
- `requirements.txt` — Python dependencies.
- `CODEBOOK_MAPPING.md` — variable names and recoding definitions used by the illustrative models.


## Requirements

Python 3.10 or newer is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

From this directory, provide the path to the downloaded public dataset:

```powershell
python PMH.py --input "C:\path\to\PMH Dataset.xlsx" --output "PMH_reproducibility_results.xlsx"
```

For descriptive and data-quality tables without logistic-regression examples:

```powershell
python PMH.py --input "C:\path\to\PMH Dataset.xlsx" --output "PMH_descriptive_results.xlsx" --skip-models
```

CSV and Excel inputs are supported. Column names are normalized by trimming whitespace, replacing spaces/hyphens/slashes with underscores, and converting to lowercase. Category labels used for binary recoding must match the released dataset and codebook. If a required column is missing, the script stops and reports its name.

## Analysis details

- PHQ-9 and GAD-7 binary outcomes are defined as total score >= 10 versus < 10.
- PHQ-9 and GAD-7 item responses are stored as text labels in the public dataset. The script converts them to 0-3 scores (Not at all = 0, Several days = 1, More than half the days = 2, Nearly every day = 3) before calculating Cronbach's alpha from complete item rows.
- Logistic regression uses the fixed predictor sets documented in `CODEBOOK_MAPPING.md`; adjusted odds ratios and Wald 95% confidence intervals are exported.
- In-sample performance uses the fitted statsmodels logistic model at a 0.5 threshold.
- Five-fold cross-validation uses scikit-learn `LogisticRegression(max_iter=1000, solver="liblinear")`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, and a 0.5 threshold.
- The logistic-regression models are illustrative exploratory analyses, not clinically validated prediction tools. Because the predictor sets were fixed after prior data analysis, cross-validated performance may be optimistic and should not be interpreted as external validation.

## Reproducibility status

The calculations were run against the dataset files deposited in Mendeley Data Version 3 (https://doi.org/10.17632/wr8n8kz4vm.3) in a Jupyter notebook using Python 3.13.5, pandas 2.2.3, numpy 2.1.3, scipy 1.15.3, statsmodels 0.14.4, scikit-learn 1.6.1, and openpyxl 3.1.5. The results matched the values reported in the manuscript: Cronbach’s alpha of .667 (PHQ-9) and .726 (GAD-7), Spearman’s rho of .618 between the two total scores, audio-file availability of 187, 187, and 186 recordings, and the logistic-regression performance in Table 8.

The standalone script `PMH.py` was run on the Version 3 dataset file on 12 October 2026 and produced the same values.
