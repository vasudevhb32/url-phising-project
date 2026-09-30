# Phishing URL Behaviour Classification (Problem 02)
Learn Depth Academy – Track 1 Capstone. Educational prototype, free/open-source tools only.

## Run
```
pip install -r requirements.txt
python train.py            # cleans data, trains 3 models, saves model.joblib + plots + metrics.json
streamlit run app.py       # prototype
```

## Dataset (credit)
`data/urldata_raw.csv` = "5.urldata.csv" (10,000 URLs, 17 features; 5,000 phishing from PhishTank, 5,000 legitimate from the
University of New Brunswick URL dataset), from GitHub repo *shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques*
(`DataFiles/5.urldata.csv`). Please cite PhishTank and UNB as original sources.

## What was done
1. **Data quality**: 0 missing values, but 5,626 of 10,000 rows were exact duplicates -> removed (4,374 left: 3,845 phishing / 529 legit).
2. **Only required rows**: balanced sample = all 529 legit + 529 random phishing = 1,058 rows (80/20 stratified split).
3. **Models**: Logistic Regression, Decision Tree (depth 4), KNN (k=15); 5-fold CV on train, metrics on held-out test.
4. **Final model: Logistic Regression** – best CV F1 and ROC-AUC, simple, interpretable coefficients, gives probabilities.

## Results (test set, 212 rows) – see metrics.json
| Model | Acc | Prec | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.792 | 0.878 | 0.679 | 0.766 | 0.882 |
| Decision Tree | 0.792 | 0.956 | 0.613 | 0.747 | 0.837 |
| KNN | 0.778 | 0.839 | 0.689 | 0.756 | 0.865 |

Confusion matrix (LR): 96 legit correct, 10 legit flagged as phishing, 34 phishing missed, 72 phishing caught.
Interpretation: precision is high (when it says phishing it is usually right) but recall is lower (about 1 in 3 phishing URLs is missed).

## Limitations (say these in the viva)
- **Data leakage / artifact**: `URL_Length` = 0 never occurs for legitimate rows, so it is a near-perfect "phishing" shortcut (largest coefficient).
  `Right_Click` behaves similarly. This is a dataset-collection artifact and may not hold on real-world URLs.
- Original dataset stores only the domain, so features are dataset-encoded 0/1 flags, not raw URL text; several features
  (DNS, traffic, domain age) need external lookups, so the app takes them as inputs.
- Small data (1,058 rows) -> results vary with the split; educational prototype only.
- Future work: use a larger raw-URL dataset (e.g. URL-Phish, Mendeley DOI 10.17632/65z9twcx3r) with lexical features computed directly from the URL.
- 
