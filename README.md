# cACLD-MASLD Phenotype Classifier — Web App

Companion tool for the paper *"Data-driven baseline phenotyping in MASLD-related
compensated advanced chronic liver disease: a multicentre cohort study"*
(Cirrincione, Pennisi, Celsa, Di Maria, Petta, Cammà — 2026, in preparation).

Given the 15 baseline clinical features of a new patient with cACLD-MASLD,
the tool assigns one of four super-phenotypes (Metabolic / Non-metabolic /
Liver / Dyslipidemic) using **nearest-prototype assignment** in standardised
feature space (DDCL-INCRT-T methodology).

**Available in English and Italian** — language selector in the top-right corner.

---

## What it does

1. Takes patient input: 15 baseline features (demographics, comorbidities,
   LSM, liver function panel, metabolic panel). Missing values allowed.
2. Standardises with the training-cohort mean and SD (N = 2,639).
3. Computes masked Euclidean distance to each of the four prototypes.
4. Returns:
   - **Predicted super-phenotype** (nearest prototype)
   - **Stability score** in [0, 1] (gap between nearest and second-nearest)
   - Distances to all four prototypes
   - 5-year cumulative incidence for the predicted phenotype (group-level)

**At least 3 features must be provided** for a valid assignment.

---

## How to deploy on Streamlit Community Cloud (free)

The fastest path to a public URL — about 5 minutes:

1. Create a new GitHub repo, e.g. `cACLD-phenotyper`.
2. Add the two files from this folder: `app.py` and `app_data.json`.
3. Add the `requirements.txt` (in this folder).
4. Go to **https://share.streamlit.io/** and click "New app".
5. Connect to the GitHub repo, select `app.py` as the main file, and click Deploy.
6. After ~2 minutes the app is live at a URL like
   `https://<your-app-name>.streamlit.app`.

The URL is stable. It can be added to the paper in galley proofs.

---

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in a browser.

---

## Files in this folder

- `app.py` — the Streamlit app
- `app_data.json` — the trained model artefacts:
  - Training-cohort mean and SD for z-score standardisation
  - The four super-phenotype prototypes (centroids in z-score space)
  - 5-year CIF table from the paper (Table 2)
- `requirements.txt` — Python dependencies
- `README.md` — this file

---

## Citation

If you use this tool, please cite:

> Cirrincione G, Pennisi G, Celsa C, Di Maria G, Petta S, Cammà C. Data-driven
> baseline phenotyping in MASLD-related compensated advanced chronic liver
> disease: a multicentre cohort study. 2026 (in preparation).

Method:

> Cirrincione G, Messaoudi S, Cammà C. DDCL-INCRT-T: Self-Organising Hierarchical
> Prototype Clustering for Tabular Data, with Application to Clinical Phenotyping.
> Artificial Intelligence in Medicine, 2026 (in resubmission).

---

## Disclaimer

**This is a research tool. It is NOT intended for clinical decision-making.**

The phenotype assignment is based on a multicentre cohort and has not been
prospectively validated for individual patient prognosis. Clinical decisions
about diagnosis, prognosis, surveillance, or treatment should be based on
full patient context, current clinical guidelines, and the judgement of the
treating physician.

No patient data entered into this tool is stored or transmitted to any
external server.

---

## Contact

For questions: giansalvo.cirrincione@u-picardie.fr
