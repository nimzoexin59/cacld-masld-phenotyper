"""
cACLD-MASLD Phenotype Classifier — Streamlit web app (EN/IT)

Companion tool for the paper:
"Data-driven baseline phenotyping in MASLD-related compensated advanced
chronic liver disease: a multicentre cohort study" (in preparation, 2026).

Given the 15 baseline clinical features of a new patient with cACLD-MASLD,
this tool assigns one of four super-phenotypes (Metabolic / Non-metabolic /
Liver / Dyslipidemic) using nearest-prototype assignment in standardised
feature space, following the DDCL-INCRT-T methodology.

Research tool — NOT for clinical decision-making.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----- LOAD MODEL DATA -----
DATA_FILE = Path(__file__).parent / "app_data.json"
with open(DATA_FILE) as f:
    M = json.load(f)

FEATURES = M["features"]
MEAN = M["mean"]
SD = M["sd"]
PROTOTYPES_Z = {k: np.array([v[feat] for feat in FEATURES])
                for k, v in M["prototypes_z"].items()}
CIF_5Y = M["cif_5y"]
CLUSTER_SIZES = M["cluster_sizes"]

PHENOTYPES = ["Metabolic", "Non-metabolic", "Liver", "Dyslipidemic"]
PHENOTYPE_COLOR = {
    "Metabolic":     "#1f77b4",
    "Non-metabolic": "#2ca02c",
    "Liver":         "#d62728",
    "Dyslipidemic":  "#ff7f0e",
}
OUTCOMES = ["Decompensation", "HCC", "Liver event", "CV event", "EHC event", "Overall death"]

# ----- I18N STRINGS -----
T = {
    "en": {
        "title": "cACLD-MASLD Phenotype Classifier",
        "subtitle": ("Companion tool for *Cirrincione, Pennisi, Celsa, Di Maria, Petta, Cammà, "
                     "'Data-driven baseline phenotyping in MASLD-related compensated advanced chronic "
                     "liver disease' (2026, in preparation)*."),
        "disclaimer": ("**Research tool — NOT for clinical decision-making.** "
                       "This classifier assigns a baseline phenotype based on the multicentre cohort "
                       "described in the paper. Predictions have not been prospectively validated. "
                       "Clinical decisions should be based on full patient context and standard guidelines."),
        "how_it_works": "How it works",
        "how_it_works_body": (
            "The tool implements the nearest-prototype assignment of the DDCL-INCRT-T "
            "clustering method. Each of the four super-phenotypes is represented by a "
            "centroid in 15-dimensional feature space, computed from the {N:,} patient "
            "cohort described in the paper.\n\n"
            "For an input patient:\n\n"
            "1. Each input value is **z-score standardised** using the training-cohort mean and SD.\n"
            "2. The **masked Euclidean distance** to each of the four prototypes is computed "
            "using only the features actually entered (no imputation).\n"
            "3. The patient is assigned to the **nearest prototype**.\n"
            "4. A **stability score** in [0, 1] is computed as the relative gap between the nearest "
            "and second-nearest distances; values near 1 indicate confident assignment, values near 0 "
            "indicate the patient is on the boundary between two phenotypes.\n\n"
            "At least 3 features must be provided for a valid assignment."
        ),
        "about_cohort": "About the cohort",
        "patient_data": "Patient baseline data",
        "patient_data_caption": "Fill in as many fields as available. All are optional; at least 3 are needed.",
        "demographics": "Demographics & comorbidities",
        "liver_function": "Liver function & fibrosis",
        "metabolic_panel": "Metabolic panel",
        "classify": "Classify patient",
        "empty_choice": "—",
        "yes": "Yes",
        "no": "No",
        "male": "Male",
        "female": "Female",
        "field_age": "Age (years)",
        "field_gender": "Gender",
        "field_diabetes": "Diabetes",
        "field_htn": "Hypertension",
        "field_lsm": "LSM (kPa)",
        "field_platelets": "Platelets (×10⁹/L)",
        "field_albumin": "Albumin (g/dL)",
        "field_bilirubin": "Bilirubin (mg/dL)",
        "field_inr": "INR",
        "field_creatinin": "Creatinin (mg/dL)",
        "field_alt": "ALT (U/L)",
        "field_cholesterol": "Cholesterol (mg/dL)",
        "field_triglycerides": "Triglycerides (mg/dL)",
        "field_hdl": "HDL (mg/dL)",
        "field_glucose": "Fasting glucose (mg/dL)",
        "err_too_few": "Please provide at least 3 baseline values for a valid assignment.",
        "result_predicted": "Predicted phenotype: {phen}",
        "result_stats": "Stability score: {s:.2f}  |  Features used: {n} / 15  |  Second nearest: {second}",
        "distances_h": "Distances to phenotype prototypes",
        "distances_caption": ("Lower distance = more similar to that phenotype. The stability score reflects "
                              "how clearly the patient sits in one phenotype vs the next nearest."),
        "col_phenotype": "Phenotype",
        "col_distance": "Squared distance (z-score space, rescaled)",
        "col_predicted": "Predicted",
        "interp_high": "✅ **High confidence**: clearly closer to the predicted phenotype than to any other.",
        "interp_mod": "🟡 **Moderate confidence**: phenotype assignment is reasonable but not far from the next nearest.",
        "interp_bord": "⚠️ **Borderline**: the patient is on the boundary between phenotypes. Consider also the second-nearest phenotype ({second}).",
        "risk_h": "5-year incidence in {phen} phenotype (from cohort)",
        "risk_caption": ("These are the 5-year cumulative incidence estimates from Table 2 of the paper "
                         "(Aalen–Johansen, competing risks) for the predicted phenotype. They are "
                         "*group-level* statistics, not individual predictions for this patient."),
        "col_outcome": "Outcome",
        "col_cif5y": "5-year incidence (%)",
        "footer": ("Tool developed by G. Cirrincione (Laboratoire LTI, UPJV Amiens) for the cACLD-MASLD "
                   "phenotyping companion paper (2026). Method: DDCL-INCRT-T (Cirrincione, Messaoudi, Cammà, "
                   "in resubmission to Artificial Intelligence in Medicine, 2026)."),
        "outcome_decomp": "Decompensation",
        "outcome_hcc": "HCC",
        "outcome_liver": "Liver event",
        "outcome_cv": "CV event",
        "outcome_ehc": "EHC event",
        "outcome_death": "Overall death",
        "phen_desc": {
            "Metabolic": ("Metabolic syndrome features dominant: high prevalence of diabetes "
                          "(89%) and hypertension (75%); moderate fibrosis (LSM ~16 kPa); "
                          "balanced sex distribution. Largest super-phenotype (n=1,245, 47%)."),
            "Non-metabolic": ("Low prevalence of metabolic comorbidity (DM 3%, HTN 21%); "
                              "younger patients (mean age 52) with isolated ALT elevation; "
                              "preserved liver function. Rule-out phenotype with the lowest "
                              "5-year incidence of all outcomes. (n=598, 23%)."),
            "Liver": ("Advanced fibrosis or compensated cirrhosis: high LSM (mean 32 kPa), "
                      "low platelets (~130), elevated bilirubin and INR. Dominant phenotype "
                      "for liver-related outcomes and mortality. (n=596, 23%)."),
            "Dyslipidemic": ("Severe metabolic dyslipidemia: very high triglycerides "
                             "(mean 390 mg/dL), elevated ALT (mean 157 U/L) and cholesterol "
                             "(236 mg/dL); younger patients. Smallest super-phenotype "
                             "(n=200, 8%)."),
        },
    },
    "it": {
        "title": "Classificatore di fenotipo cACLD-MASLD",
        "subtitle": ("Strumento di accompagnamento al lavoro *Cirrincione, Pennisi, Celsa, Di Maria, Petta, Cammà, "
                     "'Data-driven baseline phenotyping in MASLD-related compensated advanced chronic "
                     "liver disease' (2026, in preparazione)*."),
        "disclaimer": ("**Strumento di ricerca — NON destinato a decisioni cliniche.** "
                       "Questo classificatore assegna un fenotipo basale sulla base della coorte multicentrica "
                       "descritta nel paper. Le previsioni non sono state validate prospetticamente. "
                       "Le decisioni cliniche devono basarsi sul contesto completo del paziente e sulle linee guida standard."),
        "how_it_works": "Come funziona",
        "how_it_works_body": (
            "Lo strumento implementa l'assegnazione al prototipo più vicino del metodo di clustering "
            "DDCL-INCRT-T. Ognuno dei quattro super-fenotipi è rappresentato da un centroide nello "
            "spazio delle 15 feature, calcolato sulla coorte di {N:,} pazienti del paper.\n\n"
            "Per un paziente inserito:\n\n"
            "1. Ogni valore viene **standardizzato in z-score** usando media e SD della coorte di training.\n"
            "2. Si calcola la **distanza euclidea masked** a ciascuno dei quattro prototipi, "
            "usando solo le feature effettivamente inserite (nessuna imputazione).\n"
            "3. Il paziente viene assegnato al **prototipo più vicino**.\n"
            "4. Si calcola uno **stability score** in [0, 1] come gap relativo tra la distanza al primo "
            "e al secondo prototipo più vicini; valori vicini a 1 indicano assegnazione netta, valori "
            "vicini a 0 indicano un paziente al confine tra due fenotipi.\n\n"
            "Servono almeno 3 feature per un'assegnazione valida."
        ),
        "about_cohort": "Sulla coorte",
        "patient_data": "Dati basali del paziente",
        "patient_data_caption": "Compila tutti i campi che hai. Tutti sono opzionali; ne servono almeno 3.",
        "demographics": "Demografia & comorbidità",
        "liver_function": "Funzionalità epatica & fibrosi",
        "metabolic_panel": "Profilo metabolico",
        "classify": "Classifica paziente",
        "empty_choice": "—",
        "yes": "Sì",
        "no": "No",
        "male": "Maschio",
        "female": "Femmina",
        "field_age": "Età (anni)",
        "field_gender": "Sesso",
        "field_diabetes": "Diabete",
        "field_htn": "Ipertensione",
        "field_lsm": "LSM (kPa)",
        "field_platelets": "Piastrine (×10⁹/L)",
        "field_albumin": "Albumina (g/dL)",
        "field_bilirubin": "Bilirubina (mg/dL)",
        "field_inr": "INR",
        "field_creatinin": "Creatinina (mg/dL)",
        "field_alt": "ALT (U/L)",
        "field_cholesterol": "Colesterolo (mg/dL)",
        "field_triglycerides": "Trigliceridi (mg/dL)",
        "field_hdl": "HDL (mg/dL)",
        "field_glucose": "Glicemia a digiuno (mg/dL)",
        "err_too_few": "Inserisci almeno 3 valori basali per un'assegnazione valida.",
        "result_predicted": "Fenotipo predetto: {phen}",
        "result_stats": "Stability score: {s:.2f}  |  Feature usate: {n} / 15  |  Secondo più vicino: {second}",
        "distances_h": "Distanze ai prototipi dei fenotipi",
        "distances_caption": ("Distanza minore = più simile a quel fenotipo. Lo stability score riflette quanto "
                              "il paziente sta nettamente in un fenotipo rispetto al successivo."),
        "col_phenotype": "Fenotipo",
        "col_distance": "Distanza quadratica (spazio z-score, ri-scalata)",
        "col_predicted": "Predetto",
        "interp_high": "✅ **Alta confidenza**: chiaramente più vicino al fenotipo predetto rispetto agli altri.",
        "interp_mod": "🟡 **Confidenza moderata**: l'assegnazione è ragionevole ma non distante dal secondo più vicino.",
        "interp_bord": "⚠️ **Borderline**: il paziente è al confine tra fenotipi. Considera anche il secondo più vicino ({second}).",
        "risk_h": "Incidenza a 5 anni nel fenotipo {phen} (dalla coorte)",
        "risk_caption": ("Sono le incidenze cumulative a 5 anni della Tabella 2 del paper "
                         "(Aalen–Johansen, competing risks) per il fenotipo predetto. Sono statistiche "
                         "*di gruppo*, non previsioni individuali per questo paziente."),
        "col_outcome": "Outcome",
        "col_cif5y": "Incidenza a 5 anni (%)",
        "footer": ("Strumento sviluppato da G. Cirrincione (Laboratoire LTI, UPJV Amiens) per il paper companion "
                   "cACLD-MASLD (2026). Metodo: DDCL-INCRT-T (Cirrincione, Messaoudi, Cammà, in resubmission "
                   "ad Artificial Intelligence in Medicine, 2026)."),
        "outcome_decomp": "Scompenso",
        "outcome_hcc": "HCC",
        "outcome_liver": "Evento epatico",
        "outcome_cv": "Evento CV",
        "outcome_ehc": "Evento EHC",
        "outcome_death": "Mortalità complessiva",
        "phen_desc": {
            "Metabolic": ("Sindrome metabolica dominante: alta prevalenza di diabete (89%) e ipertensione (75%); "
                          "fibrosi moderata (LSM ~16 kPa); distribuzione sessuale bilanciata. "
                          "Super-fenotipo più numeroso (n=1.245, 47%)."),
            "Non-metabolic": ("Bassa prevalenza di comorbidità metabolica (DM 3%, HTN 21%); pazienti più giovani "
                              "(età media 52) con ALT elevato isolato; funzione epatica preservata. "
                              "Fenotipo di rule-out con la minima incidenza a 5 anni in tutti gli outcome. (n=598, 23%)."),
            "Liver": ("Fibrosi avanzata o cirrosi compensata: LSM elevato (media 32 kPa), piastrine basse (~130), "
                      "bilirubina e INR elevati. Fenotipo dominante per outcome epatici e mortalità. (n=596, 23%)."),
            "Dyslipidemic": ("Dislipidemia metabolica grave: trigliceridi molto alti (media 390 mg/dL), ALT (157 U/L) "
                             "e colesterolo (236 mg/dL) elevati; pazienti più giovani. "
                             "Super-fenotipo più piccolo (n=200, 8%)."),
        },
    },
}

# outcome key → translation key
OUTCOME_KEY = {
    "Decompensation": "outcome_decomp",
    "HCC": "outcome_hcc",
    "Liver event": "outcome_liver",
    "CV event": "outcome_cv",
    "EHC event": "outcome_ehc",
    "Overall death": "outcome_death",
}


# ----- LOGIC -----
def standardize(values: dict) -> dict:
    z = {}
    for f in FEATURES:
        v = values.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            z[f] = None
        else:
            z[f] = (v - MEAN[f]) / SD[f]
    return z


def masked_distance(z_patient: dict, prototype: np.ndarray) -> float:
    observed = [f for f in FEATURES if z_patient[f] is not None]
    if len(observed) < 3:
        return np.nan
    d = len(FEATURES)
    diffs = np.array([z_patient[f] - prototype[FEATURES.index(f)] for f in observed])
    sq = float((diffs ** 2).sum())
    return sq * (d / len(observed))


def assign(values: dict):
    z = standardize(values)
    n_obs = sum(1 for f in FEATURES if z[f] is not None)
    if n_obs < 3:
        return None
    distances = {n: masked_distance(z, p) for n, p in PROTOTYPES_Z.items()}
    sorted_d = sorted(distances.items(), key=lambda x: x[1])
    nearest, second = sorted_d[0], sorted_d[1]
    d1, d2 = nearest[1], second[1]
    s = (np.sqrt(d2) - np.sqrt(d1)) / (np.sqrt(d2) + np.sqrt(d1)) if (d1 + d2) > 0 else 0.0
    return {
        "predicted_phenotype": nearest[0],
        "second_nearest": second[0],
        "distances": distances,
        "stability": s,
        "n_observed": n_obs,
    }


# ----- STREAMLIT UI -----
st.set_page_config(
    page_title="cACLD-MASLD Phenotype Classifier",
    page_icon="🩺",
    layout="wide",
)

# Language selector (top-right)
top_l, top_r = st.columns([4, 1])
with top_r:
    lang_choice = st.selectbox(
        "🌐 Language / Lingua",
        ["English", "Italiano"],
        index=0,
        label_visibility="collapsed",
    )
lang = "en" if lang_choice == "English" else "it"
t = T[lang]

st.title(t["title"])
st.markdown(t["subtitle"])
st.warning(t["disclaimer"], icon="⚠️")

with st.expander(t["how_it_works"], expanded=False):
    st.markdown(t["how_it_works_body"].format(N=sum(CLUSTER_SIZES.values())))

with st.sidebar:
    st.header(t["about_cohort"])
    for phen in PHENOTYPES:
        st.markdown(
            f"<span style='color:{PHENOTYPE_COLOR[phen]}; font-weight:bold'>● {phen}</span> "
            f"<small>(n = {CLUSTER_SIZES[phen]:,})</small>",
            unsafe_allow_html=True,
        )
        st.caption(t["phen_desc"][phen])
        st.divider()

st.subheader(t["patient_data"])
st.caption(t["patient_data_caption"])

st.markdown(f"**{t['demographics']}**")
col_dem = st.columns([1, 1, 1, 1])
with col_dem[0]:
    age = st.number_input(t["field_age"], min_value=0.0, max_value=120.0, value=None, step=1.0, format="%g")
with col_dem[1]:
    gender_label = st.selectbox(t["field_gender"], [t["empty_choice"], t["male"], t["female"]], index=0)
    if gender_label == t["empty_choice"]:
        gender = None
    elif gender_label == t["male"]:
        gender = 1.0
    else:
        gender = 2.0
with col_dem[2]:
    diab_label = st.selectbox(t["field_diabetes"], [t["empty_choice"], t["no"], t["yes"]], index=0)
    if diab_label == t["empty_choice"]:
        diabetes = None
    elif diab_label == t["no"]:
        diabetes = 0.0
    else:
        diabetes = 1.0
with col_dem[3]:
    htn_label = st.selectbox(t["field_htn"], [t["empty_choice"], t["no"], t["yes"]], index=0)
    if htn_label == t["empty_choice"]:
        hypertension = None
    elif htn_label == t["no"]:
        hypertension = 0.0
    else:
        hypertension = 1.0

st.markdown(f"**{t['liver_function']}**")
col_liver = st.columns([1, 1, 1, 1, 1])
with col_liver[0]:
    lsm = st.number_input(t["field_lsm"], min_value=0.0, value=None, step=0.1, format="%g")
with col_liver[1]:
    plt_v = st.number_input(t["field_platelets"], min_value=0.0, value=None, step=1.0, format="%g")
with col_liver[2]:
    albumin = st.number_input(t["field_albumin"], min_value=0.0, value=None, step=0.1, format="%g")
with col_liver[3]:
    bilirubin = st.number_input(t["field_bilirubin"], min_value=0.0, value=None, step=0.05, format="%g")
with col_liver[4]:
    inr = st.number_input(t["field_inr"], min_value=0.0, value=None, step=0.05, format="%g")

col_liver2 = st.columns([1, 1, 1])
with col_liver2[0]:
    creatinin = st.number_input(t["field_creatinin"], min_value=0.0, value=None, step=0.05, format="%g")
with col_liver2[1]:
    alt = st.number_input(t["field_alt"], min_value=0.0, value=None, step=1.0, format="%g")

st.markdown(f"**{t['metabolic_panel']}**")
col_met = st.columns([1, 1, 1, 1])
with col_met[0]:
    cholesterol = st.number_input(t["field_cholesterol"], min_value=0.0, value=None, step=1.0, format="%g")
with col_met[1]:
    triglycerides = st.number_input(t["field_triglycerides"], min_value=0.0, value=None, step=1.0, format="%g")
with col_met[2]:
    hdl = st.number_input(t["field_hdl"], min_value=0.0, value=None, step=1.0, format="%g")
with col_met[3]:
    glucose = st.number_input(t["field_glucose"], min_value=0.0, value=None, step=1.0, format="%g")

values = {
    'Age': age, 'Gender': gender, 'Diabetes': diabetes, 'Hypertension': hypertension,
    'LSM': lsm, 'Platelets': plt_v, 'Albumin': albumin, 'Bilirubin': bilirubin,
    'INR': inr, 'Creatinin': creatinin, 'ALT': alt,
    'Cholesterol': cholesterol, 'Triglycerides': triglycerides, 'HDL': hdl, 'Glucose': glucose,
}

st.divider()
go = st.button(t["classify"], type="primary", use_container_width=True)

if go:
    res = assign(values)
    if res is None:
        st.error(t["err_too_few"])
    else:
        phen = res["predicted_phenotype"]
        color = PHENOTYPE_COLOR[phen]

        st.markdown(
            f"<div style='padding:1.5rem; border-radius:0.5rem; background-color:{color}; color:white;'>"
            f"<h2 style='margin:0; color:white;'>{t['result_predicted'].format(phen=phen)}</h2>"
            f"<p style='margin:0.5rem 0 0 0; font-size:1.1rem;'>"
            f"{t['result_stats'].format(s=res['stability'], n=res['n_observed'], second=res['second_nearest'])}"
            f"</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(t["phen_desc"][phen])

        st.subheader(t["distances_h"])
        st.caption(t["distances_caption"])
        d_df = pd.DataFrame([
            {t["col_phenotype"]: name,
             t["col_distance"]: dist,
             t["col_predicted"]: "★" if name == phen else ""}
            for name, dist in sorted(res["distances"].items(), key=lambda x: x[1])
        ])
        st.dataframe(d_df, hide_index=True, use_container_width=True)

        s = res["stability"]
        if s >= 0.20:
            st.markdown(t["interp_high"])
        elif s >= 0.08:
            st.markdown(t["interp_mod"])
        else:
            st.markdown(t["interp_bord"].format(second=res["second_nearest"]))

        st.subheader(t["risk_h"].format(phen=phen))
        st.caption(t["risk_caption"])
        risks = CIF_5Y[phen]
        risk_df = pd.DataFrame([
            {t["col_outcome"]: t[OUTCOME_KEY[ep]], t["col_cif5y"]: f"{risks[ep]:.1f}"}
            for ep in OUTCOMES
        ])
        st.dataframe(risk_df, hide_index=True, use_container_width=True)

st.divider()
st.caption(t["footer"])
