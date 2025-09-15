import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import pymc as pm
import arviz as az
from econml.dr import DRLearner
from econml.dml import CausalForestDML

# ================================
# 1. Judul Aplikasi
# ================================
st.set_page_config(page_title="Insightify", layout="wide")

st.title("📊 Insightify – Marketing Campaign Effectiveness")
st.markdown("""
Insightify membantu bisnis mengevaluasi efektivitas kampanye pemasaran 
dengan **causal inference** dan memberikan prediksi apakah campaign 
**ON Target** untuk pelanggan tertentu.
---
""")

# ================================
# 2. Load Model & Metadata
# ================================
@st.cache_resource
def load_artifacts():
    # Load model yang sudah di-save
    df_encoded = joblib.load("models/df_encoded.pkl")
    causal_forest = joblib.load("models/causal_forest.pkl")
    dr_learner = joblib.load("models/dr_learner.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Load metadata (confounders, categorical_cols, threshold)
    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)

    confounders = metadata["confounders"]
    numeric_conf = metadata["numeric_conf"]
    categorical_cols = metadata["categorical_cols"]
    threshold_value = metadata["threshold_value"]

    return df_encoded, causal_forest, dr_learner, scaler, confounders, numeric_conf, categorical_cols, threshold_value

df_encoded, causal_forest, dr_learner, scaler, confounders, numeric_conf, categorical_cols, threshold_value = load_artifacts()

# ================================
# 3. Tabs Setup
# ================================
tab1, tab2 = st.tabs(["🎯 Prediksi Individual", "📈 Business Insights"])

# ================================
# 4. Tab 1 – Prediksi Individual
# ================================
with tab1:
    st.header("Prediksi Campaign untuk Pelanggan")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia", min_value=18, max_value=70, value=30)
        income = st.number_input("Income", min_value=3_000_000, max_value=50_000_000, value=15_000_000, step=500_000)
        web = st.number_input("Web Purchases", 0, 20, 3)
        store = st.number_input("Store Purchases", 0, 20, 5)

    with col2:
        catalog = st.number_input("Catalog Purchases", 0, 10, 2)
        mntfood = st.number_input("MntFood", 0, 5_000_000, 1_000_000, step=100_000)
        mntfashion = st.number_input("MntFashion", 0, 5_000_000, 1_500_000, step=100_000)
        mntelectronics = st.number_input("MntElectronics", 0, 5_000_000, 2_000_000, step=100_000)

    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Widowed"])
    city = st.selectbox("City", ["Jakarta", "Surabaya", "Medan", "Denpasar", "Makassar"])
    period = st.selectbox("Period", ["Before", "After"])

    if st.button("Prediksi Campaign"):
        sample_customer = {
            "income": income, "age": age,
            "webpurchases": web, "storepurchases": store,
            "catalogpurchases": catalog, "mntfood": mntfood,
            "mntfashion": mntfashion, "mntelectronics": mntelectronics,
            "gender": gender, "maritalstatus": marital,
            "city": city, "period": period
        }

        # Preprocessing sama dengan training
        df_new = pd.DataFrame([sample_customer])
        df_new_encoded = pd.get_dummies(df_new, columns=categorical_cols, drop_first=True)
        df_new_aligned = df_new_encoded.reindex(columns=confounders, fill_value=0)
        df_new_aligned[numeric_conf] = scaler.transform(df_new_aligned[numeric_conf])

        # Prediksi ITE dengan Causal Forest
        ite_pred = causal_forest.effect(df_new_aligned)[0]

        if ite_pred > threshold_value:
            st.success(f"✅ Campaign ON Target! (Predicted Uplift: Rp {ite_pred:,.0f})")
        else:
            st.error(f"❌ Campaign OFF Target")

# ================================
# 5. Tab 2 – Business Insights
# ================================
with tab2:
    st.header("Business Insights dari Analisis Kausal")

    st.markdown("""
    - **Average Treatment Effect (ATE)** = dampak rata-rata campaign.  
    - **Distribusi uplift (ITE)** = variasi efek antar individu.  
    - **ROI Simulator** = proyeksi keuntungan campaign.  
    - **Batch Scoring** = uji prediksi pada 1000 pelanggan baru.  
    """)

    # Ringkasan ATE
    st.subheader("Ringkasan ATE")
    ate_dr = dr_learner.ate(pd.DataFrame(np.zeros((1, len(confounders))), columns=confounders))
    st.write(f"- Doubly Robust Learner: Rp {ate_dr:,.0f}")
    st.write(f"- Threshold bisnis (break-even uplift per customer): Rp {threshold_value:,.0f}")

    # ================================
    # Generate Synthetic Data (1000 customer)
    # ================================
    n = 1000
    np.random.seed(42)
    synthetic_data_raw = pd.DataFrame({
        "income": np.random.randint(3_000_000, 25_000_000, n),
        "age": np.random.randint(20, 60, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "maritalstatus": np.random.choice(["Single", "Married", "Widowed"], n),
        "city": np.random.choice(["Jakarta", "Surabaya", "Denpasar", "Medan", "Makassar"], n),
        "period": np.random.choice(["Before", "After"], n),
        "webpurchases": np.random.randint(0, 10, n),
        "storepurchases": np.random.randint(0, 10, n),
        "catalogpurchases": np.random.randint(0, 5, n),
        "mntfood": np.random.randint(100_000, 3_000_000, n),
        "mntfashion": np.random.randint(100_000, 3_000_000, n),
        "mntelectronics": np.random.randint(100_000, 3_000_000, n)
    })

    synthetic_encoded = pd.get_dummies(synthetic_data_raw, columns=categorical_cols, drop_first=True)
    synthetic_aligned = synthetic_encoded.reindex(columns=confounders, fill_value=0)
    synthetic_aligned[numeric_conf] = scaler.transform(synthetic_aligned[numeric_conf])

    # Prediksi untuk synthetic data
    results_synthetic = causal_forest.effect(synthetic_aligned)
    results_synthetic_df = synthetic_data_raw.copy()
    results_synthetic_df["Predicted_ITE"] = results_synthetic
    results_synthetic_df["Status"] = np.where(results_synthetic_df["Predicted_ITE"] > threshold_value, "On Target", "Off Target")

    # ================================
    # ROI Simulator (pakai synthetic data)
    # ================================
    st.subheader("ROI Simulator")
    cost = st.number_input("Total Campaign Cost (Rp)", 1_000_000_000, 10_000_000_000, 2_000_000_000, step=500_000_000)

    uplift_total = np.sum(causal_forest.effect(df_encoded[confounders]))
    roi = (uplift_total - cost) / cost


    st.write(f"Total Uplift (On Target only): Rp {uplift_total:,.0f}")
    st.write(f"ROI: {roi:.2f}")

    # ================================
    # Batch Scoring Summary
    # ================================
    st.subheader("Batch Scoring Synthetic 1000 Customer")
    st.write(results_synthetic_df["Status"].value_counts(normalize=True) * 100)

    # Pie chart
    summary_counts = results_synthetic_df["Status"].value_counts()
    colors = ['lightgreen' if label == 'On Target' else 'salmon' for label in summary_counts.index]
    col1, col2 = st.columns([1.5,2])  # rasio kolom
    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie(summary_counts, labels=summary_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize':8})
        ax.set_title("Proporsi On Target vs Off Target", fontsize=10)
        st.pyplot(fig)

    with col2:
        st.write("📊 Ringkasan Prediksi:")
        st.write(results_synthetic_df["Status"].value_counts(normalize=True) * 100)
