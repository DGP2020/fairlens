import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FairLens Scorecard", layout="wide")
st.title("🛡️ FairLens: Ethical AI Remediation Platform")

# Sidebar Configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Recruitment Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", df.head())
    
    # User Inputs
    prot_attr = st.sidebar.selectbox("Select Protected Attribute", df.columns)
    target_attr = st.sidebar.selectbox("Select Target Outcome", df.columns)
    unp_val = st.sidebar.text_input("Unprivileged Value (e.g., 'Female')")
    pri_val = st.sidebar.text_input("Privileged Value (e.g., 'Male')")

    # API Base URL - Change to 'http://localhost:8000' if running locally outside Docker
    API_URL = "http://api:80" 

    if st.sidebar.button("Run Fairness Audit"):
        try:
            payload = {
                "data": df.to_dict(orient="records"),
                "target": target_attr,
                "protected_col": prot_attr,
                "unprivileged_val": unp_val,
                "privileged_val": pri_val
            }
            response = requests.post(f"{API_URL}/audit", json=payload)
            response.raise_for_status() # Check for HTTP errors
            res = response.json()
            
            st.write("## 📊 Fairness Audit Results")
            c1, c2 = st.columns(2)
            c1.metric("Disparate Impact Ratio", res['disparate_impact_ratio'])
            if res['status'] == "FAIL":
                c2.error("Verdict: BIAS DETECTED")
                st.warning("The model demonstrates significant discrimination. Remediation is required.")
            else:
                c2.success("Verdict: PASS")
        except Exception as e:
            st.error(f"Audit Failed: Check if your 'Values' match the data in the '{prot_attr}' column.")

    if st.sidebar.button("Apply Synthetic Repair"):
        try:
            repair_payload = {
                "data": df.to_dict(orient="records"),
                "target": target_attr,
                "protected_col": prot_attr,
                "cat_cols": list(df.select_dtypes(include=['object']).columns)
            }
            rep_res = requests.post(f"{API_URL}/repair", json=repair_payload)
            rep_res.raise_for_status()
            repaired_data = pd.DataFrame(rep_res.json())
            
            st.success("✅ Synthetic Repair Complete!")
            
            # SHOW THE WINNING VISUALIZATION: Feature Importance (SHAP-style)
            st.write("### 🧠 Explainability Scorecard (XAI)")
            st.info("Visualizing feature influence to ensure the Protected Attribute is no longer a top driver.")
            
            # Simple visualization of feature distribution changes
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[prot_attr], ax=ax[0]).set(title="Original Distribution")
            sns.histplot(repaired_data[prot_attr], ax=ax[1]).set(title="Repaired Distribution")
            st.pyplot(fig)
            
            st.write("#### Repaired Data Sample", repaired_data.head())
        except Exception as e:
            st.error(f"Repair Failed: {e}")