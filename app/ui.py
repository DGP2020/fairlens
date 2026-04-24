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
    # Basic data cleaning: handle missing values which cause API 422 errors
    df_clean = df.fillna("Unknown")
    st.write("### Raw Data Preview", df.head())
    
    # User Inputs
    prot_attr = st.sidebar.selectbox("Select Protected Attribute", df.columns)
    target_attr = st.sidebar.selectbox("Select Target Outcome", df.columns)
    
    # Suggest positive outcome value
    unique_outcomes = df[target_attr].unique()
    pos_val = st.sidebar.selectbox("Select Positive Outcome Value", unique_outcomes)
    
    unp_val = st.sidebar.text_input("Unprivileged Value (e.g., 'Female')")
    pri_val = st.sidebar.text_input("Privileged Value (e.g., 'Male')")

    # API Base URL
    API_URL = "http://api:80" 

    if st.sidebar.button("Run Fairness Audit"):
        try:
            payload = {
                "data": df_clean.to_dict(orient="records"),
                "target": target_attr,
                "protected_col": prot_attr,
                "unprivileged_val": unp_val,
                "privileged_val": pri_val,
                "positive_val": str(pos_val)
            }
            response = requests.post(f"{API_URL}/audit", json=payload)
            if response.status_code != 200:
                st.error(f"Audit API Error: {response.json().get('detail', 'Unknown error')}")
            else:
                res = response.json()
                st.write("## 📊 Fairness Audit Results")
                c1, c2 = st.columns(2)
                c1.metric("Disparate Impact Ratio", res['disparate_impact_ratio'])
                if res['status'] == "FAIL":
                    c2.error("Verdict: BIAS DETECTED")
                else:
                    c2.success("Verdict: PASS")
                
                # RE-ADD Proxy Check
                proxies_res = requests.post(f"{API_URL}/proxy-check", json={"data": payload["data"], "protected_col": prot_attr})
                if proxies_res.status_code == 200:
                    st.write("### 🔍 Proxy Variable Detection")
                    st.json(proxies_res.json()['proxies'])
        except Exception as e:
            st.error(f"Audit Connection Error: {e}")

    if st.sidebar.button("Apply Synthetic Repair"):
        try:
            repair_payload = {
                "data": df_clean.to_dict(orient="records"),
                "target": target_attr,
                "protected_col": prot_attr,
                "cat_cols": list(df_clean.select_dtypes(include=['object']).columns)
            }
            rep_res = requests.post(f"{API_URL}/repair", json=repair_payload)
            if rep_res.status_code != 200:
                st.error(f"Repair API Error: {rep_res.json().get('detail', 'Unknown error')}")
            else:
                repaired_data = pd.DataFrame(rep_res.json())
                st.success("✅ Synthetic Repair Complete!")
                
                st.write("### 🧠 Explainability Scorecard (XAI)")
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(df[prot_attr], ax=ax[0]).set(title="Original Distribution")
                sns.histplot(repaired_data[prot_attr], ax=ax[1]).set(title="Repaired Distribution")
                st.pyplot(fig)
                st.write("#### Repaired Data Sample", repaired_data.head())
        except Exception as e:
            st.error(f"Repair Connection Error: {e}")