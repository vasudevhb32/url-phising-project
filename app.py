import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Phishing URL Classifier", page_icon="🛡️")
bundle = joblib.load("model.joblib")
model, FEATURES = bundle["model"], bundle["features"]

st.title("🛡️ Phishing URL Behaviour Classifier")
st.caption("Educational prototype (Logistic Regression). Enter the URL's feature values as encoded in the dataset.")

# label, kind ("bin" = 0/1 flag, "int" = integer), help text
SPEC = {
    "Have_IP": ("URL uses an IP address", "bin"),
    "Have_At": ("URL contains '@'", "bin"),
    "URL_Length": ("URL length flag (1 = long, 0 = short)", "bin"),
    "URL_Depth": ("Number of '/' path levels (0-20)", "int"),
    "Redirection": ("'//' redirection in path", "bin"),
    "https_Domain": ("'https' token inside domain name", "bin"),
    "TinyURL": ("Shortening service used", "bin"),
    "Prefix/Suffix": ("'-' in domain name", "bin"),
    "DNS_Record": ("No DNS record found", "bin"),
    "Web_Traffic": ("Low web traffic / rank", "bin"),
    "Domain_Age": ("Domain younger than 12 months", "bin"),
    "Domain_End": ("Domain expires within 6 months", "bin"),
    "iFrame": ("Invisible iFrame present", "bin"),
    "Mouse_Over": ("Mouse-over changes status bar", "bin"),
    "Right_Click": ("Right-click behaviour flag", "bin"),
    "Web_Forwards": ("Multiple page forwards", "bin"),
}

# defaults = median legitimate row from the training data
DEFAULT_ONE = {"URL_Length", "Web_Traffic", "Domain_Age", "Domain_End", "Right_Click"}
values = {}
cols = st.columns(2)
for i, f in enumerate(FEATURES):
    label, kind = SPEC[f]
    with cols[i % 2]:
        if kind == "bin":
            values[f] = st.selectbox(f"{f} — {label}", [0, 1], index=1 if f in DEFAULT_ONE else 0, key=f)
        else:
            values[f] = st.number_input(f"{f} — {label}", min_value=0, max_value=20, value=3, step=1, key=f)

if st.button("Classify", type="primary"):
    row = pd.DataFrame([values])[FEATURES]
    if row.isna().any().any():
        st.error("Please fill in all fields.")
    else:
        p = float(model.predict_proba(row)[0, 1])  # same scaler + model as training
        risk = "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
        (st.error if p >= 0.5 else st.success)(
            f"Prediction: {'PHISHING / suspicious' if p >= 0.5 else 'LEGITIMATE'}")
        st.metric("Phishing probability", f"{p:.1%}")
        st.write(f"Risk level: **{risk}**")
