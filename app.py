import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="centered"
)

# Load the saved model and scaler
@st.cache_resource
def load_artifacts():
    with open('parkinson_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")

st.title("🧠 Parkinson's Disease Detection System")
st.write("This system utilizes an optimized Support Vector Machine (SVM) classifier to predict the likelihood of Parkinson's Disease based on vocal metrics.")

st.info("Please enter the clinical biomedical voice measurements below:")

# Organized input layout using columns
col1, col2, col3 = st.columns(3)

with col1:
    fo = st.number_input("MDVP:Fo (Hz)", min_value=0.0, value=120.0, step=0.1, help="Average vocal fundamental frequency")
    fhi = st.number_input("MDVP:Fhi (Hz)", min_value=0.0, value=150.0, step=0.1, help="Maximum vocal fundamental frequency")
    flo = st.number_input("MDVP:Flo (Hz)", min_value=0.0, value=100.0, step=0.1, help="Minimum vocal fundamental frequency")
    jitter = st.number_input("MDVP:Jitter (%)", min_value=0.0, format="%.5f", value=0.0078, step=0.0001)
    jitter_abs = st.number_input("MDVP:Jitter (Abs)", min_value=0.0, format="%.5f", value=0.00007, step=0.00001)
    rap = st.number_input("MDVP:RAP", min_value=0.0, format="%.5f", value=0.0037, step=0.0001)
    ppq = st.number_input("MDVP:PPQ", min_value=0.0, format="%.5f", value=0.0055, step=0.0001)
    ddp = st.number_input("Jitter:DDP", min_value=0.0, format="%.5f", value=0.0110, step=0.0001)

with col2:
    shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.5f", value=0.0437, step=0.0001)
    shimmer_db = st.number_input("MDVP:Shimmer (dB)", min_value=0.0, value=0.426, step=0.01)
    apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, format="%.5f", value=0.0205, step=0.0001)
    apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, format="%.5f", value=0.0270, step=0.0001)
    apq = st.number_input("MDVP:APQ", min_value=0.0, format="%.5f", value=0.0297, step=0.0001)
    dda = st.number_input("Shimmer:DDA", min_value=0.0, format="%.5f", value=0.0654, step=0.0001)
    nhr = st.number_input("NHR", min_value=0.0, format="%.5f", value=0.0221, step=0.0001)

with col3:
    hnr = st.number_input("HNR", min_value=0.0, value=21.0, step=0.1)
    rpde = st.number_input("RPDE", min_value=0.0, value=0.414, step=0.01)
    dfa = st.number_input("DFA", min_value=0.0, value=0.815, step=0.01)
    spread1 = st.number_input("spread1", value=-4.81, step=0.01)
    spread2 = st.number_input("spread2", value=0.266, step=0.01)
    d2 = st.number_input("D2", min_value=0.0, value=2.30, step=0.01)
    ppe = st.number_input("PPE", min_value=0.0, value=0.284, step=0.01)

# Process inputs for prediction
input_features = np.array([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp,
                            shimmer, shimmer_db, apq3, apq5, apq, dda,
                            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

if st.button("Run Diagnostic Prediction", type="primary"):
    try:
        # 1. Scale the input using the loaded training scaler
        scaled_features = scaler.transform(input_features)
        
        # 2. Generate Prediction
        prediction = model.predict(scaled_features)[0]
        
        st.write("--- ")
        if prediction == 1:
            st.error("### ⚠️ Diagnosis Result: Parkinson's Positive (Class 1)")
            st.write("The model predicted clinical indices indicating positive status. Please consult with a healthcare professional.")
        else:
            st.success("### ✅ Diagnosis Result: Healthy / Parkinson's Negative (Class 0)")
            st.write("The model predicted features aligned within normal parameters.")
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")