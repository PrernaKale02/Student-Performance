import streamlit as st
import numpy as np
import joblib
import json
import h5py
from tensorflow import keras

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #e3f2fd;
        color: #1a237e;
    }
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    .stApp,
    .stApp p,
    .stApp span,
    .stApp div,
    .stApp label,
    .stApp li {
        color: black;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .css-1d391kg {
        background-color: white !important;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 25px rgba(26, 35, 126, 0.08);
        border: 1px solid #bbdefb;
    }
    h1, h2, h3 {
        color: #0d47a1 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    h4 {
        color: #1565c0 !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        background: linear-gradient(45deg, #1976d2, #42a5f5);
        color: black;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.3);
        color: black;
    }
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #bbdefb;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .stSlider > div > div > div > div {
        background-color: #1976d2 !important;
    }
    label {
        color: #0d47a1 !important;
        font-weight: 500 !important;
    }
    .stMarkdown p {
        color: #1a237e !important;
        font-size: 1.1rem;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    .stSelectbox * ,
    .stSlider * ,
    .stNumberInput * ,
    .stTextInput * ,
    .stSuccess {
        color: black !important;
    }
    [data-baseweb="select"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
def build_model_from_h5(path):
    with h5py.File(path, "r") as h5_file:
        model_config = h5_file.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

    config = json.loads(model_config)
    layers_config = config["config"]["layers"]
    model_layers = []
    input_shape = None

    for layer in layers_config:
        class_name = layer["class_name"]
        layer_config = layer["config"]

        if class_name == "InputLayer":
            batch_shape = layer_config.get("batch_shape") or layer_config.get("batch_input_shape")
            input_shape = tuple(batch_shape[1:])
        elif class_name == "Dense":
            if not model_layers and input_shape is not None:
                model_layers.append(keras.Input(shape=input_shape))
            model_layers.append(
                keras.layers.Dense(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    use_bias=layer_config["use_bias"],
                    name=layer_config.get("name"),
                )
            )

    model = keras.Sequential(model_layers, name=config["config"].get("name"))
    model.load_weights(path)
    return model


@st.cache_resource
def load_assets():
    reg_model = build_model_from_h5("regression_model.h5")
    class_model = build_model_from_h5("classifier_model.h5")
    scaler = joblib.load("scaler.pkl")
    return reg_model, class_model, scaler


reg_model, class_model, scaler = load_assets()

# --- HEADER (NO IMAGE) ---
st.title("🎓 Student Performance Predictor")
st.markdown("### Predict exam success with AI-driven insights.")
st.write("Unlock the potential of academic data to forecast grades and scores.")

st.divider()

# --- INPUT SECTION ---
st.subheader("📝 Enter Student Metrics")
col_obs1, col_obs2 = st.columns(2)

with col_obs1:
    st.markdown("#### 📖 Academic Profile")
    hours = st.slider("Hours Studied", 0, 50, 20, help="Total hours spent studying.")
    attendance = st.slider("Attendance (%)", 50, 100, 80)
    previous = st.slider("Previous Score", 40, 100, 70)
    tutoring = st.slider("Tutoring Sessions", 0, 10, 2)

with col_obs2:
    st.markdown("#### 🏃 Lifestyle & Habits")
    sleep = st.slider("Sleep Hours", 4, 10, 7)
    physical = st.slider("Physical Activity", 0, 6, 3)
    extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])

extra_val = 1 if extra == "Yes" else 0

# --- PREDICTION LOGIC ---
input_data = np.array([[hours, attendance, sleep, previous, tutoring, physical, extra_val]])
scaled = scaler.transform(input_data)

st.write("") # Spacer
if st.button("🚀 Predict Performance", use_container_width=True):
    with st.spinner("Analyzing data..."):
        score_pred = float(reg_model.predict(scaled, verbose=0)[0][0])
        grade_probs = class_model.predict(scaled, verbose=0)[0]
        grade_index = int(np.argmax(grade_probs))
        final_status = "Fail" if grade_index == 0 else "Pass"
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Estimated Exam Score", value=f"{round(score_pred, 2)} / 100")
            
        with res_col2:
            st.metric(label="Final Grade Category", value=final_status)
        
        # Add a progress bar for the score
        st.progress(min(max(int(score_pred), 0), 100))
        st.caption(f"Confidence: {grade_probs[grade_index] * 100:.1f}%")
        
        st.success("Analysis complete! Keep up the great work!")

# --- FOOTER ---
st.markdown("<br><hr><center>Advanced Student Analytics</center>", unsafe_allow_html=True)
