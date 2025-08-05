import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def model_names(model):
    try:
        return model.names
    except:
        return ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
