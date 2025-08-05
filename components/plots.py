import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

from components.layout import elegant_landing, page_heading, elegant_section, section_close
from components.sidebar import elegant_sidebar
from components.plots import (
    model_performance_comparison_chart,
    classwise_precision_comparison_chart,
    classwise_recall_comparison_chart,
    classwise_map50_comparison_chart,
    classwise_map50_95_comparison_chart
)

# Set up layout
st.set_page_config(page_title="YOLOv8 Model Comparison", layout="wide")
elegant_sidebar()
elegant_landing()
page_heading("🔬 Model Evaluation Dashboard")

# Section 1: Overall Performance Metrics
elegant_section("📊 Overall Model Performance")
model_performance_comparison_chart(
    baseline=[0.881, 0.739, 0.819, 0.684],
    tuned=[0.925, 0.852, 0.914, 0.867],
    metrics=["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
)
section_close()

# Section 2: Classwise Precision
elegant_section("🎯 Classwise Precision")
classwise_precision_comparison_chart(
    baseline=[0.931, 0.785, 0.898],
    tuned=[0.944, 0.879, 0.950],
    classes=["Astronaut", "Rocket", "Planet"]
)
section_close()

# Section 3: Classwise Recall
elegant_section("🔁 Classwise Recall")
classwise_recall_comparison_chart(
    baseline=[0.698, 0.720, 0.850],
    tuned=[0.880, 0.837, 0.838],
    classes=["Astronaut", "Rocket", "Planet"]
)
section_close()

# Section 4: Classwise mAP@0.5
elegant_section("📍 Classwise mAP@0.5")
classwise_map50_comparison_chart(
    baseline=[0.931, 0.785, 0.848],
    tuned=[0.928, 0.902, 0.912],
    classes=["Astronaut", "Rocket", "Planet"]
)
section_close()

# Section 5: Classwise mAP@0.5:0.95
elegant_section("📐 Classwise mAP@0.5:0.95")
classwise_map50_95_comparison_chart(
    baseline=[0.780, 0.740, 0.710],
    tuned=[0.898, 0.848, 0.854],
    classes=["Astronaut", "Rocket", "Planet"]
)
section_close()

# End of the report
st.markdown("---")
st.success("🎉 Congratulations! You've successfully visualized your YOLOv8 model's performance.")
