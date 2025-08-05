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
from utils.report_data import REPORT

st.set_page_config(page_title="BoinkVision: Elegant Space Asset Monitor", page_icon="assets/images/logo.png", layout="wide")

# Load elegant CSS
with open("assets/css/elegant_dark.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

elegant_sidebar()

MODEL_PATH = "model/best.pt"

@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = None
if os.path.exists(MODEL_PATH):
    model = load_yolo_model(MODEL_PATH)
else:
    st.error("Model file could not be loaded. Please check the file path: model/best.pt.")

if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    if elegant_landing():
        st.session_state.page = "main"
        st.rerun()
elif st.session_state.page == "main":
    tabs = st.tabs(["Detection", "Analytics", "Full Report", "Docs & Team"])

    # --- Detection Tab ---
    with tabs[0]:
        page_heading("Asset Detection & Visualization")

        elegant_section(
            "How it works",
            "Upload an image from ISS or industrial environments. Adjust detection confidence and run inference to see precise, real-time asset detection."
        )

        uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
        confidence = st.slider("Detection Confidence", 0.2, 0.95, 0.4, 0.01, 
                            help="Adjust the threshold for prediction confidence.")

        run_ready = uploaded_file is not None and model is not None
        run_button = st.button("Run Detection", disabled=not run_ready, 
                            help="Run the detection model on the uploaded image.")

        section_close()

        if run_button and run_ready:
            image = Image.open(uploaded_file)
            st.markdown("---")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True, caption="Your uploaded input")

            with col2:
                st.subheader("Detected Assets")
                with st.spinner("Running detection..."):
                    results = model.predict(image, conf=confidence)
                    annotated = results[0].plot()
                    st.image(annotated, use_column_width=True, caption="Model predictions")

                st.markdown("---")
                st.subheader("Detection Summary")

                detected_counts = {}
                names = model.names
                for r in results:
                    for c in r.boxes.cls:
                        detected_counts[names[int(c)]] = detected_counts.get(names[int(c)], 0) + 1

                if detected_counts:
                    metric_cols = st.columns(len(detected_counts))
                    for idx, (name, count) in enumerate(detected_counts.items()):
                        with metric_cols[idx]:
                            st.metric(label=name.title(), value=count)
                else:
                    st.info("No assets detected at the current confidence threshold.")

            if st.button("Reset Detection"):  # Reset button below images
                st.experimental_rerun()

        elif uploaded_file and not model:
            st.error("Model is not loaded. Please wait and retry.")
        else:
            st.info("Upload an image to begin detection.")

    # --- Analytics Tab ---
    with tabs[1]:
        page_heading("Comprehensive Analytics & Visual Explanation")
        elegant_section("Performance Analysis",
            "This section provides a clear and structured overview of our YOLOv8x model's performance, based on the Duality AI Falcon synthetic dataset. "
            "Each chart is explained for quick interpretation. Hover on bars for exact values!")

        st.subheader("Model Performance Comparison")
        st.markdown("""
        **Explanation:**  
        This grouped bar chart compares the baseline and fine-tuned YOLOv8x models on four critical metrics: Precision, Recall, mAP@0.5, and mAP@0.5-0.95. The tuned model demonstrates significant improvements in all metrics, validating the effectiveness of our training pipeline.
        """)
        model_performance_comparison_chart(key="analytics-metrics")
        st.markdown("**Conclusion:**  Tuned model outperforms the baseline in every metric, ensuring higher reliability for ISS safety tasks.")

        st.subheader("Classwise Precision Comparison")
        st.markdown("""
        **Explanation:**  
        This plot visualizes the precision for each object class, showing substantial improvement with the tuned model—particularly for 'Toolbox' and 'Oxygen Tank', where precision reaches perfection.
        """)
        classwise_precision_comparison_chart(key="analytics-class-precision")
        st.markdown("**Conclusion:**  Near-perfect classwise precision dramatically reduces false positives for all safety objects.")

        st.subheader("Classwise Recall Comparison")
        st.markdown("""
        **Explanation:**  
        Recall measures the model’s ability to detect all instances of each object class. The tuned model has fewer missed detections for critical ISS assets.
        """)
        classwise_recall_comparison_chart(key="analytics-class-recall")
        st.markdown("**Conclusion:**  Enhanced recall guarantees even rare or partially visible safety assets are reliably detected.")

        st.subheader("Classwise mAP@0.5 Comparison")
        st.markdown("""
        **Explanation:**  
        mAP@0.5 shows detection accuracy at a moderate threshold for each class.
        """)
        classwise_map50_comparison_chart(key="analytics-class-map50")
        st.markdown("**Conclusion:**  Consistent detection quality across object types confirms tuned model robustness.")

        st.subheader("Classwise mAP@0.5:0.95 Comparison")
        st.markdown("""
        **Explanation:**  
        The strictest metric, averaged over high IoU thresholds.
        """)
        classwise_map50_95_comparison_chart(key="analytics-class-map95")
        st.markdown("**Conclusion:**  Tuned model remains highly accurate under strict overlap criteria.")

        st.subheader("Confusion Matrix of Tuned Model")
        st.markdown("""
        **Explanation:**  
        The confusion matrix quantifies classification accuracy and errors. Strong diagonal values = correct identification, off-diagonal = rare misclassifications.
        """)
        z = [[160, 0, 2], [0, 172, 0], [0, 0, 162]]
        class_names = ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
        import plotly.graph_objs as go
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title='Count')
        ))
        fig.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            template="plotly_white",
            width=700,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="analytics-cm")
        st.markdown("**Conclusion:** The matrix confirms extremely low misclassification rates, evidencing robust discrimination between asset types.")

        section_close()
        st.markdown("""
        ---
        **Summary:**  
        These visual analyses collectively show that our tuned YOLOv8x model meets and exceeds real-world ISS asset tracking requirements, with measurable improvement over the baseline and minimal risk of critical errors.
        """)

    # --- Full Report Tab ---
    def render_full_report():
        st.header("Detailed Project Report")

        st.subheader("Project Summary")
        st.markdown("""
        This project focuses on developing a highly accurate object detection system 
        using the YOLOv8 deep learning model to identify critical safety equipment—
        Fire Extinguishers, Tool Boxes, Oxygen Tanks—in images. Utilizing a custom-curated dataset,
        the model was fine-tuned with advanced augmentations and hyperparameter tuning to achieve robust performance.
        """)

        st.subheader("Methodology")
        st.markdown("""
        - Dataset preparation and mounting in Colab environment.
        - Custom YAML dataset configuration specifying classes and paths.
        - YOLOv8x model initialized with pretrained checkpoint.
        - Fine-tuned for 100 epochs using data augmentations like mosaic, mixup, HSV shifts.
        - Early stopping to prevent overfitting.
        """)

        st.subheader("Test Results & Evaluation Metrics")
        st.markdown("""
        - Fitness: 0.9549
        - Precision: 0.9892
        - Recall: 0.9567
        - mAP50: 0.9770
        - mAP50-95: 0.9524

        The model demonstrates outstanding accuracy and operational efficiency.
        """)

        st.subheader("Challenges")
        st.markdown("""
        - Overfitting detected after certain epochs.
        - Addressed by tuning augmentations (reduced mosaic, controlled mixup, HSV adjustments)
        - Used early stopping with patience to save the best model.
        """)

        st.subheader("Conclusion")
        st.markdown("""
        The tuned YOLOv8x model significantly outperforms the baseline,
        yielding near-perfect precision and recall, stable training, and robust detection in varied conditions.
        """)

        st.subheader("Future Work")
        st.markdown("""
        - Expand augmentation pipeline (zoom, flipping, blurring, cropping).
        - Improve robustness under extreme conditions.
        - Broaden detected classes to include PPE and other assets.
        """)

        st.subheader("Use Case Application")
        st.markdown("""
        An automated industrial safety and inventory monitoring system using this detection model can:
        - Provide real-time alerts for missing/misplaced safety equipment.
        - Optimize operational efficiency by tracking critical tools.
        - Scale across multiple facilities and integrate with existing management software.
        """)

    with tabs[2]:
        render_full_report()

    # --- Docs & Team Tab ---
    with tabs[3]:
        st.header("Documentation & Team")

        st.subheader("Project Overview")
        st.markdown("""
        BoinkVision AI is a cutting-edge object detection system designed for the ISS, focusing on key safety assets like Fire Extinguishers, Toolboxes, and Oxygen Tanks.
        The model is based on YOLOv8x, fine-tuned on synthetic datasets produced by Duality AI’s Falcon digital twin platform, ensuring robust accuracy under challenging orbital conditions.
        """)

        st.subheader("Model & Training")
        st.markdown("""
        - Base Architecture: YOLOv8x (Ultralytics implementation)
        - Dataset: Duality AI Falcon synthetic images
        - Training: 100 epochs with advanced augmentations including mosaic, mixup, HSV shifts, random scaling and translation
        - Optimization: AdamW optimizer with low learning rate and early stopping to prevent overfitting
        - Evaluation: Achieved precision 0.989, recall 0.957, mAP50 0.977 on test set
        """)

        st.subheader("Retraining with Falcon")
        st.markdown("""
        The model is designed to stay up-to-date via continuous retraining with newly generated Falcon synthetic data.

        Steps to retrain:
        1. Download the latest synthetic dataset from the [Duality Falcon platform](https://falcon.duality.ai/).
        2. Configure the dataset paths in the YAML config file.
        3. Run the Ultralytics YOLOv8 training pipeline with updated data.
        4. Validate performance and export new model weights.
        5. Update deployment with retrained weights for continual improvement.
        """)

        st.subheader("Software Environment & Dependencies")
        st.markdown("""
        - Python 3.9+
        - Streamlit (for UI)
        - Ultralytics YOLOv8  
        - PIL (Pillow)  
        - Requests  
        - Pandas, NumPy (for analytics)  
        - Plotly (for charts)  
        """)

        st.subheader("Team & Contact")
        st.markdown("""
        **OINK BOINK Team**  
        - Namit Rana – Lead AI Engineer  
        - Khushboo Kataria – Machine Learning Specialist  

        For questions, collaboration, or mentorship:  
        - Email: your_team_email@example.com  
        - GitHub: [YourGitHubProfile](https://github.com/YourGitHubProfile)  
        - LinkedIn: [YourLinkedInProfile](https://linkedin.com/in/YourLinkedInProfile)
        """)

        st.markdown("---")
        st.caption("BoinkVision AI\nDuality AI Space Station Hackathon, 2025")
        section_close()
