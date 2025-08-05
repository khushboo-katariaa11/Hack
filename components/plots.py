import streamlit as st
import plotly.graph_objs as go

def model_performance_comparison_chart(key=None):
    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    baseline = [0.881, 0.739, 0.819, 0.684]
    tuned = [0.925, 0.852, 0.914, 0.867]  # updated from val24
    fig = go.Figure(data=[
        go.Bar(name="Baseline Model", x=metrics, y=baseline, marker_color="#8EA7E9"),
        go.Bar(name="Tuned Model", x=metrics, y=tuned, marker_color="#0984e3"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0.6, 1.0], tickformat=".2%"),
        template="plotly_white",
        title="Model Performance Comparison"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def classwise_precision_comparison_chart(key=None):
    class_names = ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
    baseline = [0.731, 0.685, 0.698]
    tuned = [0.944, 0.879, 0.95]  # from test results

    fig = go.Figure(data=[
        go.Bar(name="Baseline Model", x=class_names, y=baseline, marker_color="#8EA7E9"),
        go.Bar(name="Tuned Model", x=class_names, y=tuned, marker_color="#0984e3"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Precision", range=[0.7, 1.01], tickformat=".2%"),
        template="plotly_white",
        title="Classwise Precision: Baseline vs Tuned"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def classwise_recall_comparison_chart(key=None):
    class_names = ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
    baseline = [0.698, 0.72, 0.65]
    tuned = [0.88, 0.837, 0.838]

    fig = go.Figure(data=[
        go.Bar(name="Baseline Model", x=class_names, y=baseline, marker_color="#8EA7E9"),
        go.Bar(name="Tuned Model", x=class_names, y=tuned, marker_color="#0984e3"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Recall", range=[0.6, 1.0], tickformat=".2%"),
        template="plotly_white",
        title="Classwise Recall: Baseline vs Tuned"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def classwise_map50_comparison_chart(key=None):
    class_names = ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
    baseline = [0.731, 0.785, 0.848]
    tuned = [0.928, 0.902, 0.912]

    fig = go.Figure(data=[
        go.Bar(name="Baseline Model", x=class_names, y=baseline, marker_color="#8EA7E9"),
        go.Bar(name="Tuned Model", x=class_names, y=tuned, marker_color="#0984e3"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="mAP50", range=[0.7, 1.0], tickformat=".2%"),
        template="plotly_white",
        title="Classwise mAP@0.5: Baseline vs Tuned"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def classwise_map50_95_comparison_chart(key=None):
    class_names = ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
    baseline = [0.78, 0.74, 0.71]
    tuned = [0.898, 0.848, 0.854]

    fig = go.Figure(data=[
        go.Bar(name="Baseline Model", x=class_names, y=baseline, marker_color="#8EA7E9"),
        go.Bar(name="Tuned Model", x=class_names, y=tuned, marker_color="#0984e3"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="mAP50-95", range=[0.65, 1.0], tickformat=".2%"),
        template="plotly_white",
        title="Classwise mAP@0.5:0.95: Baseline vs Tuned"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
