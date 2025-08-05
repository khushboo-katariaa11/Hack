import streamlit as st
import plotly.graph_objs as go

def model_performance_comparison_chart(key=None):
    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    baseline = [0.881, 0.739, 0.819, 0.684]
    tuned = [0.9892, 0.9567, 0.977, 0.9524]
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
    baseline = [0.931, 0.785, 0.898]
    tuned = [1.0, 1.0, 0.968]
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
    baseline = [0.698, 0.72, 0.85]
    tuned = [1.0, 0.9567, 0.95]
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
    baseline = [0.931, 0.785, 0.848]
    tuned = [1.0, 1.0, 0.977]
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
    tuned = [0.95, 0.96, 0.95]
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
