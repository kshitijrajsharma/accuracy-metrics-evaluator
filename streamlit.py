import json
import os
import tempfile
from datetime import datetime

import streamlit as st
from evaluate import analyze_predictions

st.set_page_config(page_title="Prediction Evaluation Tool")


def save_uploaded_file(uploadedfile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp_file:
        tmp_file.write(uploadedfile.getvalue())
        return tmp_file.name


def main():
    st.title("GeoJSON Prediction Evaluation Tool")

    st.markdown(
        """
    ### Instructions:
    1. Upload your prediction GeoJSON file
    2. Upload your ground truth (labels) GeoJSON file
    3. Adjust the overlap threshold if needed
    4. Click 'Run Analysis' to get results
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Predictions")
        pred_file = st.file_uploader("Choose prediction GeoJSON file", type=["geojson"])

    with col2:
        st.subheader("Upload Ground Truth")
        truth_file = st.file_uploader(
            "Choose ground truth GeoJSON file", type=["geojson"]
        )

    overlap_threshold = st.slider(
        "Overlap Threshold for determining TP (%)", 0, 100, 10, 5
    )

    if pred_file and truth_file:
        if st.button("Run Analysis"):
            with st.spinner("Analyzing predictions..."):
                try:
                    pred_path = save_uploaded_file(pred_file)
                    truth_path = save_uploaded_file(truth_file)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_geojson = f"predictions_analyzed_{timestamp}.geojson"
                    output_metrics = f"analysis_metrics_{timestamp}.json"

                    analyzed_predictions, metrics = analyze_predictions(
                        pred_path, truth_path, overlap_threshold=overlap_threshold
                    )

                    analyzed_predictions.to_file(output_geojson, driver="GeoJSON")
                    with open(output_metrics, "w") as f:
                        json.dump(metrics, f, indent=2)

                    st.success("Analysis completed successfully!")

                    metrics_tab, data_tab = st.tabs(["Metrics", "Detailed Data"])

                    with metrics_tab:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                            st.metric("True Positives", metrics["true_positives"])
                            st.metric("False Positives", metrics["false_positives"])

                        with col2:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                            st.metric("False Negatives", metrics["false_negatives"])
                            st.metric("Average IoU", f"{metrics['average_iou']:.3f}")

                        with col3:
                            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                            st.metric(
                                "Average Overlap %",
                                f"{metrics['average_overlap_percentage']:.1f}%",
                            )
                            st.metric("Total Predictions", metrics["total_predictions"])

                        st.json(metrics)

                    with data_tab:
                        st.dataframe(analyzed_predictions)

                    col1, col2 = st.columns(2)
                    with col1:
                        with open(output_geojson, "rb") as f:
                            st.download_button(
                                "Download Analyzed Predictions",
                                f,
                                file_name=output_geojson,
                                mime="application/json",
                            )

                    with col2:
                        with open(output_metrics, "rb") as f:
                            st.download_button(
                                "Download Metrics JSON",
                                f,
                                file_name=output_metrics,
                                mime="application/json",
                            )

                    os.unlink(pred_path)
                    os.unlink(truth_path)
                    os.unlink(output_geojson)
                    os.unlink(output_metrics)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")


if __name__ == "__main__":
    main()
