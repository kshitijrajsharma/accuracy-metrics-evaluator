# GeoJSON Prediction Evaluation Tool

A tool for comparing predicted GeoJSON features against ground truth data, calculating various accuracy metrics and overlap statistics. This tool is particularly useful for evaluating the performance of geographic object detection and segmentation models.

## Features

- Upload and compare prediction and ground truth GeoJSON files
- Calculate comprehensive evaluation metrics
- Interactive web interface for easy use
- Downloadable results in GeoJSON and JSON formats
- Detailed visualization of metrics and results

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install streamlit geopandas pandas numpy
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Launch the Streamlit app
2. Upload your prediction GeoJSON file
3. Upload your ground truth (labels) GeoJSON file
4. Adjust the IoU threshold if needed (default: 0.5)
5. Click "Run Analysis"
6. View results and download analyzed data

## Input Format

Both prediction and ground truth files should be in GeoJSON format containing geometry features.

## Output Files

1. `predictions_analyzed_{timestamp}.geojson`: Original predictions with added evaluation attributes
2. `analysis_metrics_{timestamp}.json`: Comprehensive metrics in JSON format

## Metrics Explanation

### Core Metrics

- **Precision**: Percentage of correct predictions among all predictions made
  - Formula: True Positives / (True Positives + False Positives)
  - Range: 0 to 1 (higher is better)

- **Recall**: Percentage of ground truth objects that were correctly detected
  - Formula: True Positives / (True Positives + False Negatives)
  - Range: 0 to 1 (higher is better)

- **F1 Score**: Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Range: 0 to 1 (higher is better)

### Classification Categories

- **True Positive (TP)**: Prediction that correctly matches ground truth above the IoU threshold
  - **TP_OVERSIZED**: True positive but prediction is >150% of ground truth size
  - **TP_UNDERSIZED**: True positive but prediction is <67% of ground truth size
- **False Positive (FP)**: Prediction that doesn't match any ground truth object
- **False Negative (FN)**: Ground truth object that wasn't detected

### Overlap Metrics

- **IoU Score**: Intersection over Union for each prediction
  - Formula: Area of Intersection / Area of Union
  - Range: 0 to 1 (higher indicates better overlap)

- **Overlap Percentage**: How much of the ground truth is covered by the prediction
  - Formula: (Intersection Area / Ground Truth Area) * 100
  - Can be >100% if prediction is larger than ground truth

- **Size Ratio**: Relative size of prediction compared to ground truth
  - Formula: Prediction Area / Ground Truth Area
  - 1.0 indicates perfect size match
  - >1.0 indicates oversized prediction
  - <1.0 indicates undersized prediction

### Summary Statistics

- **Average Overlap Percentage**: Mean overlap across all true positive predictions
- **Average Size Ratio**: Mean size ratio across all true positive predictions
- **Total Predictions**: Number of predicted objects
- **Total Ground Truth**: Number of ground truth objects
- **True Positives Count**: Number of correct predictions
- **False Positives Count**: Number of incorrect predictions
- **False Negatives Count**: Number of missed ground truth objects

## Size Classification Thresholds

- Oversized: Size ratio > 1.5 (150% of ground truth)
- Undersized: Size ratio < 0.67 (67% of ground truth)
- Good size: Size ratio between 0.67 and 1.5
