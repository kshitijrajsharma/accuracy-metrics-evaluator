import json

import geopandas as gpd
import numpy as np


def calculate_overlap_metrics(pred_geom, truth_geom):
    """
    Calculate detailed overlap metrics between prediction and ground truth geometries
    """
    try:
        intersection_area = pred_geom.intersection(truth_geom).area
        pred_area = pred_geom.area
        truth_area = truth_geom.area

        overlap_pred = (intersection_area / pred_area) * 100 if pred_area > 0 else 0
        overlap_truth = (intersection_area / truth_area) * 100 if truth_area > 0 else 0

        size_ratio = pred_area / truth_area if truth_area > 0 else float("inf")

        return {
            "overlap_percentage": overlap_truth,
            "size_ratio": size_ratio,
            "intersection_area": intersection_area,
            "pred_area": pred_area,
            "truth_area": truth_area,
        }
    except:
        return {
            "overlap_percentage": 0,
            "size_ratio": 0,
            "intersection_area": 0,
            "pred_area": 0,
            "truth_area": 0,
        }


def calculate_iou(geom1, geom2):
    """
    Calculate Intersection over Union (IoU) between two geometries
    """
    try:
        intersection = geom1.intersection(geom2).area
        union = geom1.union(geom2).area
        iou = intersection / union if union > 0 else 0
        return iou
    except:
        return 0


def classify_prediction(pred_geom, truth_gdf, iou_threshold=0.5):
    """
    Classify a prediction geometry and calculate overlap metrics
    """
    ious = []
    overlap_metrics_list = []

    for truth_geom in truth_gdf.geometry:
        iou = calculate_iou(pred_geom, truth_geom)
        metrics = calculate_overlap_metrics(pred_geom, truth_geom)
        ious.append(iou)
        overlap_metrics_list.append(metrics)

    if not ious:
        return "FP", 0, {"overlap_percentage": 0, "size_ratio": 0}

    max_iou_idx = np.argmax(ious)
    max_iou = ious[max_iou_idx]
    best_overlap_metrics = overlap_metrics_list[max_iou_idx]

    if max_iou >= iou_threshold:
        size_ratio = best_overlap_metrics["size_ratio"]
        if size_ratio > 1.5:
            classification = "TP_OVERSIZED"
        elif size_ratio < 0.67:
            classification = "TP_UNDERSIZED"
        else:
            classification = "TP"
    else:
        classification = "FP"

    return classification, max_iou, best_overlap_metrics


def analyze_predictions(predictions_file, truth_file, iou_threshold=0.5):
    """
    Analyze predictions against ground truth and add classification attributes
    """
    pred_gdf = gpd.read_file(predictions_file)
    truth_gdf = gpd.read_file(truth_file)

    if pred_gdf.crs != truth_gdf.crs:
        truth_gdf = truth_gdf.to_crs(pred_gdf.crs)

    pred_gdf["classification"] = None
    pred_gdf["iou_score"] = None
    pred_gdf["overlap_percentage"] = None
    pred_gdf["size_ratio"] = None
    pred_gdf["area"] = pred_gdf.geometry.area

    for idx, pred_row in pred_gdf.iterrows():
        classification, iou, overlap_metrics = classify_prediction(
            pred_row.geometry, truth_gdf, iou_threshold
        )
        pred_gdf.at[idx, "classification"] = classification
        pred_gdf.at[idx, "iou_score"] = iou
        pred_gdf.at[idx, "overlap_percentage"] = overlap_metrics["overlap_percentage"]
        pred_gdf.at[idx, "size_ratio"] = overlap_metrics["size_ratio"]

    total_predictions = len(pred_gdf)
    total_truth = len(truth_gdf)
    true_positives = sum(
        pred_gdf["classification"].isin(["TP", "TP_OVERSIZED", "TP_UNDERSIZED"])
    )
    oversized = sum(pred_gdf["classification"] == "TP_OVERSIZED")
    undersized = sum(pred_gdf["classification"] == "TP_UNDERSIZED")
    false_positives = sum(pred_gdf["classification"] == "FP")
    false_negatives = total_truth - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    avg_overlap = pred_gdf[
        pred_gdf["classification"].isin(["TP", "TP_OVERSIZED", "TP_UNDERSIZED"])
    ]["overlap_percentage"].mean()
    avg_size_ratio = pred_gdf[
        pred_gdf["classification"].isin(["TP", "TP_OVERSIZED", "TP_UNDERSIZED"])
    ]["size_ratio"].mean()

    metrics = {
        "total_predictions": total_predictions,
        "total_ground_truth": total_truth,
        "true_positives": true_positives,
        "true_positives_oversized": oversized,
        "true_positives_undersized": undersized,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "average_overlap_percentage": avg_overlap,
        "average_size_ratio": avg_size_ratio,
    }

    return pred_gdf, metrics


def save_results(gdf, output_file, metrics_file=None):
    """
    Save the analyzed GeoDataFrame to a new GeoJSON file
    """
    gdf.to_file(output_file, driver="GeoJSON")

    if metrics_file:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    predictions_file = "predictions.geojson"
    truth_file = "labels.geojson"
    output_file = "predictions_analyzed.geojson"
    metrics_file = "analysis_metrics.json"

    analyzed_predictions, metrics = analyze_predictions(predictions_file, truth_file)
    save_results(analyzed_predictions, output_file, metrics_file)
