import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union


def calculate_overlap_metrics(pred_geom, truth_geom):
    try:
        intersection_area = pred_geom.intersection(truth_geom).area
        union_area = pred_geom.union(truth_geom).area
        pred_area = pred_geom.area
        truth_area = truth_geom.area

        overlap_pred = (intersection_area / pred_area) * 100 if pred_area > 0 else 0
        overlap_truth = (intersection_area / truth_area) * 100 if truth_area > 0 else 0
        iou = intersection_area / union_area if union_area > 0 else 0

        return {
            "overlap_percentage": overlap_truth,
            "intersection_area": intersection_area,
            "pred_area": pred_area,
            "truth_area": truth_area,
            "iou": iou,
        }
    except:
        return {
            "overlap_percentage": 0,
            "intersection_area": 0,
            "pred_area": 0,
            "truth_area": 0,
            "iou": 0,
        }


def classify_prediction(pred_geom, truth_gdf, overlap_threshold=5):
    overlaps = []
    metrics_list = []

    for truth_geom in truth_gdf.geometry:
        metrics = calculate_overlap_metrics(pred_geom, truth_geom)
        overlaps.append(metrics["overlap_percentage"])
        metrics_list.append(metrics)

    if not overlaps:
        return "FP", 0, {"overlap_percentage": 0, "iou": 0}

    max_overlap_idx = np.argmax(overlaps)
    max_overlap = overlaps[max_overlap_idx]
    best_metrics = metrics_list[max_overlap_idx]

    classification = "TP" if max_overlap >= overlap_threshold else "FP"
    return classification, max_overlap, best_metrics


def identify_false_negatives(pred_gdf, truth_gdf, overlap_threshold=5):
    matched_truth_indices = set()

    for idx, pred_row in pred_gdf[pred_gdf["classification"] == "TP"].iterrows():
        overlaps = []
        for truth_idx, truth_row in truth_gdf.iterrows():
            metrics = calculate_overlap_metrics(pred_row.geometry, truth_row.geometry)
            if metrics["overlap_percentage"] >= overlap_threshold:
                matched_truth_indices.add(truth_idx)

    false_negatives = truth_gdf.loc[~truth_gdf.index.isin(matched_truth_indices)].copy()
    false_negatives["classification"] = "FN"
    false_negatives["overlap_percentage"] = 0
    false_negatives["iou"] = 0
    false_negatives["area"] = false_negatives.geometry.area

    return false_negatives


def identify_true_negatives(combined_gdf):
    """
    Identify true negative regions by subtracting all existing geometries from the bounding box.
    Returns a GeoDataFrame containing the true negative regions.
    """

    bounds = combined_gdf.total_bounds
    bounding_box = box(bounds[0], bounds[1], bounds[2], bounds[3])

    # combine all geom
    all_geometries = unary_union(combined_gdf.geometry)

    # difference between bounding box and all geometries
    true_negative_geometry = bounding_box.difference(all_geometries)

    if true_negative_geometry.geom_type == "MultiPolygon":
        tn_geometries = list(true_negative_geometry.geoms)
    else:
        tn_geometries = [true_negative_geometry]

    tn_data = {
        "geometry": tn_geometries,
        "classification": ["TN"] * len(tn_geometries),
        "overlap_percentage": [0] * len(tn_geometries),
        "iou": [0] * len(tn_geometries),
        "area": [geom.area for geom in tn_geometries],
    }

    tn_gdf = gpd.GeoDataFrame(tn_data)
    # print(tn_gdf)
    return tn_gdf


def analyze_predictions(predictions_file, truth_file, overlap_threshold=5):
    pred_gdf = gpd.read_file(predictions_file)
    truth_gdf = gpd.read_file(truth_file)

    if pred_gdf.crs != truth_gdf.crs:
        truth_gdf = truth_gdf.to_crs(pred_gdf.crs)

    pred_gdf["classification"] = None
    pred_gdf["overlap_percentage"] = None
    pred_gdf["iou"] = None
    pred_gdf["area"] = pred_gdf.geometry.area

    # Classify predictions as TP or FP
    for idx, pred_row in pred_gdf.iterrows():
        classification, overlap, metrics = classify_prediction(
            pred_row.geometry, truth_gdf, overlap_threshold
        )
        pred_gdf.at[idx, "classification"] = classification
        pred_gdf.at[idx, "overlap_percentage"] = metrics["overlap_percentage"]
        pred_gdf.at[idx, "iou"] = metrics["iou"]

    fn_gdf = identify_false_negatives(pred_gdf, truth_gdf, overlap_threshold)

    combined_gdf = gpd.GeoDataFrame(pd.concat([pred_gdf, fn_gdf], ignore_index=True))

    tn_gdf = identify_true_negatives(combined_gdf)
    print(tn_gdf)
    final_gdf = gpd.GeoDataFrame(pd.concat([combined_gdf, tn_gdf], ignore_index=True))

    total_predictions = len(pred_gdf)
    total_truth = len(truth_gdf)
    true_positives = sum(final_gdf["classification"] == "TP")
    false_positives = sum(final_gdf["classification"] == "FP")
    false_negatives = sum(final_gdf["classification"] == "FN")
    true_negatives = sum(final_gdf["classification"] == "TN")

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

    avg_overlap = final_gdf[final_gdf["classification"] == "TP"][
        "overlap_percentage"
    ].mean()
    avg_iou = final_gdf[final_gdf["classification"] == "TP"]["iou"].mean()

    metrics = {
        "total_predictions": total_predictions,
        "total_ground_truth": total_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "average_overlap_percentage": avg_overlap,
        "average_iou": avg_iou,
    }

    return final_gdf, metrics


def save_results(gdf, output_file, metrics_file=None):
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
