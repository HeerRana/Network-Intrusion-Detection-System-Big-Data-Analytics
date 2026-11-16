"""Local dashboard for the UNSW-NB15 intrusion detection project."""

import csv
import io
import json
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "pyspark_results"
MODEL = RESULTS / "ids_binary_model"
DATA = ROOT / "data"
MAX_ROWS = 10_000


@st.cache_data
def read_csv(name):
    return pd.read_csv(RESULTS / name)


def percent(value):
    return f"{value:.2%}"


def overview():
    summary = read_csv("part1_summary.csv").iloc[0]
    columns = st.columns(4)
    columns[0].metric("Total flows", f"{summary['Total Records']:,.0f}")
    columns[1].metric("Normal", f"{summary['Normal Traffic']:,.0f}")
    columns[2].metric("Attacks", f"{summary['Attack Traffic']:,.0f}")
    columns[3].metric("Attack share", f"{summary['Attack %']:.2f}%")

    st.subheader("Attack categories")
    counts = read_csv("multiclass_class_distribution.csv").set_index("attack_category")
    attacks = counts.drop(index="Normal")
    st.bar_chart(attacks, horizontal=True, sort=False, x_label="Network flows")
    st.dataframe(attacks.rename(columns={"count": "Flows"}))

    with st.expander("Dataset feature dictionary"):
        st.dataframe(
            pd.read_csv(DATA / "UNSW-NB15_features.csv", encoding="latin-1"),
            hide_index=True,
        )


def model_results():
    mode = st.segmented_control(
        "Classifier", ["Binary", "Multiclass"], default="Binary", selection_mode="single"
    ) or "Binary"
    binary = mode == "Binary"
    key = mode.lower()
    metrics_file = "binary_classification_metrics.csv" if binary else "multiclass_metrics.csv"
    metrics = read_csv(metrics_file).set_index("Metric")["Value"]
    report = read_csv(f"{key}_classification_report.csv").set_index("Unnamed: 0")
    report.index.name = "Class"

    columns = st.columns(3)
    columns[0].metric("Accuracy", percent(metrics["Accuracy"]))
    columns[1].metric("Weighted F1", percent(report.loc["weighted avg", "f1-score"]))
    columns[2].metric("Macro F1", percent(report.loc["macro avg", "f1-score"]))
    st.caption("Evaluation on 761,911 test flows from a seeded 70/30 dataset split.")

    if binary:
        columns = st.columns(3)
        columns[0].metric("Missed attacks", f"{metrics['False Negatives']:,.0f}")
        columns[1].metric("False alerts", f"{metrics['False Positives']:,.0f}")
        columns[2].metric("ROC AUC", percent(metrics["AUC-ROC"]))
    else:
        st.info("Macro F1 weights each class equally. Check recall and test counts for rare attacks.")

    st.subheader("Per-class performance")
    classes = report.drop(index=["accuracy", "macro avg", "weighted avg"]).copy()
    for column in ["precision", "recall", "f1-score"]:
        classes[column] = classes[column].map(percent)
    classes["support"] = classes["support"].astype(int)
    st.dataframe(classes.rename(columns={"support": "Test flows"}))
    matrix, importance = st.tabs(["Confusion matrix", "Feature importance"])
    with matrix:
        image = "binary_confusion_matrix.png" if binary else "multiclass_confusion_normalized.png"
        st.image(str(RESULTS / image))
    with importance:
        st.image(str(RESULTS / f"{key}_feature_importance.png"))


def model_features():
    metadata = next(MODEL.glob("stages/0_VectorAssembler_*/metadata/part-*.txt"))
    return json.loads(metadata.read_text(encoding="utf-8"))["paramMap"]["inputCols"]


def validate_features(rows, features):
    rows = rows.copy()
    rows.columns = rows.columns.str.strip().str.lower()
    if rows.columns.duplicated().any():
        raise ValueError("Column names must be unique after trimming and ignoring case.")
    missing = [name for name in features if name not in rows]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if rows.empty or len(rows) > MAX_ROWS:
        raise ValueError(f"Upload between 1 and {MAX_ROWS:,} flows.")

    inputs = rows[features].replace(r"^\s*$", np.nan, regex=True)
    numeric = inputs.apply(pd.to_numeric, errors="coerce")
    invalid = (inputs.notna() & numeric.isna()) | numeric.isin([np.inf, -np.inf])
    if invalid.any().any():
        bad_columns = invalid.columns[invalid.any()].tolist()
        raise ValueError("Expected finite numeric values in: " + ", ".join(bad_columns))
    return numeric.fillna(0).astype(float)


def read_upload(contents):
    with io.TextIOWrapper(io.BytesIO(contents), encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        names = [name.strip().lower() for name in next(reader, [])]
        if not names:
            raise ValueError("The CSV must include a header row.")
        if len(names) != len(set(names)):
            raise ValueError("Column names must be unique after trimming and ignoring case.")
        values = list(islice((row for row in reader if row), MAX_ROWS + 1))
        if any(len(row) != len(names) for row in values):
            raise ValueError("Each CSV row must have the same number of fields as the header.")
        return pd.DataFrame(values, columns=names)


@st.cache_resource(show_spinner=False)
def load_model():
    import pyarrow.parquet as parquet

    scaler_path = next(MODEL.glob("stages/1_StandardScaler_*/data/*.parquet"))
    forest_path = next(MODEL.glob("stages/2_RandomForestClassifier_*/data/*.parquet"))
    scaler = parquet.ParquetFile(scaler_path).read().to_pylist()[0]
    std = np.array(scaler["std"]["values"])
    scale = np.divide(1, std, out=np.zeros_like(std), where=std != 0)

    records = parquet.ParquetFile(forest_path).read().to_pylist()
    records.sort(key=lambda row: (row["treeID"], row["nodeData"]["id"]))
    roots = []
    splits = []
    probabilities = []
    for index, record in enumerate(records):
        node = record["nodeData"]
        offset = index - node["id"]
        if node["id"] == 0:
            roots.append(index)
        if node["leftChild"] == -1:
            splits.append((0, 0.0, index, index))
        else:
            split = node["split"]
            if split["numCategories"] != -1:
                raise ValueError("This dashboard expects the saved forest's numeric splits.")
            splits.append((
                split["featureIndex"],
                split["leftCategoriesOrThreshold"][0],
                offset + node["leftChild"],
                offset + node["rightChild"],
            ))
        counts = np.array(node["impurityStats"])
        probabilities.append(counts / counts.sum() if counts.sum() else counts)

    feature, threshold, left, right = map(np.array, zip(*splits))
    return scale, roots, feature, threshold, left, right, np.array(probabilities)


def score(rows):
    scale, roots, feature, threshold, left, right, probabilities = load_model()
    values = rows.to_numpy() * scale
    totals = np.zeros((len(rows), 2))
    flow = np.arange(len(rows))
    # Spark sums the normalized class counts at each tree's reached leaf.
    for root in roots:
        node = np.full(len(rows), root)
        while np.any(node != left[node]):
            node = np.where(values[flow, feature[node]] <= threshold[node], left[node], right[node])
        totals += probabilities[node]
    labels = np.array(["Normal", "Attack"])[totals.argmax(axis=1)]
    return rows.reset_index(drop=True).assign(flow_number=flow + 1, prediction=labels)


def predict():
    st.subheader("Score network flows")
    st.write(
        "Upload a CSV of pre-extracted flow features, or try 10 flows from UNSW-NB15. "
        f"Up to {MAX_ROWS:,} flows per upload. Blank feature values are treated as zero."
    )
    features = model_features()
    sample = pd.read_csv(DATA / "dashboard_sample_flows.csv", keep_default_na=False)
    sample = validate_features(sample, features)
    st.download_button("Download sample CSV", sample.to_csv(index=False), "sample_flows.csv")
    with st.expander("Required columns"):
        st.code(", ".join(features), language=None)

    upload = st.file_uploader("Flow CSV", type=["csv"])
    run = st.button("Score uploaded CSV" if upload else "Score sample flows", type="primary")
    upload_bytes = upload.getvalue() if upload else None

    if run:
        st.session_state.pop("scored_flows", None)
        try:
            rows = (
                validate_features(read_upload(upload_bytes), features)
                if upload else sample
            )
            with st.spinner("Scoring flows…"):
                st.session_state["scored_flows"] = (upload_bytes, score(rows))
        except (ValueError, csv.Error) as error:
            st.error(str(error))
        except (ImportError, OSError, RuntimeError) as error:
            st.error(f"Could not load the saved model. {error}")
        except Exception as error:
            st.error("The saved model could not score these flows.")
            with st.expander("Technical details"):
                st.exception(error)

    cached = st.session_state.get("scored_flows")
    if cached is not None and cached[0] == upload_bytes:
        predictions = cached[1]
        st.metric("Flows flagged as attacks", f"{predictions['prediction'].eq('Attack').sum():,}")
        displayed = ["flow_number", "prediction", *features]
        st.dataframe(predictions[displayed], hide_index=True)
        st.download_button(
            "Download predictions", predictions[displayed].to_csv(index=False), "predictions.csv"
        )


def main():
    st.set_page_config(page_title="UNSW-NB15 Intrusion Detection", layout="wide")
    st.title("Network Intrusion Detection")
    st.caption("UNSW-NB15 · Traffic analysis and Random Forest evaluation")
    page = st.sidebar.radio("Dashboard", ["Dataset overview", "Model results", "Predict from CSV"])
    if page == "Dataset overview":
        overview()
    elif page == "Model results":
        model_results()
    else:
        predict()


if __name__ == "__main__":
    main()
