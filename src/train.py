"""Run the complete CS439 final project experiment.

Project: Hybrid Unsupervised Risk Phenotyping and Supervised Classification
for Breast Cancer Diagnosis.

This script is intentionally self-contained. It loads the Wisconsin Diagnostic
Breast Cancer dataset from scikit-learn, performs a strict stratified train/test
split, fits all preprocessing only on the training set, compares supervised
baselines, and evaluates a hybrid K-Means + Logistic Regression model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
DATA_DIR = ROOT / "data"
for p in (FIG_DIR, TABLE_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)


def load_wdbc() -> Tuple[pd.DataFrame, pd.Series]:
    """Load WDBC and map the binary target to 1 = malignant, 0 = benign."""
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = pd.Series((bunch.target == 0).astype(int), name="malignant")
    full = X.copy()
    full["malignant"] = y
    full.to_csv(DATA_DIR / "wdbc_clean.csv", index=False)
    return X, y


def predict_scores(model, X) -> np.ndarray:
    """Return scores for ROC/PR metrics."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_model(name: str, model, X_test, y_test) -> Dict[str, float | str]:
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_k_by_silhouette(X_train_scaled: np.ndarray) -> int:
    """Select k using training-only silhouette scores."""
    from sklearn.metrics import silhouette_score

    rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=8, random_state=RANDOM_STATE, algorithm="lloyd")
        labels = km.fit_predict(X_train_scaled)
        rows.append({"k": k, "silhouette": silhouette_score(X_train_scaled, labels), "inertia": km.inertia_})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "kmeans_model_selection.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["k"], df["silhouette"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Training silhouette score")
    plt.title("K-Means model selection")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_silhouette.png", dpi=220)
    plt.close()

    return int(df.sort_values("silhouette", ascending=False).iloc[0]["k"])


def make_hybrid_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale features, learn K-Means phenotypes, one-hot encode cluster IDs."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k = choose_k_by_silhouette(X_train_scaled)
    kmeans = KMeans(n_clusters=k, n_init=12, random_state=RANDOM_STATE, algorithm="lloyd")
    train_cluster = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
    test_cluster = kmeans.predict(X_test_scaled).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_cluster_ohe = encoder.fit_transform(train_cluster)
    test_cluster_ohe = encoder.transform(test_cluster)

    X_train_hybrid = np.hstack([X_train_scaled, train_cluster_ohe])
    X_test_hybrid = np.hstack([X_test_scaled, test_cluster_ohe])

    metadata = {
        "selected_k": k,
        "cluster_train_counts": pd.Series(train_cluster.ravel()).value_counts().sort_index().to_dict(),
        "cluster_test_counts": pd.Series(test_cluster.ravel()).value_counts().sort_index().to_dict(),
    }
    with open(TABLE_DIR / "hybrid_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return X_train_hybrid, X_test_hybrid, metadata, scaler, train_cluster.ravel()


def cross_validation_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute compact 3-fold CV for core supervised baselines."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, random_state=RANDOM_STATE, class_weight="balanced_subsample"
        ),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=3.0, kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"],
            n_jobs=1,
        )
        row = {"model": name}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "cross_validation_summary.csv", index=False)
    return df


def plot_metric_bars(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.set_index("model")[["accuracy", "recall", "f1", "roc_auc"]]
    ax = plot_df.plot(kind="bar", figsize=(8, 4), ylim=(0.75, 1.01))
    ax.set_ylabel("Score")
    ax.set_title("Model comparison across key metrics")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "metric_comparison.png", dpi=220)
    plt.close()


def plot_confusion(model, X_test, y_test, name: str, path: Path) -> None:
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Benign", "Malignant"], values_format="d"
    )
    disp.ax_.set_title(name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_pca(X_train: pd.DataFrame, y_train: pd.Series, scaler: StandardScaler, cluster_labels: np.ndarray) -> None:
    X_scaled = scaler.transform(X_train)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, s=22, alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("PCA projection of K-Means phenotypes")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_kmeans_clusters.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y_train, s=22, alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("PCA projection colored by diagnosis")
    cbar = plt.colorbar(scatter, label="Diagnosis")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Benign", "Malignant"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_diagnosis.png", dpi=220)
    plt.close()


def plot_roc_pr(fitted: Dict[str, object], X_test, y_test, hybrid, X_test_hybrid) -> None:
    plt.figure(figsize=(6, 5))
    for name, model in fitted.items():
        scores = predict_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, scores):.3f})")
    scores = hybrid.predict_proba(X_test_hybrid)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, scores)
    plt.plot(fpr, tpr, label=f"Hybrid K-Means + Logistic (AUC={roc_auc_score(y_test, scores):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on held-out test set")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    for name, model in fitted.items():
        scores = predict_scores(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        plt.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, scores):.3f})")
    precision, recall, _ = precision_recall_curve(y_test, scores)
    scores = hybrid.predict_proba(X_test_hybrid)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, scores)
    plt.plot(recall, precision, label=f"Hybrid K-Means + Logistic (AP={average_precision_score(y_test, scores):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves on held-out test set")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_curves.png", dpi=220)
    plt.close()


def main() -> None:
    X, y = load_wdbc()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    with open(TABLE_DIR / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_total": len(X), "n_train": len(X_train), "n_test": len(X_test),
            "positive_class": "malignant", "positive_class_count_total": int(y.sum()),
            "negative_class_count_total": int((1 - y).sum()),
            "test_size": 0.20, "random_state": RANDOM_STATE,
        }, f, indent=2)

    cross_validation_summary(X, y)

    models = {
        "Majority Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced_subsample"
        ),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }

    fitted, metrics = {}, []
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        metrics.append(evaluate_model(name, model, X_test, y_test))

    X_train_hybrid, X_test_hybrid, _, scaler, train_clusters = make_hybrid_features(X_train, X_test)
    hybrid = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    hybrid.fit(X_train_hybrid, y_train)
    metrics.append(evaluate_model("Hybrid K-Means + Logistic", hybrid, X_test_hybrid, y_test))

    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(TABLE_DIR / "test_metrics.csv", index=False)

    plot_metric_bars(metrics_df)
    plot_confusion(fitted["Logistic Regression"], X_test, y_test, "Logistic Regression confusion matrix", FIG_DIR / "confusion_logistic.png")
    plot_confusion(fitted["Random Forest"], X_test, y_test, "Random Forest confusion matrix", FIG_DIR / "confusion_random_forest.png")
    plot_confusion(hybrid, X_test_hybrid, y_test, "Hybrid confusion matrix", FIG_DIR / "confusion_hybrid.png")
    plot_pca(X_train, y_train, scaler, train_clusters)
    plot_roc_pr(fitted, X_test, y_test, hybrid, X_test_hybrid)

    best_name = metrics_df[metrics_df["model"] != "Hybrid K-Means + Logistic"].iloc[0]["model"]
    best_model = fitted[best_name]
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=3, random_state=RANDOM_STATE, scoring="roc_auc")
    imp_df = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std})
    imp_df = imp_df.sort_values("importance_mean", ascending=False)
    imp_df.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)
    top = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Mean decrease in ROC-AUC after permutation")
    plt.title(f"Permutation importance: {best_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance.png", dpi=220)
    plt.close()

    print("Saved outputs to:")
    print(f"  {TABLE_DIR}")
    print(f"  {FIG_DIR}")
    print("\nHeld-out test metrics:")
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
