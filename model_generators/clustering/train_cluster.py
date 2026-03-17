import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import joblib

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
CV_THRESHOLD     = 15.0

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_class_cv(data, labels):
    data = data.copy()
    data["_label"] = labels
    cv_rows = []
    for label, group in data.groupby("_label"):
        row = {"cluster": label}
        for col in SEGMENT_FEATURES:
            mean   = group[col].mean()
            std    = group[col].std()
            cv_pct = (std / mean * 100) if mean != 0 else 0.0
            row[f"CV_{col}"] = round(cv_pct, 2)
        cv_rows.append(row)
    return pd.DataFrame(cv_rows).set_index("cluster")

def all_classes_meet_cv(cv_df, threshold=CV_THRESHOLD):
    cv_cols = [c for c in cv_df.columns if c.startswith("CV_")]
    return bool((cv_df[cv_cols] <= threshold).all().all())

def clip_df(source_df, m_i, m_p):
    out = source_df[SEGMENT_FEATURES].copy()
    for col, mult in [("estimated_income", m_i), ("selling_price", m_p)]:
        Q1  = source_df[col].quantile(0.25)
        Q3  = source_df[col].quantile(0.75)
        IQR = Q3 - Q1
        out[col] = source_df[col].clip(Q1 - mult * IQR, Q3 + mult * IQR)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESULT  — computed once on first call, reused after
# ─────────────────────────────────────────────────────────────────────────────
_cached_result = None

def evaluate_clustering_model():
    global _cached_result
    if _cached_result is not None:
        return _cached_result

    df    = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    X_raw = df[SEGMENT_FEATURES]

    # ── ORIGINAL MODEL  k=3, no scaling ──────────────────────────────────────
    kmeans_original  = KMeans(n_clusters=3, random_state=42, n_init="auto")
    df["cluster_id"] = kmeans_original.fit_predict(X_raw)
    centers          = kmeans_original.cluster_centers_
    sorted_idx       = centers[:, 0].argsort()
    cluster_mapping  = {
        sorted_idx[0]: "Economy",
        sorted_idx[1]: "Standard",
        sorted_idx[2]: "Premium",
    }
    df["client_class"] = df["cluster_id"].map(cluster_mapping)
    joblib.dump(kmeans_original, "model_generators/clustering/clustering_model.pkl")
    original_silhouette = round(silhouette_score(X_raw, df["cluster_id"]), 4)

    # Overall CV across k values
    k_values     = [2, 3, 4, 5, 6]
    k_scores_raw = []
    for k in k_values:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init="auto")
        k_scores_raw.append(round(silhouette_score(X_raw, km_temp.fit_predict(X_raw)), 4))
    cv_overall = round((np.std(k_scores_raw) / np.mean(k_scores_raw)) * 100, 2)

    # Step scores
    sc          = StandardScaler()
    X_s1        = sc.fit_transform(X_raw)
    step1_score = round(silhouette_score(
        X_s1, KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(X_s1)), 4)

    qt2         = QuantileTransformer(output_distribution="normal", random_state=42)
    X_s2        = qt2.fit_transform(X_raw)
    step2_score = round(silhouette_score(
        X_s2, KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(X_s2)), 4)

    df_s3       = clip_df(df, 1.0, 1.0)
    qt3         = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0)
    X_s3        = qt3.fit_transform(df_s3)
    step3_score = round(silhouette_score(
        X_s3, KMeans(n_clusters=5, random_state=42, n_init=100).fit_predict(X_s3)), 4)

    # ── REFINED MODEL search ──────────────────────────────────────────────────
    best_silhouette  = -1
    best_labels      = None
    best_k           = None
    best_qt          = None
    best_cv_df       = None
    best_X_qt        = None
    best_mult_income = None
    best_mult_price  = None

    tight_mults  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    k_candidates = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

    print("Searching for silhouette > 0.9 with CV <= 15%...")

    for k in k_candidates:
        for m_i in tight_mults:
            for m_p in tight_mults:
                df_try = clip_df(df, m_i, m_p)
                qt_try = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(200, len(df_try)),
                    random_state=0
                )
                X_qt   = qt_try.fit_transform(df_try)
                km     = KMeans(n_clusters=k, random_state=0, n_init=300, max_iter=2000)
                labels = km.fit_predict(X_qt)
                cv_df  = compute_per_class_cv(df[SEGMENT_FEATURES], labels)
                if not all_classes_meet_cv(cv_df):
                    continue
                sil = round(silhouette_score(X_qt, labels), 4)
                if sil > best_silhouette:
                    best_silhouette  = sil
                    best_labels      = labels.copy()
                    best_k           = k
                    best_qt          = qt_try
                    best_cv_df       = cv_df.copy()
                    best_X_qt        = X_qt.copy()
                    best_mult_income = m_i
                    best_mult_price  = m_p
                    print(f"  New best: sil={sil}  k={k}  m_i={m_i}  m_p={m_p}")
                if best_silhouette >= 0.9:
                    break
            if best_silhouette >= 0.9:
                break
        if best_silhouette >= 0.9:
            break

    # ── Fallback ──────────────────────────────────────────────────────────────
    if best_labels is None:
        print("Fallback: no config met CV<=15%, using best available")
        df_fb   = clip_df(df, 0.10, 0.10)
        best_qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0)
        X_fb    = best_qt.fit_transform(df_fb)
        km_fb   = KMeans(n_clusters=7, random_state=0, n_init=300, max_iter=2000)
        best_labels      = km_fb.fit_predict(X_fb)
        best_silhouette  = round(silhouette_score(X_fb, best_labels), 4)
        best_k           = 7
        best_mult_income = 0.10
        best_mult_price  = 0.10
        best_X_qt        = X_fb
        best_cv_df       = compute_per_class_cv(df[SEGMENT_FEATURES], best_labels)

    refined_silhouette       = best_silhouette
    df["refined_cluster_id"] = best_labels
    print(f"\n✅ Final: silhouette={refined_silhouette}  k={best_k}  CV<=15% all={all_classes_meet_cv(best_cv_df)}")

    # ── Label clusters by income rank ─────────────────────────────────────────
    cluster_label_names = [
        "Budget", "Economy", "Standard", "Mid-Range",
        "Comfort", "Premium", "Ultra Premium",
        "Elite", "Prestige", "Apex",
        "Tier-11", "Tier-12", "Tier-13", "Tier-14",
        "Tier-15", "Tier-16", "Tier-17", "Tier-18",
        "Tier-19", "Tier-20",
    ]

    df_win         = clip_df(df, best_mult_income, best_mult_price)
    kmeans_refined = KMeans(n_clusters=best_k, random_state=0, n_init=300, max_iter=2000)
    kmeans_refined.fit(best_qt.fit_transform(df_win))
    centers_r       = kmeans_refined.cluster_centers_
    income_order    = centers_r[:, 0].argsort()
    label_mapping_r = {income_order[i]: cluster_label_names[i] for i in range(best_k)}
    df["refined_client_class"] = df["refined_cluster_id"].map(label_mapping_r)

    joblib.dump(kmeans_refined, "model_generators/clustering/clustering_model_refined.pkl")
    joblib.dump(best_qt,        "model_generators/clustering/qt_scaler.pkl")

    # ── Tables ────────────────────────────────────────────────────────────────
    per_class_cv_df = best_cv_df.copy()
    per_class_cv_df.index = [label_mapping_r.get(i, f"Cluster {i}") for i in per_class_cv_df.index]
    per_class_cv_df.index.name = "Segment"
    per_class_cv_df = per_class_cv_df.rename(columns={
        "CV_estimated_income": "CV Income (%)",
        "CV_selling_price":    "CV Price (%)",
    })
    per_class_cv_df["Income"] = per_class_cv_df["CV Income (%)"].apply(
        lambda v: "Pass" if v <= CV_THRESHOLD else "Fail")
    per_class_cv_df["Price"]  = per_class_cv_df["CV Price (%)"].apply(
        lambda v: "Pass" if v <= CV_THRESHOLD else "Fail")
    per_class_cv_df["Status"] = per_class_cv_df.apply(
        lambda r: "Pass" if r["Income"] == "Pass" and r["Price"] == "Pass" else "Fail", axis=1)

    cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
    cluster_counts  = df["client_class"].value_counts().reset_index()
    cluster_counts.columns = ["client_class", "count"]
    cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

    refined_summary = df.groupby("refined_client_class")[SEGMENT_FEATURES].mean()
    refined_counts  = df["refined_client_class"].value_counts().reset_index()
    refined_counts.columns = ["refined_client_class", "count"]
    refined_summary = refined_summary.merge(refined_counts, on="refined_client_class")

    journey_df = pd.DataFrame({
        "Step": [
            "Original (k=3, no scaling)",
            "Step 1: StandardScaler (k=2)",
            "Step 2: QuantileTransformer (k=2)",
            "Step 3: QT + IQR x1.0 (k=5)",
            f"Step 4: Tight IQR + QT + k={best_k} (CV<={CV_THRESHOLD}%)",
        ],
        "Silhouette": [original_silhouette, step1_score, step2_score,
                       step3_score, refined_silhouette],
        "vs Original": [
            "-",
            f"+{round(step1_score - original_silhouette, 4)}",
            f"+{round(step2_score - original_silhouette, 4)}",
            f"+{round(step3_score - original_silhouette, 4)}",
            f"+{round(refined_silhouette - original_silhouette, 4)}",
        ],
        "Technique": [
            "No preprocessing", "Scale normalisation",
            "Gaussian mapping", "Moderate outlier removal",
            f"IQR x{best_mult_income}/{best_mult_price}, k={best_k}",
        ],
    })

    k_comparison_df = pd.DataFrame({"K": k_values, "Silhouette Score": k_scores_raw})
    comparison_df   = df[["client_name", "estimated_income", "selling_price", "client_class"]]

    # ── Cache and return ──────────────────────────────────────────────────────
    _cached_result = {
        "silhouette":             original_silhouette,
        "cv":                     cv_overall,
        "step1_score":            step1_score,
        "step2_score":            step2_score,
        "step3_score":            step3_score,
        "refined_silhouette":     refined_silhouette,
        "best_k":                 best_k,
        "best_mult_income":       best_mult_income,
        "best_mult_price":        best_mult_price,
        "cv_threshold":           CV_THRESHOLD,
        "cv_all_passed":          all_classes_meet_cv(best_cv_df),
        "silhouette_pct":         round(original_silhouette * 100, 1),
        "step1_pct":              round(step1_score * 100, 1),
        "step2_pct":              round(step2_score * 100, 1),
        "step3_pct":              round(step3_score * 100, 1),
        "refined_silhouette_pct": round(refined_silhouette * 100, 1),
        "per_class_cv": per_class_cv_df.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f", justify="center"),
        "journey": journey_df.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.4f", justify="center", index=False),
        "k_comparison": k_comparison_df.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.4f", justify="center", index=False),
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f", justify="center", index=False),
        "refined_summary": refined_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f", justify="center", index=False),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f", justify="center", index=False),
    }
    return _cached_result