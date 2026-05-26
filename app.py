from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_TITLE = "EduPro Student Segmentation & Course Recommendation"
DATA_FILE = Path(__file__).parent / "EduPro Online Platform.xlsx"
RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "Users": {"UserID", "Age", "Gender"},
    "Courses": {
        "CourseID",
        "CourseName",
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "CourseRating",
    },
    "Transactions": {"UserID", "CourseID", "TransactionDate", "Amount"},
}

LEVEL_WEIGHT = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}


@dataclass(frozen=True)
class ModelArtifacts:
    users: pd.DataFrame
    courses: pd.DataFrame
    transactions: pd.DataFrame
    enrollments: pd.DataFrame
    profiles: pd.DataFrame
    features: pd.DataFrame
    feature_matrix: np.ndarray
    cluster_summary: pd.DataFrame
    k_metrics: pd.DataFrame
    pca_frame: pd.DataFrame
    silhouette: float
    hierarchical_silhouette: float
    intra_cluster_similarity: float
    selected_k: int


def _mode_or_unknown(series: pd.Series) -> str:
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else "Unknown"


def _normalize(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def _safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return 0.0
    return float(silhouette_score(matrix, labels))


def _name_segment(row: pd.Series) -> str:
    if row["diversity_score"] >= 5:
        return "Cross-Domain Explorers"
    if row["avg_spend_per_learner"] >= row["spend_q75"] and row["paid_enrollment_rate"] >= 0.5:
        return "Career Investment Learners"
    if row["advanced_share"] >= 0.45 or row["learning_depth_index"] >= 1.3:
        return "Advanced Skill Builders"
    if row["diversity_score"] <= 2 and row["total_courses_enrolled"] >= row["enrollment_median"]:
        return "Focused Specialists"
    if row["beginner_share"] >= 0.55:
        return "Foundation Builders"
    return "Balanced Progressors"


def _validate_workbook(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Keep the Excel file in the repository root."
        )

    workbook = pd.ExcelFile(path)
    missing_sheets = sorted(set(REQUIRED_COLUMNS) - set(workbook.sheet_names))
    if missing_sheets:
        raise ValueError(f"Missing required sheet(s): {', '.join(missing_sheets)}")

    for sheet_name, required in REQUIRED_COLUMNS.items():
        columns = set(pd.read_excel(path, sheet_name=sheet_name, nrows=0).columns)
        missing_columns = sorted(required - columns)
        if missing_columns:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing column(s): {', '.join(missing_columns)}"
            )


@st.cache_data(show_spinner=False)
def load_workbook(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook_path = Path(path)
    _validate_workbook(workbook_path)

    users = pd.read_excel(workbook_path, sheet_name="Users")
    courses = pd.read_excel(workbook_path, sheet_name="Courses")
    transactions = pd.read_excel(workbook_path, sheet_name="Transactions")

    users["UserID"] = users["UserID"].astype(str)
    courses["CourseID"] = courses["CourseID"].astype(str)
    transactions["UserID"] = transactions["UserID"].astype(str)
    transactions["CourseID"] = transactions["CourseID"].astype(str)
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    transactions["Amount"] = pd.to_numeric(transactions["Amount"], errors="coerce").fillna(0)
    courses["CourseRating"] = pd.to_numeric(courses["CourseRating"], errors="coerce")
    if "TransactionID" not in transactions.columns:
        transactions["TransactionID"] = [f"T{i:05d}" for i in range(1, len(transactions) + 1)]
    if "CourseName" not in courses.columns:
        courses["CourseName"] = courses["CourseID"]
    if "CourseDuration" not in courses.columns:
        courses["CourseDuration"] = 0.0

    return users, courses, transactions


def build_learner_profiles(
    users: pd.DataFrame, courses: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    course_features = courses.copy()
    course_features["level_weight"] = course_features["CourseLevel"].map(LEVEL_WEIGHT).fillna(1)

    enrollments = transactions.merge(course_features, on="CourseID", how="left")
    enrollments["is_paid"] = (enrollments["Amount"] > 0).astype(int)
    enrollments["is_beginner"] = (enrollments["CourseLevel"] == "Beginner").astype(int)
    enrollments["is_advanced"] = (enrollments["CourseLevel"] == "Advanced").astype(int)

    date_min = enrollments.groupby("UserID")["TransactionDate"].min()
    date_max = enrollments.groupby("UserID")["TransactionDate"].max()
    active_months = ((date_max.dt.year - date_min.dt.year) * 12 + date_max.dt.month - date_min.dt.month + 1)
    active_months = active_months.clip(lower=1)

    aggregates = enrollments.groupby("UserID").agg(
        total_courses_enrolled=("TransactionID", "count"),
        unique_courses_enrolled=("CourseID", "nunique"),
        total_spent=("Amount", "sum"),
        avg_spend_per_learner=("Amount", "mean"),
        paid_enrollment_rate=("is_paid", "mean"),
        diversity_score=("CourseCategory", "nunique"),
        avg_course_rating_enrolled=("CourseRating", "mean"),
        avg_course_duration=("CourseDuration", "mean"),
        beginner_count=("is_beginner", "sum"),
        advanced_count=("is_advanced", "sum"),
        avg_level_weight=("level_weight", "mean"),
        preferred_course_category=("CourseCategory", _mode_or_unknown),
        preferred_course_level=("CourseLevel", _mode_or_unknown),
        preferred_course_type=("CourseType", _mode_or_unknown),
        first_enrollment=("TransactionDate", "min"),
        last_enrollment=("TransactionDate", "max"),
    )

    aggregates["active_months"] = active_months
    aggregates["enrollment_frequency"] = (
        aggregates["total_courses_enrolled"] / aggregates["active_months"]
    )
    aggregates["avg_courses_per_category"] = (
        aggregates["total_courses_enrolled"] / aggregates["diversity_score"].replace(0, np.nan)
    ).fillna(0)
    aggregates["beginner_share"] = (
        aggregates["beginner_count"] / aggregates["total_courses_enrolled"].replace(0, np.nan)
    ).fillna(0)
    aggregates["advanced_share"] = (
        aggregates["advanced_count"] / aggregates["total_courses_enrolled"].replace(0, np.nan)
    ).fillna(0)
    aggregates["learning_depth_index"] = (
        (aggregates["advanced_count"] + 0.5 * aggregates["avg_level_weight"])
        / (aggregates["beginner_count"] + 1)
    )

    profiles = users.merge(aggregates.reset_index(), on="UserID", how="left")
    numeric_defaults = {
        "total_courses_enrolled": 0,
        "unique_courses_enrolled": 0,
        "total_spent": 0,
        "avg_spend_per_learner": 0,
        "paid_enrollment_rate": 0,
        "diversity_score": 0,
        "avg_course_rating_enrolled": courses["CourseRating"].mean(),
        "avg_course_duration": courses.get("CourseDuration", pd.Series([0])).mean(),
        "beginner_count": 0,
        "advanced_count": 0,
        "avg_level_weight": 1,
        "active_months": 0,
        "enrollment_frequency": 0,
        "avg_courses_per_category": 0,
        "beginner_share": 0,
        "advanced_share": 0,
        "learning_depth_index": 0,
    }
    profiles = profiles.fillna(numeric_defaults)
    for column in ["preferred_course_category", "preferred_course_level", "preferred_course_type"]:
        profiles[column] = profiles[column].fillna("Unknown")

    return profiles, enrollments


def build_feature_matrix(profiles: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    numeric_features = [
        "Age",
        "total_courses_enrolled",
        "avg_courses_per_category",
        "enrollment_frequency",
        "avg_spend_per_learner",
        "paid_enrollment_rate",
        "diversity_score",
        "learning_depth_index",
        "avg_course_rating_enrolled",
        "advanced_share",
        "beginner_share",
    ]
    categorical_features = [
        "Gender",
        "preferred_course_category",
        "preferred_course_level",
        "preferred_course_type",
    ]
    feature_frame = profiles[numeric_features + categorical_features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    matrix = preprocessor.fit_transform(feature_frame)
    return feature_frame, matrix, preprocessor


def select_cluster_count(feature_matrix: np.ndarray, min_k: int = 3, max_k: int = 7) -> tuple[int, pd.DataFrame]:
    rows = []
    upper = min(max_k, len(feature_matrix) - 1)
    for k in range(min_k, upper + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(feature_matrix)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": _safe_silhouette(feature_matrix, labels),
            }
        )
    metrics = pd.DataFrame(rows)
    selected_k = int(metrics.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"])
    return selected_k, metrics


def segment_profiles(
    profiles: pd.DataFrame, feature_matrix: np.ndarray, selected_k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float]:
    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=30)
    labels = kmeans.fit_predict(feature_matrix)
    silhouette = _safe_silhouette(feature_matrix, labels)

    hierarchy = AgglomerativeClustering(n_clusters=selected_k)
    hierarchy_labels = hierarchy.fit_predict(feature_matrix)
    hierarchical_silhouette = _safe_silhouette(feature_matrix, hierarchy_labels)

    clustered = profiles.copy()
    clustered["Cluster"] = labels

    spend_q75 = clustered["avg_spend_per_learner"].quantile(0.75)
    enrollment_median = clustered["total_courses_enrolled"].median()
    cluster_stats = clustered.groupby("Cluster").agg(
        learners=("UserID", "count"),
        avg_age=("Age", "mean"),
        total_courses_enrolled=("total_courses_enrolled", "mean"),
        diversity_score=("diversity_score", "mean"),
        avg_spend_per_learner=("avg_spend_per_learner", "mean"),
        paid_enrollment_rate=("paid_enrollment_rate", "mean"),
        advanced_share=("advanced_share", "mean"),
        beginner_share=("beginner_share", "mean"),
        learning_depth_index=("learning_depth_index", "mean"),
        avg_course_rating_enrolled=("avg_course_rating_enrolled", "mean"),
        preferred_course_category=("preferred_course_category", _mode_or_unknown),
        preferred_course_level=("preferred_course_level", _mode_or_unknown),
    )
    naming_frame = cluster_stats.assign(spend_q75=spend_q75, enrollment_median=enrollment_median)
    cluster_names = naming_frame.apply(_name_segment, axis=1)
    unique_names: dict[str, int] = {}
    final_names = {}
    for cluster_id, name in cluster_names.items():
        count = unique_names.get(name, 0) + 1
        unique_names[name] = count
        final_names[cluster_id] = name if count == 1 else f"{name} {count}"

    clustered["Segment"] = clustered["Cluster"].map(final_names)
    cluster_summary = cluster_stats.reset_index()
    cluster_summary["Segment"] = cluster_summary["Cluster"].map(final_names)
    cluster_summary = cluster_summary[
        ["Cluster", "Segment"]
        + [column for column in cluster_summary.columns if column not in {"Cluster", "Segment"}]
    ].sort_values("Cluster")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(feature_matrix)
    pca_frame = clustered[["UserID", "UserName", "Segment", "Cluster"]].copy()
    pca_frame["PC1"] = components[:, 0]
    pca_frame["PC2"] = components[:, 1]

    similarities = []
    for cluster in sorted(np.unique(labels)):
        rows = feature_matrix[labels == cluster]
        if len(rows) > 1:
            distances = pairwise_distances(rows, metric="cosine")
            upper = distances[np.triu_indices_from(distances, k=1)]
            similarities.extend(1 - upper)
    intra_cluster_similarity = float(np.mean(similarities)) if similarities else 0.0

    return (
        clustered,
        cluster_summary,
        pca_frame,
        silhouette,
        hierarchical_silhouette,
        intra_cluster_similarity,
    )


@st.cache_data(show_spinner="Building learner segmentation model...")
def build_model(path: str, k_override: int | None = None) -> ModelArtifacts:
    users, courses, transactions = load_workbook(path)
    profiles, enrollments = build_learner_profiles(users, courses, transactions)
    features, feature_matrix, _ = build_feature_matrix(profiles)
    selected_k, k_metrics = select_cluster_count(feature_matrix)
    if k_override is not None:
        selected_k = k_override

    (
        clustered,
        cluster_summary,
        pca_frame,
        silhouette,
        hierarchical_silhouette,
        intra_cluster_similarity,
    ) = segment_profiles(profiles, feature_matrix, selected_k)

    return ModelArtifacts(
        users=users,
        courses=courses,
        transactions=transactions,
        enrollments=enrollments,
        profiles=clustered,
        features=features,
        feature_matrix=feature_matrix,
        cluster_summary=cluster_summary,
        k_metrics=k_metrics,
        pca_frame=pca_frame,
        silhouette=silhouette,
        hierarchical_silhouette=hierarchical_silhouette,
        intra_cluster_similarity=intra_cluster_similarity,
        selected_k=selected_k,
    )


def recommend_courses(
    user_id: str,
    artifacts: ModelArtifacts,
    category_filter: str,
    level_filter: str,
    max_recommendations: int,
) -> pd.DataFrame:
    profile = artifacts.profiles.loc[artifacts.profiles["UserID"] == user_id].iloc[0]
    enrolled_ids = set(artifacts.enrollments.loc[artifacts.enrollments["UserID"] == user_id, "CourseID"])
    cluster_users = artifacts.profiles.loc[
        artifacts.profiles["Cluster"] == profile["Cluster"], "UserID"
    ]
    cluster_enrollments = artifacts.enrollments[artifacts.enrollments["UserID"].isin(cluster_users)]

    popularity = cluster_enrollments.groupby("CourseID").agg(
        cluster_enrollments=("UserID", "count"),
        cluster_avg_spend=("Amount", "mean"),
    )

    recommendations = artifacts.courses.copy().merge(
        popularity, on="CourseID", how="left"
    )
    recommendations[["cluster_enrollments", "cluster_avg_spend"]] = recommendations[
        ["cluster_enrollments", "cluster_avg_spend"]
    ].fillna(0)
    recommendations = recommendations[~recommendations["CourseID"].isin(enrolled_ids)].copy()

    if category_filter != "All":
        recommendations = recommendations[recommendations["CourseCategory"] == category_filter]
    if level_filter != "All":
        recommendations = recommendations[recommendations["CourseLevel"] == level_filter]

    if recommendations.empty:
        return recommendations

    recommendations["rating_component"] = _normalize(recommendations["CourseRating"])
    recommendations["popularity_component"] = _normalize(recommendations["cluster_enrollments"])
    recommendations["category_match"] = (
        recommendations["CourseCategory"] == profile["preferred_course_category"]
    ).astype(float)
    recommendations["level_match"] = (
        recommendations["CourseLevel"] == profile["preferred_course_level"]
    ).astype(float)
    recommendations["type_match"] = (
        recommendations["CourseType"] == profile["preferred_course_type"]
    ).astype(float)
    recommendations["recommendation_score"] = (
        0.35 * recommendations["rating_component"]
        + 0.30 * recommendations["popularity_component"]
        + 0.20 * recommendations["category_match"]
        + 0.10 * recommendations["level_match"]
        + 0.05 * recommendations["type_match"]
    )
    recommendations["Why recommended"] = recommendations.apply(
        lambda row: ", ".join(
            reason
            for reason in [
                "preferred category" if row["category_match"] else "",
                "preferred level" if row["level_match"] else "",
                "popular in segment" if row["popularity_component"] >= 0.5 else "",
                "high course rating" if row["rating_component"] >= 0.7 else "",
            ]
            if reason
        )
        or "balanced fit for this learner segment",
        axis=1,
    )
    return recommendations.sort_values(
        ["recommendation_score", "CourseRating", "cluster_enrollments"],
        ascending=False,
    ).head(max_recommendations)


def build_learning_path(recommendations: pd.DataFrame, preferred_category: str) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    path_pool = recommendations.copy()
    preferred = path_pool[path_pool["CourseCategory"] == preferred_category]
    if len(preferred) >= 2:
        path_pool = preferred

    rows = []
    for level in ["Beginner", "Intermediate", "Advanced"]:
        level_rows = path_pool[path_pool["CourseLevel"] == level]
        if not level_rows.empty:
            row = level_rows.sort_values("recommendation_score", ascending=False).iloc[0]
            rows.append(
                {
                    "Stage": level,
                    "CourseName": row["CourseName"],
                    "CourseCategory": row["CourseCategory"],
                    "CourseRating": row["CourseRating"],
                }
            )
    return pd.DataFrame(rows)


def render_metric_cards(artifacts: ModelArtifacts) -> None:
    total_revenue = artifacts.transactions["Amount"].sum()
    paid_share = (artifacts.transactions["Amount"] > 0).mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learners", f"{len(artifacts.users):,}")
    col2.metric("Enrollments", f"{len(artifacts.transactions):,}")
    col3.metric("Revenue", f"${total_revenue:,.0f}")
    col4.metric("Paid enrollment share", f"{paid_share:.1%}")


def render_overview(artifacts: ModelArtifacts) -> None:
    render_metric_cards(artifacts)

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.scatter(
            artifacts.pca_frame,
            x="PC1",
            y="PC2",
            color="Segment",
            hover_data=["UserID", "UserName"],
            title="Learner Segments in Reduced Feature Space",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=470, legend_title_text="Segment")
        st.plotly_chart(fig, width="stretch")

    with right:
        metrics = artifacts.k_metrics.copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics["k"],
                y=metrics["silhouette_score"],
                mode="lines+markers",
                name="Silhouette",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metrics["k"],
                y=_normalize(metrics["inertia"]),
                mode="lines+markers",
                name="Scaled inertia",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Cluster Selection Diagnostics",
            height=280,
            yaxis_title="Silhouette",
            yaxis2=dict(title="Scaled inertia", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("K selected", artifacts.selected_k)
        col2.metric("K-Means silhouette", f"{artifacts.silhouette:.3f}")
        col3.metric("Hierarchical validation", f"{artifacts.hierarchical_silhouette:.3f}")
        st.metric("Intra-cluster similarity", f"{artifacts.intra_cluster_similarity:.3f}")


def render_segments(artifacts: ModelArtifacts) -> None:
    st.dataframe(
        artifacts.cluster_summary[
            [
                "Segment",
                "learners",
                "total_courses_enrolled",
                "diversity_score",
                "avg_spend_per_learner",
                "paid_enrollment_rate",
                "advanced_share",
                "preferred_course_category",
                "preferred_course_level",
            ]
        ].rename(
            columns={
                "learners": "Learners",
                "total_courses_enrolled": "Avg Enrollments",
                "diversity_score": "Avg Diversity",
                "avg_spend_per_learner": "Avg Spend",
                "paid_enrollment_rate": "Paid Share",
                "advanced_share": "Advanced Share",
                "preferred_course_category": "Top Category",
                "preferred_course_level": "Top Level",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        segment_counts = artifacts.profiles["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Learners"]
        fig = px.bar(
            segment_counts,
            x="Segment",
            y="Learners",
            color="Segment",
            title="Segment Size",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, width="stretch")

    with right:
        radar_metrics = artifacts.cluster_summary[
            [
                "Segment",
                "total_courses_enrolled",
                "diversity_score",
                "avg_spend_per_learner",
                "advanced_share",
                "paid_enrollment_rate",
            ]
        ].copy()
        for column in radar_metrics.columns[1:]:
            radar_metrics[column] = _normalize(radar_metrics[column])
        fig = go.Figure()
        for _, row in radar_metrics.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=row.iloc[1:].tolist(),
                    theta=[
                        "Enrollments",
                        "Diversity",
                        "Spend",
                        "Advanced Share",
                        "Paid Share",
                    ],
                    fill="toself",
                    name=row["Segment"],
                )
            )
        fig.update_layout(title="Segment Comparison", height=380, polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, width="stretch")


def render_profile_and_recommendations(artifacts: ModelArtifacts) -> None:
    profile_options = (
        artifacts.profiles.assign(
            label=lambda frame: frame["UserID"]
            + " | "
            + frame.get("UserName", frame["UserID"]).astype(str)
            + " | "
            + frame["Segment"]
        )
        .sort_values("UserID")
        [["label", "UserID"]]
    )

    left, right = st.columns([0.38, 0.62])
    with left:
        selected_label = st.selectbox("Learner profile", profile_options["label"])
        selected_user_id = profile_options.loc[
            profile_options["label"] == selected_label, "UserID"
        ].iloc[0]
        category_filter = st.selectbox(
            "Category filter",
            ["All"] + sorted(artifacts.courses["CourseCategory"].dropna().unique().tolist()),
        )
        level_filter = st.selectbox(
            "Level filter",
            ["All"] + ["Beginner", "Intermediate", "Advanced"],
        )
        recommendation_count = st.slider("Recommendations", 3, 12, 6)

    learner = artifacts.profiles.loc[artifacts.profiles["UserID"] == selected_user_id].iloc[0]
    user_enrollments = artifacts.enrollments[artifacts.enrollments["UserID"] == selected_user_id]
    recommendations = recommend_courses(
        selected_user_id,
        artifacts,
        category_filter,
        level_filter,
        recommendation_count,
    )

    with right:
        st.subheader(f"{learner['UserID']} - {learner.get('UserName', 'Learner')}")
        cols = st.columns(4)
        cols[0].metric("Segment", learner["Segment"])
        cols[1].metric("Enrolled", int(learner["total_courses_enrolled"]))
        cols[2].metric("Diversity", int(learner["diversity_score"]))
        cols[3].metric("Avg spend", f"${learner['avg_spend_per_learner']:,.0f}")

        st.write(
            f"Preferred path: **{learner['preferred_course_category']}** - "
            f"**{learner['preferred_course_level']}** - **{learner['preferred_course_type']}**"
        )

        recent = user_enrollments.sort_values("TransactionDate", ascending=False).head(6)
        st.dataframe(
            recent[
                [
                    "TransactionDate",
                    "CourseName",
                    "CourseCategory",
                    "CourseLevel",
                    "CourseRating",
                    "Amount",
                ]
            ].rename(
                columns={
                    "TransactionDate": "Date",
                    "CourseName": "Course",
                    "CourseCategory": "Category",
                    "CourseLevel": "Level",
                    "CourseRating": "Rating",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Personalized Recommendations")
    if recommendations.empty:
        st.info("No unseen courses match the selected filters. Clear a filter to expand the pool.")
    else:
        st.dataframe(
            recommendations[
                [
                    "CourseName",
                    "CourseCategory",
                    "CourseLevel",
                    "CourseType",
                    "CourseRating",
                    "cluster_enrollments",
                    "recommendation_score",
                    "Why recommended",
                ]
            ].rename(
                columns={
                    "CourseName": "Course",
                    "CourseCategory": "Category",
                    "CourseLevel": "Level",
                    "CourseType": "Type",
                    "CourseRating": "Rating",
                    "cluster_enrollments": "Segment Enrollments",
                    "recommendation_score": "Score",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        path = build_learning_path(recommendations, learner["preferred_course_category"])
        if not path.empty:
            st.subheader("Recommended Learning Path")
            st.dataframe(path, width="stretch", hide_index=True)


def render_insights(artifacts: ModelArtifacts) -> None:
    enrollments = artifacts.enrollments
    monthly = (
        enrollments.assign(Month=enrollments["TransactionDate"].dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(Enrollments=("TransactionID", "count"), Revenue=("Amount", "sum"))
        .reset_index()
    )
    category = (
        enrollments.groupby("CourseCategory")
        .agg(Enrollments=("TransactionID", "count"), Revenue=("Amount", "sum"), Rating=("CourseRating", "mean"))
        .reset_index()
        .sort_values("Enrollments", ascending=False)
    )

    left, right = st.columns(2)
    with left:
        fig = px.line(monthly, x="Month", y=["Enrollments", "Revenue"], title="Monthly Engagement and Revenue")
        fig.update_layout(height=360, legend_title_text="Metric")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(
            category,
            x="CourseCategory",
            y="Enrollments",
            color="Revenue",
            title="Category Demand and Revenue",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(height=360, xaxis_tickangle=-35)
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        **Research Summary**

        EduPro's learner base is segmented from enrollment volume, category diversity,
        spending behavior, course rating exposure, preferred level, and learning depth.
        The recommendation layer combines content fit with segment-level popularity and
        course quality, so a learner receives courses that match both personal history
        and the behavior of similar learners.

        **Government Stakeholder View**

        The model can support inclusive digital-skilling policy by identifying beginner
        learners who need guided foundation paths, advanced learners ready for deeper
        professional credentials, and cross-domain explorers who may benefit from
        structured career tracks.
        """
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":mortar_board:", layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Model Settings")
        auto_artifacts = build_model(str(DATA_FILE), None)
        auto_k = auto_artifacts.selected_k
        k_choice = st.select_slider(
            "Cluster count",
            options=list(range(3, 8)),
            value=auto_k,
            help="Auto-selected by highest silhouette score. Override to compare segment granularity.",
        )
        artifacts = auto_artifacts if k_choice == auto_k else build_model(str(DATA_FILE), int(k_choice))
        st.caption(f"Dataset: `{DATA_FILE.name}`")
        st.caption(
            f"{artifacts.transactions['TransactionDate'].min():%Y-%m-%d} to "
            f"{artifacts.transactions['TransactionDate'].max():%Y-%m-%d}"
        )

    overview_tab, profile_tab, segments_tab, insights_tab = st.tabs(
        ["Dashboard", "Learner Explorer", "Segment Comparison", "Insights"]
    )

    with overview_tab:
        render_overview(artifacts)
    with profile_tab:
        render_profile_and_recommendations(artifacts)
    with segments_tab:
        render_segments(artifacts)
    with insights_tab:
        render_insights(artifacts)


if __name__ == "__main__":
    main()
