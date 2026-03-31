"""
Social signal aggregation utilities.

Aggregates outputs from the three ML pipelines (sentiment, zero-shot, clusters)
and Google Trends into a unified social signal scorecard.
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from src.config import reviews_config, REVIEW_CLASS_TO_CATEGORY

logger = logging.getLogger(__name__)


def map_reviews_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Map review Class Name to the unified category taxonomy."""
    df = df.copy()
    df["category"] = df[reviews_config.class_column].map(REVIEW_CLASS_TO_CATEGORY)
    unmapped = df["category"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} reviews have unmapped class names")
        df["category"] = df["category"].fillna("Other")
    return df


def aggregate_sentiment_by_category(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Aggregate fine-tuned sentiment scores per category."""
    agg = (
        df.groupby("category")
        .agg(
            mean_sentiment=(sentiment_col, "mean"),
            median_sentiment=(sentiment_col, "median"),
            std_sentiment=(sentiment_col, "std"),
            review_count=(sentiment_col, "count"),
            positive_ratio=(sentiment_col, lambda x: (x > 0.5).mean()),
        )
        .reset_index()
    )
    agg["sentiment_normalized"] = _min_max_normalize(agg["mean_sentiment"])
    return agg


def aggregate_style_distribution(
    style_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate zero-shot style classifications per category.

    Returns a pivot: category x style_label with buzz proportions.
    """
    merged = style_df.copy()
    merged["category"] = reviews_df["category"].values

    style_counts = (
        merged
        .groupby(["category", "style_label"])
        .size()
        .reset_index(name="count")
    )
    totals = style_counts.groupby("category")["count"].sum().rename("total")
    style_counts = style_counts.merge(totals, on="category")
    style_counts["proportion"] = style_counts["count"] / style_counts["total"]

    dominant_style = (
        style_counts
        .sort_values("proportion", ascending=False)
        .groupby("category")
        .first()
        .reset_index()
        .rename(columns={"style_label": "dominant_style", "proportion": "dominant_proportion"})
    )

    style_entropy = (
        style_counts
        .groupby("category")
        .apply(
            lambda g: -np.sum(g["proportion"] * np.log(g["proportion"] + 1e-10)),
            include_groups=False,
        )
        .rename("style_diversity")
        .reset_index()
    )

    result = dominant_style[["category", "dominant_style", "dominant_proportion"]].merge(
        style_entropy, on="category"
    )
    result["style_buzz_normalized"] = _min_max_normalize(result["dominant_proportion"])
    return result


def aggregate_cluster_strength(
    cluster_labels: np.ndarray,
    categories: pd.Series,
    cluster_label_map: dict[int, str],
) -> pd.DataFrame:
    """Compute cluster density/strength per category."""
    df = pd.DataFrame({
        "category": categories.values,
        "cluster": cluster_labels,
    })
    df = df[df["cluster"] != -1]

    cluster_sizes = df.groupby(["category", "cluster"]).size().reset_index(name="size")
    cat_totals = df.groupby("category").size().rename("total")
    cluster_sizes = cluster_sizes.merge(cat_totals, on="category")
    cluster_sizes["density"] = cluster_sizes["size"] / cluster_sizes["total"]

    top_cluster = (
        cluster_sizes
        .sort_values("density", ascending=False)
        .groupby("category")
        .first()
        .reset_index()
    )
    top_cluster["cluster_theme"] = top_cluster["cluster"].map(cluster_label_map)
    top_cluster["cluster_strength_normalized"] = _min_max_normalize(top_cluster["density"])

    return top_cluster[["category", "cluster", "cluster_theme", "density", "cluster_strength_normalized"]]


def build_social_signal_scorecard(
    sentiment_agg: pd.DataFrame,
    style_agg: pd.DataFrame,
    cluster_agg: pd.DataFrame,
    trends_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge all social signal aggregations into a single scorecard.

    One row per category.
    """
    scorecard = sentiment_agg.merge(style_agg, on="category", how="outer")
    scorecard = scorecard.merge(cluster_agg, on="category", how="outer")

    if trends_agg is not None:
        scorecard = scorecard.merge(trends_agg, on="category", how="outer")
    else:
        scorecard["trend_momentum_normalized"] = 0.5
        scorecard["breakout_flag"] = False

    fill_cols = [
        "sentiment_normalized", "style_buzz_normalized",
        "cluster_strength_normalized", "trend_momentum_normalized",
    ]
    for col in fill_cols:
        if col in scorecard.columns:
            scorecard[col] = scorecard[col].fillna(0.5)

    logger.info(f"Social signal scorecard: {len(scorecard)} categories")
    return scorecard


def _min_max_normalize(series: pd.Series) -> pd.Series:
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)
