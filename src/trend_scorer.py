"""
Trend Fusion Engine, the core intelligence layer.

Merges sell-through data with social signals via common taxonomy,
computes composite trend scores, and classifies into actionable quadrants.
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from src.config import trend_scoring_config, CATEGORY_TAXONOMY

logger = logging.getLogger(__name__)


def map_hm_product_to_category(product_type: str) -> str:
    """Map an H&M product_type_name to the unified category taxonomy."""
    product_lower = product_type.lower() if isinstance(product_type, str) else ""
    for category, keywords in CATEGORY_TAXONOMY.items():
        for kw in keywords:
            if kw.lower() in product_lower or product_lower in kw.lower():
                return category
    return "Other"


def build_taxonomy_bridge(
    sell_through: pd.DataFrame,
    social_signals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map both scorecards to the unified category taxonomy.

    Returns (sell_through_mapped, social_signals) with 'category' column aligned.
    """
    if "category" not in sell_through.columns:
        sell_through = sell_through.copy()
        sell_through["category"] = sell_through["product_type_name"].apply(
            map_hm_product_to_category
        )

    st_by_category = (
        sell_through
        .groupby("category")
        .agg(
            sell_through_composite=("sell_through_composite", "mean"),
            velocity_score_normalized=("velocity_score_normalized", "mean"),
            total_revenue=("total_revenue", "sum"),
            total_units=("total_units", "sum"),
            wow_growth_rate=("wow_growth_rate", "mean"),
            n_style_color_combos=("product_type_name", "count"),
        )
        .reset_index()
    )

    common = set(st_by_category["category"]) & set(social_signals["category"])
    logger.info(f"Taxonomy bridge: {len(common)} shared categories out of "
                f"{len(set(st_by_category['category']))} sell-through, "
                f"{len(set(social_signals['category']))} social")

    return st_by_category, social_signals


def compute_composite_score(
    merged: pd.DataFrame,
    config=None,
) -> pd.DataFrame:
    """
    Compute the composite trend intelligence score.

    Score = weighted sum of normalized sell-through, sentiment,
    trend momentum, style buzz, and cluster strength.
    """
    config = config or trend_scoring_config
    merged = merged.copy()

    signal_cols = {
        "sell_through_composite": config.w_sell_through,
        "sentiment_normalized": config.w_sentiment,
        "trend_momentum_normalized": config.w_trend_momentum,
        "style_buzz_normalized": config.w_style_buzz,
        "cluster_strength_normalized": config.w_cluster_strength,
    }

    for col in signal_cols:
        if col not in merged.columns:
            logger.warning(f"Missing signal column '{col}', defaulting to 0.5")
            merged[col] = 0.5

    merged["trend_intelligence_score"] = sum(
        merged[col] * weight for col, weight in signal_cols.items()
    )

    merged["social_composite"] = (
        config.w_sentiment * merged["sentiment_normalized"]
        + config.w_trend_momentum * merged["trend_momentum_normalized"]
        + config.w_style_buzz * merged["style_buzz_normalized"]
        + config.w_cluster_strength * merged["cluster_strength_normalized"]
    ) / (1 - config.w_sell_through)

    logger.info(f"Composite scores computed for {len(merged)} categories")
    return merged


def classify_quadrants(
    df: pd.DataFrame,
    social_col: str = "social_composite",
    sales_col: str = "sell_through_composite",
    threshold: float | str | None = "median",
) -> pd.DataFrame:
    """
    Classify each category into trend quadrants.

    HOT: high social + high sales -> over-index aggressively
    EMERGING: high social + low sales -> test and invest
    FADING: low social + high sales -> maintain, do not grow
    COLD: low social + low sales -> deprioritize

    threshold: "median" (data-driven, recommended), or a fixed float like 0.5.
    """
    df = df.copy()

    if threshold == "median":
        social_threshold = df[social_col].median()
        sales_threshold = df[sales_col].median()
        logger.info(f"Quadrant thresholds (median): social={social_threshold:.3f}, "
                     f"sales={sales_threshold:.3f}")
    else:
        t = threshold if isinstance(threshold, (int, float)) else trend_scoring_config.quadrant_threshold
        social_threshold = t
        sales_threshold = t
        logger.info(f"Quadrant thresholds (fixed): {t}")

    high_social = df[social_col] >= social_threshold
    high_sales = df[sales_col] >= sales_threshold

    df["quadrant"] = np.select(
        [high_social & high_sales, high_social & ~high_sales,
         ~high_social & high_sales],
        ["HOT", "EMERGING", "FADING"],
        default="COLD",
    )

    df["recommendation"] = np.select(
        [df["quadrant"] == "HOT", df["quadrant"] == "EMERGING",
         df["quadrant"] == "FADING"],
        [
            "Over-index aggressively, strong demand + rising social signals",
            "Test & invest, high social buzz, sales haven't caught up yet",
            "Maintain current levels, selling well but social interest waning",
        ],
        default="Deprioritize, low demand and weak social signals",
    )

    for q in ["HOT", "EMERGING", "FADING", "COLD"]:
        count = (df["quadrant"] == q).sum()
        logger.info(f"  {q}: {count} categories")

    return df


def build_trend_intelligence_master(
    sell_through: pd.DataFrame,
    social_signals: pd.DataFrame,
    config=None,
) -> pd.DataFrame:
    """
    End-to-end trend intelligence: taxonomy mapping, scoring, classification.
    """
    st_mapped, ss = build_taxonomy_bridge(sell_through, social_signals)

    merged = st_mapped.merge(ss, on="category", how="outer")

    fill_defaults = {
        "sell_through_composite": 0.5,
        "sentiment_normalized": 0.5,
        "trend_momentum_normalized": 0.5,
        "style_buzz_normalized": 0.5,
        "cluster_strength_normalized": 0.5,
    }
    merged = merged.fillna(fill_defaults)

    merged = compute_composite_score(merged, config)
    merged = classify_quadrants(merged)

    merged = merged.sort_values("trend_intelligence_score", ascending=False)

    logger.info(f"Trend intelligence master: {len(merged)} categories")
    return merged


def generate_top_n_recommendations(
    master: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract the top-N actionable recommendations."""
    recs = (
        master
        .sort_values("trend_intelligence_score", ascending=False)
        .head(top_n)
        [["category", "quadrant", "trend_intelligence_score",
          "sell_through_composite", "social_composite",
          "dominant_style", "recommendation"]]
        .copy()
    )
    recs["rank"] = range(1, len(recs) + 1)
    recs["confidence"] = (recs["trend_intelligence_score"] * 100).round(1)
    return recs
