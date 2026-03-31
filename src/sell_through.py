"""
Sell-through analytics pipeline.

Computes velocity, sell-through rate, revenue intensity, and week-over-week
growth from H&M transaction data. Outputs a scorecard at the
(product_type, colour_group) level.
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from src.config import hm_config, RANDOM_SEED

logger = logging.getLogger(__name__)


def enrich_transactions(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """Join transactions with article attributes."""
    merge_cols = [hm_config.article_id_column] + hm_config.key_attributes
    available = [c for c in merge_cols if c in articles.columns]
    enriched = transactions.merge(
        articles[available],
        on=hm_config.article_id_column,
        how="left",
    )
    enriched["year_week"] = (
        enriched[hm_config.date_column].dt.isocalendar().year.astype(str)
        + "-W"
        + enriched[hm_config.date_column].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    enriched["year_month"] = enriched[hm_config.date_column].dt.to_period("M").astype(str)
    logger.info(f"Enriched transactions: {len(enriched):,} rows")
    return enriched


def compute_velocity(
    enriched: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Units sold per week, aggregated by group columns."""
    group_cols = group_cols or ["product_type_name", "colour_group_name"]
    weekly = (
        enriched
        .groupby(group_cols + ["year_week"])
        .agg(
            units_sold=(hm_config.article_id_column, "count"),
            revenue=(hm_config.price_column, "sum"),
        )
        .reset_index()
    )
    velocity = (
        weekly
        .groupby(group_cols)
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            weeks_active=("year_week", "nunique"),
            avg_weekly_units=("units_sold", "mean"),
            median_weekly_units=("units_sold", "median"),
        )
        .reset_index()
    )
    velocity["velocity_score"] = velocity["avg_weekly_units"]
    velocity["revenue_per_sku"] = velocity["total_revenue"] / velocity["weeks_active"]
    return velocity.sort_values("velocity_score", ascending=False)


def compute_wow_growth(
    enriched: pd.DataFrame,
    group_cols: list[str] | None = None,
    trailing_weeks: int = 8,
) -> pd.DataFrame:
    """Week-over-week growth rate for the trailing N weeks."""
    group_cols = group_cols or ["product_type_name", "colour_group_name"]
    weekly = (
        enriched
        .groupby(group_cols + ["year_week"])
        .agg(units_sold=(hm_config.article_id_column, "count"))
        .reset_index()
        .sort_values("year_week")
    )
    all_weeks = sorted(weekly["year_week"].unique())
    recent_weeks = all_weeks[-trailing_weeks:] if len(all_weeks) >= trailing_weeks else all_weeks
    midpoint = len(recent_weeks) // 2
    first_half_weeks = set(recent_weeks[:midpoint])
    second_half_weeks = set(recent_weeks[midpoint:])

    recent = weekly[weekly["year_week"].isin(set(recent_weeks))]
    first_half = (
        recent[recent["year_week"].isin(first_half_weeks)]
        .groupby(group_cols)["units_sold"]
        .sum()
        .rename("first_half_units")
    )
    second_half = (
        recent[recent["year_week"].isin(second_half_weeks)]
        .groupby(group_cols)["units_sold"]
        .sum()
        .rename("second_half_units")
    )
    growth = pd.concat([first_half, second_half], axis=1).fillna(0)
    growth["wow_growth_rate"] = np.where(
        growth["first_half_units"] > 0,
        (growth["second_half_units"] - growth["first_half_units"]) / growth["first_half_units"],
        np.where(growth["second_half_units"] > 0, 1.0, 0.0),
    )
    return growth.reset_index()


def compute_seasonal_decomposition(
    enriched: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Monthly sales volume by category for seasonal pattern analysis."""
    group_cols = group_cols or ["product_type_name"]
    monthly = (
        enriched
        .groupby(group_cols + ["year_month"])
        .agg(units_sold=(hm_config.article_id_column, "count"))
        .reset_index()
    )
    return monthly


def build_sell_through_scorecard(
    enriched: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the complete sell-through scorecard.

    One row per (product_type, colour_group) with velocity, growth,
    and revenue metrics.
    """
    group_cols = group_cols or ["product_type_name", "colour_group_name"]

    velocity = compute_velocity(enriched, group_cols)
    growth = compute_wow_growth(enriched, group_cols)

    scorecard = velocity.merge(
        growth[group_cols + ["wow_growth_rate", "first_half_units", "second_half_units"]],
        on=group_cols,
        how="left",
    )
    scorecard["wow_growth_rate"] = scorecard["wow_growth_rate"].fillna(0)

    for col in ["velocity_score", "total_revenue", "revenue_per_sku"]:
        norm_col = f"{col}_normalized"
        col_min, col_max = scorecard[col].min(), scorecard[col].max()
        scorecard[norm_col] = np.where(
            col_max > col_min,
            (scorecard[col] - col_min) / (col_max - col_min),
            0.5,
        )

    scorecard["sell_through_composite"] = (
        0.50 * scorecard["velocity_score_normalized"]
        + 0.30 * scorecard["total_revenue_normalized"]
        + 0.20 * scorecard["wow_growth_rate"].clip(-1, 1).add(1).div(2)
    )

    logger.info(f"Sell-through scorecard: {len(scorecard):,} style-color combos")
    return scorecard.sort_values("sell_through_composite", ascending=False)


def classify_sell_through(
    scorecard: pd.DataFrame,
    velocity_threshold: float = 0.6,
    growth_threshold: float = 0.05,
) -> pd.DataFrame:
    """Tag each style-color as Rising Star, Fading Favorite, Steady, or Dormant."""
    conditions = [
        (scorecard["velocity_score_normalized"] >= velocity_threshold)
        & (scorecard["wow_growth_rate"] >= growth_threshold),
        (scorecard["velocity_score_normalized"] >= velocity_threshold)
        & (scorecard["wow_growth_rate"] < growth_threshold),
        (scorecard["velocity_score_normalized"] < velocity_threshold)
        & (scorecard["wow_growth_rate"] >= growth_threshold),
    ]
    labels = ["Rising Star", "Fading Favorite", "Emerging", "Dormant"]
    scorecard = scorecard.copy()
    scorecard["sell_through_class"] = np.select(conditions, labels[:3], default=labels[3])
    return scorecard
