"""
Google Trends data retrieval and momentum scoring.

Uses pytrends to pull search interest data for fashion keywords,
then computes momentum scores and breakout detection.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np

from src.config import google_trends_config

logger = logging.getLogger(__name__)


def fetch_trends_data(
    keywords: list[str] | None = None,
    timeframe: str | None = None,
    geo: str | None = None,
    batch_size: int = 5,
    sleep_between: float = 15.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch Google Trends interest-over-time for fashion keywords.

    Queries in batches of 5 (API limit) with rate limiting and
    exponential backoff on 429 responses.
    Returns a DataFrame with date index and keyword columns.
    """
    from pytrends.request import TrendReq

    keywords = keywords or google_trends_config.fashion_keywords
    timeframe = timeframe or google_trends_config.timeframe
    geo = geo or google_trends_config.geo

    pytrends = TrendReq(hl="en-US", tz=360)
    all_data = []
    total_batches = (len(keywords) + batch_size - 1) // batch_size

    for i in range(0, len(keywords), batch_size):
        batch = keywords[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Fetching trends batch {batch_num}/{total_batches}: {batch}")

        for attempt in range(1, max_retries + 1):
            try:
                pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
                df = pytrends.interest_over_time()
                if not df.empty and "isPartial" in df.columns:
                    df = df.drop("isPartial", axis=1)
                all_data.append(df)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = sleep_between * (2 ** (attempt - 1))
                    logger.warning(f"Rate limited (429) on batch {batch_num}, "
                                   f"retry {attempt}/{max_retries} in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"Failed batch {batch_num} after {attempt} attempts: {e}")
                    break

        if i + batch_size < len(keywords):
            time.sleep(sleep_between)

    if not all_data:
        logger.error("No trends data retrieved")
        return pd.DataFrame()

    combined = pd.concat(all_data, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    logger.info(f"Trends data: {combined.shape[0]} dates x {combined.shape[1]} keywords")
    return combined


def compute_trend_momentum(
    trends_df: pd.DataFrame,
    trailing_weeks: int = 12,
) -> pd.DataFrame:
    """
    Compute momentum (slope) and breakout flags for each keyword.

    Momentum = linear regression slope over trailing N weeks.
    Breakout = latest value > 1.5 * rolling mean.
    """
    if trends_df.empty:
        return pd.DataFrame(
            columns=["keyword", "momentum", "current_interest", "mean_interest",
                     "breakout_flag", "category"]
        )

    recent = trends_df.tail(trailing_weeks)
    results = []

    for keyword in trends_df.columns:
        series = recent[keyword].dropna()
        if len(series) < 4:
            continue

        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]

        current = series.iloc[-1]
        mean_val = series.mean()
        breakout = current > 1.5 * mean_val if mean_val > 0 else False

        category = google_trends_config.keyword_category_map.get(keyword, "Other")

        results.append({
            "keyword": keyword,
            "momentum": slope,
            "current_interest": int(current),
            "mean_interest": round(float(mean_val), 1),
            "breakout_flag": breakout,
            "category": category,
        })

    momentum_df = pd.DataFrame(results)
    logger.info(f"Trend momentum: {len(momentum_df)} keywords, "
                f"{momentum_df['breakout_flag'].sum()} breakouts")
    return momentum_df


def aggregate_trends_by_category(
    momentum_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate trend signals to the category level."""
    if momentum_df.empty:
        return pd.DataFrame(columns=["category", "trend_momentum_normalized", "breakout_flag"])

    agg = (
        momentum_df
        .groupby("category")
        .agg(
            avg_momentum=("momentum", "mean"),
            max_momentum=("momentum", "max"),
            avg_interest=("current_interest", "mean"),
            breakout_count=("breakout_flag", "sum"),
            keyword_count=("keyword", "count"),
        )
        .reset_index()
    )
    agg["breakout_flag"] = agg["breakout_count"] > 0
    agg["trend_momentum_normalized"] = _min_max_normalize(agg["avg_momentum"])
    return agg


def _min_max_normalize(series: pd.Series) -> pd.Series:
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)
