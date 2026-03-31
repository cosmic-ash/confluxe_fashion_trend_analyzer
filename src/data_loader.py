"""
Data loading and validation utilities.

Handles loading CSVs/parquet, schema validation, and basic data quality checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.config import (
    DATA_RAW,
    DATA_RAW_HM,
    DATA_RAW_REVIEWS,
    DATA_PROCESSED,
    hm_config,
    reviews_config,
)

logger = logging.getLogger(__name__)


#Schema Validation

def validate_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Raise ValueError if required columns are missing."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    logger.info(f"[{dataset_name}] Schema OK {len(df):,} rows, {len(df.columns)} cols")


def log_data_quality(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Log and return basic data quality metrics."""
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
    }
    logger.info(
        f"[{dataset_name}] {stats['rows']:,} rows | "
        f"{stats['memory_mb']:.1f} MB | "
        f"Null cols: {sum(1 for v in stats['null_pct'].values() if v > 0)}"
    )
    return stats


# H&M Data

def load_hm_articles(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate H&M articles metadata."""
    data_dir = data_dir or DATA_RAW_HM
    path = data_dir / hm_config.articles_file
    df = pd.read_csv(path, dtype={"article_id": str})
    validate_schema(df, hm_config.key_attributes + [hm_config.article_id_column], "HM-Articles")
    log_data_quality(df, "HM-Articles")
    return df


def load_hm_transactions(
    data_dir: Path | None = None,
    sample_frac: float | None = None,
) -> pd.DataFrame:
    """Load H&M transactions with date parsing and optional sampling."""
    data_dir = data_dir or DATA_RAW_HM
    path = data_dir / hm_config.transactions_file
    df = pd.read_csv(
        path,
        dtype={"article_id": str, "customer_id": str},
        parse_dates=[hm_config.date_column],
    )
    validate_schema(
        df,
        [hm_config.date_column, hm_config.price_column,
         hm_config.article_id_column, hm_config.customer_id_column],
        "HM-Transactions",
    )
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        logger.info(f"Sampled {len(df):,} transactions ({sample_frac:.0%})")
    log_data_quality(df, "HM-Transactions")
    return df


#Reviews Data

def load_reviews(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and clean Women's E-Commerce Clothing Reviews."""
    data_dir = data_dir or DATA_RAW_REVIEWS
    path = data_dir / reviews_config.file
    df = pd.read_csv(path)
    validate_schema(
        df,
        [reviews_config.text_column, reviews_config.rating_column,
         reviews_config.department_column, reviews_config.class_column],
        "Reviews",
    )
    pre_count = len(df)
    df = df.dropna(subset=[reviews_config.text_column])
    df = df[df[reviews_config.text_column].str.len() >= reviews_config.min_review_length]
    df = df.reset_index(drop=True)
    logger.info(f"Reviews: {pre_count:,} -> {len(df):,} after cleaning")
    log_data_quality(df, "Reviews")
    return df


# Parquet I/O

def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame as parquet in the processed directory."""
    path = DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved {name} -> {path} ({len(df):,} rows)")
    return path


def load_processed(name: str) -> pd.DataFrame:
    """Load a processed parquet file."""
    path = DATA_PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {name} <- {path} ({len(df):,} rows)")
    return df
