"""
Plotly visualization components for the trend intelligence dashboard.

Each function returns a plotly Figure object that can be displayed
in Jupyter or exported.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PLOTLY_TEMPLATE = "plotly_white"
COLOR_QUADRANT = {
    "HOT": "#FF4136",
    "EMERGING": "#FF851B",
    "FADING": "#AAAAAA",
    "COLD": "#0074D9",
}


def trend_quadrant_scatter(
    df: pd.DataFrame,
    social_col: str = "social_composite",
    sales_col: str = "sell_through_composite",
    size_col: str = "total_revenue",
    label_col: str = "category",
) -> go.Figure:
    """
    The hero chart: Trend Quadrant Scatter Plot.

    x = Social Signal Score, y = Sell-Through Score
    size = Revenue, color = Quadrant
    """
    fig = px.scatter(
        df,
        x=social_col,
        y=sales_col,
        size=size_col,
        color="quadrant",
        color_discrete_map=COLOR_QUADRANT,
        text=label_col,
        hover_data=["trend_intelligence_score", "dominant_style"],
        template=PLOTLY_TEMPLATE,
        title="Trend Intelligence Quadrant: What to Over-Index On",
        labels={
            social_col: "Social Signal Score",
            sales_col: "Sell-Through Score",
        },
    )

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

    fig.add_annotation(x=0.75, y=0.85, text="HOT", showarrow=False,
                       font=dict(size=16, color=COLOR_QUADRANT["HOT"]), opacity=0.3)
    fig.add_annotation(x=0.75, y=0.15, text="EMERGING", showarrow=False,
                       font=dict(size=16, color=COLOR_QUADRANT["EMERGING"]), opacity=0.3)
    fig.add_annotation(x=0.25, y=0.85, text="FADING", showarrow=False,
                       font=dict(size=16, color=COLOR_QUADRANT["FADING"]), opacity=0.3)
    fig.add_annotation(x=0.25, y=0.15, text="COLD", showarrow=False,
                       font=dict(size=16, color=COLOR_QUADRANT["COLD"]), opacity=0.3)

    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=600, width=900, showlegend=True)
    return fig


def style_color_heatmap(
    df: pd.DataFrame,
    row_col: str = "product_type_name",
    col_col: str = "colour_group_name",
    value_col: str = "sell_through_composite",
    top_n_rows: int = 15,
    top_n_cols: int = 12,
) -> go.Figure:
    """Style x Color heatmap colored by composite score."""
    top_rows = df.groupby(row_col)[value_col].mean().nlargest(top_n_rows).index
    top_cols = df.groupby(col_col)[value_col].mean().nlargest(top_n_cols).index

    filtered = df[df[row_col].isin(top_rows) & df[col_col].isin(top_cols)]
    pivot = filtered.pivot_table(values=value_col, index=row_col, columns=col_col, aggfunc="mean")

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        title="Style x Color Sell-Through Heatmap",
        labels=dict(color="Composite Score"),
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(height=500, width=900)
    return fig


def trend_momentum_timeseries(
    trends_df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """Line chart of Google Trends interest over time for top keywords."""
    if trends_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Google Trends data available", showarrow=False)
        return fig

    means = trends_df.mean().nlargest(top_n)
    fig = go.Figure()
    for keyword in means.index:
        fig.add_trace(go.Scatter(
            x=trends_df.index,
            y=trends_df[keyword],
            name=keyword,
            mode="lines",
        ))

    fig.update_layout(
        title="Fashion Trend Momentum: Google Trends Interest Over Time",
        xaxis_title="Date",
        yaxis_title="Search Interest (0–100)",
        template=PLOTLY_TEMPLATE,
        height=450,
        width=900,
        hovermode="x unified",
    )
    return fig


def category_treemap(
    df: pd.DataFrame,
    value_col: str = "trend_intelligence_score",
) -> go.Figure:
    """Treemap of categories sized by trend intelligence score."""
    fig = px.treemap(
        df,
        path=["quadrant", "category"],
        values=value_col,
        color="quadrant",
        color_discrete_map=COLOR_QUADRANT,
        title="Category Drill-Down by Trend Intelligence Score",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=500, width=900)
    return fig


def recommendations_table(recs_df: pd.DataFrame) -> go.Figure:
    """Interactive table of top N recommendations."""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Rank", "Category", "Quadrant", "Score", "Confidence",
                    "Dominant Style", "Recommendation"],
            fill_color="#2C3E50",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[
                recs_df["rank"],
                recs_df["category"],
                recs_df["quadrant"],
                recs_df["trend_intelligence_score"].round(3),
                recs_df["confidence"].astype(str) + "%",
                recs_df.get("dominant_style", "—"),
                recs_df["recommendation"],
            ],
            fill_color=[
                ["white"] * len(recs_df),
                ["white"] * len(recs_df),
                [COLOR_QUADRANT.get(q, "#EEE") for q in recs_df["quadrant"]],
                ["white"] * len(recs_df),
                ["white"] * len(recs_df),
                ["white"] * len(recs_df),
                ["white"] * len(recs_df),
            ],
            align="left",
            font=dict(size=11),
            height=30,
        ),
    )])
    fig.update_layout(
        title="Top Trend Recommendations: Styles to Over-Index On",
        height=400, width=1100,
    )
    return fig


def methodology_waterfall(
    category: str,
    scores: dict[str, float],
    weights: dict[str, float],
) -> go.Figure:
    """Horizontal bar chart showing signal contribution to a category's score."""
    components = list(scores.keys())
    contributions = [scores[c] * weights.get(c, 0) for c in components]
    total = sum(contributions)

    sorted_pairs = sorted(zip(components, contributions), key=lambda x: x[1])
    components = [p[0] for p in sorted_pairs]
    contributions = [p[1] for p in sorted_pairs]
    weight_pcts = [f"{weights.get(c, 0) * 100:.0f}%" for c in components]
    raw_scores = [scores[c] for c in components]

    colors = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C", "#9B59B6"]
    bar_colors = [colors[i % len(colors)] for i in range(len(components))]

    labels = [f"{c}  (wt: {w})" for c, w in zip(components, weight_pcts)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=contributions,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="white", width=1.5),
        ),
        text=[f"<b>{v:.3f}</b>  (signal: {s:.2f} × {w})"
              for v, s, w in zip(contributions, raw_scores, weight_pcts)],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(
        x=total,
        line_dash="dash",
        line_color="#2C3E50",
        line_width=2,
        annotation_text=f"Total: {total:.3f}",
        annotation_position="top",
        annotation_font=dict(size=13, color="#2C3E50"),
    )

    fig.update_layout(
        title=dict(
            text=f"Score Decomposition — <b>{category}</b>",
            font=dict(size=16),
        ),
        xaxis_title="Contribution to Trend Intelligence Score",
        xaxis=dict(range=[0, max(total * 1.35, 0.15)], gridcolor="#ECF0F1"),
        yaxis=dict(tickfont=dict(size=12)),
        template=PLOTLY_TEMPLATE,
        height=320,
        width=800,
        margin=dict(l=180, r=120, t=60, b=50),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


def umap_cluster_scatter(
    umap_2d: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_label_map: dict[int, str] | None = None,
    texts: list[str] | None = None,
) -> go.Figure:
    """Interactive UMAP scatter plot colored by cluster."""
    df = pd.DataFrame({
        "UMAP_1": umap_2d[:, 0],
        "UMAP_2": umap_2d[:, 1],
        "cluster": cluster_labels.astype(str),
    })
    if cluster_label_map:
        df["cluster_name"] = [cluster_label_map.get(int(c), f"Cluster {c}") for c in cluster_labels]
    else:
        df["cluster_name"] = df["cluster"]
    if texts:
        df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t for t in texts]

    hover_data = ["cluster_name"]
    if texts:
        hover_data.append("text_preview")

    fig = px.scatter(
        df,
        x="UMAP_1", y="UMAP_2",
        color="cluster_name",
        hover_data=hover_data,
        title="Semantic Trend Clusters (UMAP + HDBSCAN)",
        template=PLOTLY_TEMPLATE,
        opacity=0.6,
    )
    fig.update_layout(height=600, width=900, showlegend=True)
    return fig


def sentiment_comparison_bar(
    metrics: dict[str, dict[str, float]],
) -> go.Figure:
    """Bar chart comparing model performance (fine-tuned vs base vs VADER)."""
    models = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    fig = go.Figure()
    for metric in metric_names:
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=[metrics[m][metric] for m in models],
            text=[f"{metrics[m][metric]:.3f}" for m in models],
            textposition="auto",
        ))

    fig.update_layout(
        title="Sentiment Model Performance Comparison",
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=400,
        width=700,
        yaxis_title="Score",
    )
    return fig
