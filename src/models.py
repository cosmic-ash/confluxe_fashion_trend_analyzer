"""
ML model wrappers for the social signal processing pipeline.

- Pipeline A: Fine-tuned RoBERTa sentiment (LoRA via PEFT)
- Pipeline B: BART zero-shot style classification
- Pipeline C: MiniLM embeddings + HDBSCAN clustering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    sentiment_model_config,
    zero_shot_config,
    embedding_config,
    MODELS_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Pipeline A: Fine-Tuned Fashion Sentiment (LoRA)


class ReviewDataset(Dataset):
    """Tokenized review dataset for sentiment fine tuning."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def prepare_sentiment_labels(
    df: pd.DataFrame,
    rating_col: str = "Rating",
    recommend_col: str = "Recommended IND",
) -> pd.Series:
    """
    Convert ratings + recommendation into 3 class sentiment labels.
    0=negative (rating<=2), 1=neutral (rating==3), 2=positive (rating>=4 & recommended)
    """
    labels = pd.Series(1, index=df.index, name="sentiment_label")
    labels[df[rating_col] <= 2] = 0
    labels[(df[rating_col] >= 4) & (df[recommend_col] == 1)] = 2
    return labels


def fine_tune_sentiment_model(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    config=None,
) -> tuple[Any, Any]:
    """
    Fine-tune RoBERTa for fashion sentiment using LoRA.

    Returns (model, tokenizer) with LoRA adapters applied.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    config = config or sentiment_model_config
    logger.info(f"Loading base model: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=3,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({trainable/total:.2%})")

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=config.random_seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(str(MODELS_DIR))
    tokenizer.save_pretrained(str(MODELS_DIR))
    logger.info(f"Saved fine-tuned model to {MODELS_DIR}")

    return model, tokenizer


def load_fine_tuned_sentiment(config=None):
    """Load a previously fine-tuned sentiment model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    config = config or sentiment_model_config
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=3,
    )
    model = PeftModel.from_pretrained(base_model, str(MODELS_DIR))
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_sentiment(
    texts: list[str],
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Run sentiment inference on a list of texts. Returns array of shape (N, 3)."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def baseline_vader_sentiment(texts: list[str]) -> np.ndarray:
    """VADER baseline sentiment for comparison. Returns compound scores."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])



# Pipeline B: Zero-Shot Style Classification


def classify_styles_zero_shot(
    texts: list[str],
    config=None,
    batch_size: int | None = None,
    max_samples: int | None = 5000,
) -> pd.DataFrame:
    """
    Classify texts into style tribes using BART zero-shot NLI.
    Args:
        max_samples: Cap the number of texts to classify. Zero-shot runs one NLI
            pass per (text, label) pair, so 23K texts × 10 labels = 230K inferences.
            Sampling to 5K gives equivalent category-level statistics ~4.5× faster.
            Set to None to process all texts.

    Returns DataFrame with columns: text, style_label, confidence.
    """
    
    from transformers import pipeline
    

    config = config or zero_shot_config
    batch_size = batch_size or config.batch_size

    if max_samples and len(texts) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in indices]
        logger.info(f"Sampled {max_samples:,} / {len(texts) + max_samples:,} texts for zero-shot")

    use_fp16 = torch.cuda.is_available()

    classifier = pipeline(
        "zero-shot-classification",
        model=config.model_name,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    )

    total_batches = (len(texts) + batch_size - 1) // batch_size
    results = []
    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx : batch_idx + batch_size]
        outputs = classifier(batch, config.style_labels, multi_label=False)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for text, out in zip(batch, outputs):
            results.append({
                "text": text[:200],
                "style_label": out["labels"][0],
                "confidence": out["scores"][0],
                "all_labels": dict(zip(out["labels"], out["scores"])),
            })
        done = min(batch_idx + batch_size, len(texts))
        if (done // batch_size) % 20 == 0 or done == len(texts):
            logger.info(f"Zero-shot progress: {done:,}/{len(texts):,} texts ({done/len(texts):.0%})")

    logger.info(f"Zero-shot classified {len(results):,} texts")
    return pd.DataFrame(results)


# Pipeline C: Semantic Embedding Clusters


def compute_embeddings(
    texts: list[str],
    config=None,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute sentence embeddings using MiniLM."""
    from sentence_transformers import SentenceTransformer

    config = config or embedding_config
    model = SentenceTransformer(config.model_name, device=str(DEVICE))
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    config=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    UMAP reduction + HDBSCAN clustering.

    Returns (umap_2d, cluster_labels).
    """
    import umap
    import hdbscan

    config = config or embedding_config

    reducer = umap.UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_seed,
    )
    umap_2d = reducer.fit_transform(embeddings)
    logger.info(f"UMAP reduction: {embeddings.shape} -> {umap_2d.shape}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(umap_2d)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    return umap_2d, cluster_labels


def label_clusters(
    texts: list[str],
    cluster_labels: np.ndarray,
    top_n_terms: int = 5,
) -> dict[int, str]:
    """Generate human readable labels for each cluster using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_labels_map = {}
    unique_labels = sorted(set(cluster_labels))

    for label in unique_labels:
        if label == -1:
            cluster_labels_map[-1] = "Noise / Unclustered"
            continue
        cluster_texts = [t for t, l in zip(texts, cluster_labels) if l == label]
        if len(cluster_texts) < 3:
            cluster_labels_map[label] = f"Cluster {label} (small)"
            continue
        vectorizer = TfidfVectorizer(
            max_features=200, stop_words="english",
            ngram_range=(1, 2), min_df=2,
        )
        tfidf = vectorizer.fit_transform(cluster_texts)
        mean_tfidf = tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-top_n_terms:][::-1]
        terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        cluster_labels_map[label] = " | ".join(terms)

    logger.info(f"Labeled {len(cluster_labels_map)} clusters")
    return cluster_labels_map
