"""
Central configuration for the Fashion Trend Intelligence Pipeline.

All paths, thresholds, model hyperparameters, and scoring weights live here.
"""

from pathlib import Path
from pydantic import BaseModel, Field


#Environment Detection

IS_COLAB = Path("/content").exists()

#Paths 

if IS_COLAB:
    DRIVE_PROJECT = Path("/content/drive/MyDrive/FashionTrendAnalyzer")
    PROJECT_ROOT = DRIVE_PROJECT
    DATA_RAW_HM = DRIVE_PROJECT /"data"/"raw"
    DATA_RAW_REVIEWS = DRIVE_PROJECT/"data"/"raw"
    DATA_RAW = DRIVE_PROJECT/"data"/"raw"
    DATA_PROCESSED = DRIVE_PROJECT/"data"/"processed"
    MODELS_DIR = DRIVE_PROJECT/"models"/"fashion_sentiment"
    NOTEBOOKS_DIR = DRIVE_PROJECT/"notebooks"
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_RAW_HM = PROJECT_ROOT/"data"/"raw"
    DATA_RAW_REVIEWS = PROJECT_ROOT/"data"/"raw"
    DATA_RAW = PROJECT_ROOT/"data"/"raw"
    DATA_PROCESSED = PROJECT_ROOT/"data"/"processed"
    MODELS_DIR = PROJECT_ROOT/"models"/"fashion_sentiment"
    NOTEBOOKS_DIR = PROJECT_ROOT/"notebooks"

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


#H&M Data Config 

class HMDataConfig(BaseModel):
    articles_file: str = "articles.csv"
    transactions_file: str = "transactions_train.csv"
    customers_file: str = "customers.csv"
    date_column: str = "t_dat"
    price_column: str = "price"
    article_id_column: str = "article_id"
    customer_id_column: str = "customer_id"
    key_attributes: list[str] = Field(default=[
        "product_type_name",
        "colour_group_name",
        "section_name",
        "garment_group_name",
        "department_name",
        "index_group_name",
    ])


#Reviews Data Config 

class ReviewsDataConfig(BaseModel):
    file: str = "Womens Clothing E-Commerce Reviews.csv"
    text_column: str = "Review Text"
    rating_column: str = "Rating"
    recommend_column: str = "Recommended IND"
    department_column: str = "Department Name"
    class_column: str = "Class Name"
    division_column: str = "Division Name"
    feedback_column: str = "Positive Feedback Count"
    min_review_length: int = 20


#Model Configs 

class SentimentModelConfig(BaseModel):
    """Fine-tuned RoBERTa sentiment model(LoRA)."""
    base_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-5
    num_epochs: int = 5
    batch_size: int = 16
    max_length: int = 128
    test_size: float = 0.2
    random_seed: int = 42


class ZeroShotConfig(BaseModel):
    """BART zero-shot style classification."""
    model_name: str = "facebook/bart-large-mnli"
    style_labels: list[str] = Field(default=[
        "minimalist",
        "streetwear",
        "bohemian",
        "preppy",
        "athleisure",
        "quiet luxury",
        "Y2K revival",
        "oversized comfort",
        "bold prints",
        "sustainable basics",
    ])
    batch_size: int = 64
    confidence_threshold: float = 0.3


class EmbeddingConfig(BaseModel):
    """Sentence-transformer embeddings + clustering."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    umap_n_neighbors: int = 15
    umap_n_components: int = 2
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    random_seed: int = 42


#Google Trends Config 

class GoogleTrendsConfig(BaseModel):
    timeframe: str = "today 12-m"
    geo: str = "US"
    fashion_keywords: list[str] = Field(default=[
        "oversized blazer", "wide leg pants", "cargo pants",
        "linen shirt", "pastel dress", "maxi skirt",
        "cropped cardigan", "platform shoes", "sheer top",
        "leather jacket", "knit vest", "midi dress",
        "jogger pants", "puff sleeve", "slip dress",
        "denim jacket", "floral print", "monochrome outfit",
        "layered necklace", "bucket hat", "corset top",
        "pleated skirt", "oversized hoodie", "mesh top",
        "color blocking", "crochet top", "biker shorts",
    ])
    keyword_category_map: dict[str, str] = Field(default={
        "oversized blazer": "Jackets",
        "wide leg pants": "Trousers",
        "cargo pants": "Trousers",
        "linen shirt": "Tops",
        "pastel dress": "Dresses",
        "maxi skirt": "Skirts",
        "cropped cardigan": "Knitwear",
        "platform shoes": "Shoes",
        "sheer top": "Tops",
        "leather jacket": "Jackets",
        "knit vest": "Knitwear",
        "midi dress": "Dresses",
        "jogger pants": "Trousers",
        "puff sleeve": "Tops",
        "slip dress": "Dresses",
        "denim jacket": "Jackets",
        "floral print": "Prints & Patterns",
        "monochrome outfit": "Prints & Patterns",
        "layered necklace": "Accessories",
        "bucket hat": "Accessories",
        "corset top": "Tops",
        "pleated skirt": "Skirts",
        "oversized hoodie": "Tops",
        "mesh top": "Tops",
        "color blocking": "Prints & Patterns",
        "crochet top": "Tops",
        "biker shorts": "Shorts",
    })


#Trend Scoring Config 

class TrendScoringConfig(BaseModel):
    """Weights for the composite trend intelligence score."""
    w_sell_through: float = 0.25
    w_sentiment: float = 0.25
    w_trend_momentum: float = 0.20
    w_style_buzz: float = 0.15
    w_cluster_strength: float = 0.15

    quadrant_threshold: float = 0.5

    class Config:
        frozen = True


#Taxonomy Mapping 

CATEGORY_TAXONOMY = {
    "Tops": [
        "T-shirt", "Vest top", "Top", "Blouse", "Sweater",
        "Hoodie", "Polo shirt", "Tank Top",
    ],
    "Dresses": ["Dress"],
    "Trousers": [
        "Trousers", "Leggings/Tights", "Shorts", "Joggers",
    ],
    "Knitwear": ["Cardigan", "Knitted vest"],
    "Jackets": [
        "Jacket", "Blazer", "Coat", "Outdoor Jacket",
    ],
    "Skirts": ["Skirt"],
    "Shoes": ["Shoes", "Boots", "Sandals", "Sneakers"],
    "Accessories": [
        "Bag", "Hat/beret", "Scarf", "Belt", "Sunglasses",
        "Earring", "Necklace",
    ],
    "Shorts": ["Shorts"],
    "Prints & Patterns": [],
}

REVIEW_CLASS_TO_CATEGORY = {
    "Blouses": "Tops",
    "Knits": "Knitwear",
    "Sweaters": "Knitwear",
    "Pants": "Trousers",
    "Jeans": "Trousers",
    "Dresses": "Dresses",
    "Skirts": "Skirts",
    "Shorts": "Shorts",
    "Jackets": "Jackets",
    "Outerwear": "Jackets",
    "Fine gauge": "Knitwear",
    "Lounge": "Tops",
    "Swim": "Tops",
    "Casual bottoms": "Trousers",
    "Chemises": "Tops",
    "Layering": "Tops",
    "Legwear": "Trousers",
    "Sleep": "Tops",
    "Intimates": "Tops",
    "Trend": "Tops",
}


#Instantiate default configs 

hm_config = HMDataConfig()
reviews_config = ReviewsDataConfig()
sentiment_model_config = SentimentModelConfig()
zero_shot_config = ZeroShotConfig()
embedding_config = EmbeddingConfig()
google_trends_config = GoogleTrendsConfig()
trend_scoring_config = TrendScoringConfig()

RANDOM_SEED = 42
