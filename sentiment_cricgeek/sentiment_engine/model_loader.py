from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


@dataclass
class LoadedModels:
    """Container for reusable local models."""

    embedder: SentenceTransformer
    toxicity_classifier: Any
    emotion_classifier: Any
    sentiment_classifier: Any
    device: str


class ModelLoader:
    """Loads and caches local sentiment signals for calibrated scoring."""

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOXICITY_MODEL = "unitary/toxic-bert"
    EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    _cached_models: Optional[LoadedModels] = None

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline_device = 0 if self.device == "cuda" else -1
        print("Using device:", self.device)

    def _log_model_device(self, model: Any, name: str) -> None:
        try:
            print("Model running on:", next(model.parameters()).device, f"({name})")
        except Exception:
            print("Model running on: unknown", f"({name})")

    def load(self) -> LoadedModels:
        if ModelLoader._cached_models is not None:
            return ModelLoader._cached_models

        embedder = SentenceTransformer(self.EMBEDDING_MODEL, device=self.device)
        embedder = embedder.to(self.device)
        try:
            st_backbone = embedder._first_module().auto_model
            self._log_model_device(st_backbone, "sentence-transformer")
        except Exception:
            print("Model running on: unknown (sentence-transformer)")

        toxicity_classifier = pipeline(
            "text-classification",
            model=self.TOXICITY_MODEL,
            tokenizer=self.TOXICITY_MODEL,
            device=self.pipeline_device,
            framework="pt",
            return_all_scores=True,
            truncation=True,
        )
        if hasattr(toxicity_classifier, "model"):
            toxicity_classifier.model.to(torch.device(self.device))
            self._log_model_device(toxicity_classifier.model, "toxicity")

        emotion_classifier = pipeline(
            "text-classification",
            model=self.EMOTION_MODEL,
            tokenizer=self.EMOTION_MODEL,
            device=self.pipeline_device,
            framework="pt",
            return_all_scores=True,
            truncation=True,
        )
        if hasattr(emotion_classifier, "model"):
            emotion_classifier.model.to(torch.device(self.device))
            self._log_model_device(emotion_classifier.model, "emotion")

        sentiment_tokenizer = AutoTokenizer.from_pretrained(self.SENTIMENT_MODEL)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.SENTIMENT_MODEL,
            use_safetensors=True,
        ).to(self.device)
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            tokenizer=sentiment_tokenizer,
            device=self.pipeline_device,
            framework="pt",
            return_all_scores=True,
            truncation=True,
        )
        if hasattr(sentiment_classifier, "model"):
            sentiment_classifier.model.to(torch.device(self.device))
            self._log_model_device(sentiment_classifier.model, "sentiment")

        ModelLoader._cached_models = LoadedModels(
            embedder=embedder,
            toxicity_classifier=toxicity_classifier,
            emotion_classifier=emotion_classifier,
            sentiment_classifier=sentiment_classifier,
            device=self.device,
        )
        return ModelLoader._cached_models
