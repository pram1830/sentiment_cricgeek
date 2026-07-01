# CricGeek Writer-Aware Calibration Note

This project is calibrated for writer-aware scoring, not generic sentiment polarity.

The scoring engine combines writer DNA detection, paragraph-level component scoring, and paragraph-weighted toxicity penalties.

## Model and license note

- sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0)
- unitary/toxic-bert (license: check model card in local environment before commercial deployment)
- bhadresh-savani/distilbert-base-uncased-emotion (license: check model card in local environment before commercial deployment)
- cardiffnlp/twitter-roberta-base-sentiment-latest (license: check model card in local environment before commercial deployment)

If any model license is unclear for your deployment policy, remove or replace that model before release.
