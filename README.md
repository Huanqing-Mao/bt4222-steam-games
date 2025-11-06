# BT4222 Project Source Code

A collection of notebooks, scripts and datasets for feature engineering and recommendation experiments on Steam game data (course project).

## Repository layout

```bash
.
├── README.md
└── src
    ├── datasets # update later
    │   ├── english_reviews_1k.csv
    │   ├── english_reviews.csv
    │   ├── games.csv
    │   ├── players.csv
    │   ├── prices.csv
    │   ├── purchased_games.csv
    │   ├── reviews.csv
    │   ├── reviews_lang_detect.csv
    │   ├── sentiment_1k.csv
    │   └── sentiment_reviews_18oct.csv
    ├── feature-engineering
    │   ├── FeatureEngineering.ipynb # Addition of other features
    │   ├── ReviewSampling.ipynb
    │   └── sentiment # Sentiment Analysis 
    │       ├── run_sentiment.py
    │       ├── sentiment_analyser.py
    │       └── Sample1kTesting.ipynb
    └── model
        ├── CollaborativeBasedFiltering.ipynb
        └── ContentBasedFiltering_Final.ipynb


## Quick start

1. Create and activate a Python virtual environment (macOS):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install typical dependencies used in the notebooks:
```bash
pip install pandas numpy scikit-learn tqdm vaderSentiment googletrans==4.0.0-rc1 httpx langdetect emoji
```

3. Reproduce steps
- Open and run notebooks in `src/feature-engineering` to build and inspect features.
- Run sentiment processing script:
```bash
python src/feature-engineering/sentiment/run_sentiment.py --input english_reviews.csv --output custome_file_name --workers 4
```
- Explore modeling notebooks in `src/model` for content-based and collaborative filtering experiments.

## Notes & recommendations

- Notebooks were developed with recent Python (3.10/3.11). Adjust your environment if needed.
- Consider adding a pinned `requirements.txt` or `environment.yml` to ensure reproducibility.
- Datasets included in `src/datasets` are used by the notebooks — keep paths consistent when running code.
