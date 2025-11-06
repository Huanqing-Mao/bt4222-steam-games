# bt4222-steam-games

A collection of notebooks, scripts and datasets for feature engineering and recommendation experiments on Steam game data (course project).

## Repository layout

```bash
- src/
  - datasets/
    - games.csv, reviews.csv, english_reviews.csv, english_reviews_1k.csv, sentiment_1k.csv, sentiment_reviews_18oct.csv, prices.csv, purchased_games.csv, players.csv, reviews_lang_detect.csv
  - feature-engineering/
    - FeatureEngineering.ipynb
    - ReviewSampling.ipynb
    - sentiment/
      - run_sentiment.py
      - sentiment_analyser.py
      - Sample1kTesting.ipynb
  - model/
    - ContentBasedFiltering_Final.ipynb
    - CollaborativeBasedFiltering.ipynb


## Quick start

1. Create and activate a Python virtual environment (macOS):
```sh
python -m venv .venv
source .venv/bin/activate
```

2. Install typical dependencies used in the notebooks:
```sh
pip install pandas numpy scikit-learn tqdm vaderSentiment googletrans==4.0.0-rc1 httpx langdetect emoji
```

3. Reproduce steps
- Open and run notebooks in `src/feature-engineering` to build and inspect features.
- Run sentiment processing script:
```sh
python src/feature-engineering/sentiment/run_sentiment.py
```
- Explore modeling notebooks in `src/model` for content-based and collaborative filtering experiments.

## Notes & recommendations

- Notebooks were developed with recent Python (3.10/3.11). Adjust your environment if needed.
- Consider adding a pinned `requirements.txt` or `environment.yml` to ensure reproducibility.
- Datasets included in `src/datasets` are used by the notebooks — keep paths consistent when running code.
