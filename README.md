# BT4222 Project Source Code

A collection of notebooks, scripts and datasets for feature engineering and recommendation experiments on Steam game data (course project).
> - Notebooks were developed with recent Python (3.10/3.11). Adjust your environment if needed.
> - Datasets are not uploaded to GitHub directly due to large size, raw csv files can be accessed here: [BT4222 Group 9 Datasets](https://drive.google.com/drive/u/2/folders/1Hb68lcNCnkOnZKOcPtUTXIt3K4XvaMcb)
> - To reproduce the results, download all CSVs into a folder named `datasets` under the root directory (see file structure below).


## Repository layout
```text
.
├── README.md
└── src
    ├── datasets
    │   ├── 2_games_prices_merged.csv
    │   ├── 2_price_features.csv
    │   ├── 3_purchase_features.csv
    │   ├── english_reviews_1k.csv
    │   ├── english_reviews.csv
    │   ├── games.csv
    │   ├── games_encoded.csv
    │   ├── players.csv
    │   ├── prices.csv
    │   ├── purchased_games.csv
    │   ├── recommendations_for_all_players.csv
    │   ├── reviews.csv
    │   ├── reviews_lang_detect.csv
    │   ├── sentiment_1k.csv
    │   └── sentiment_reviews_18oct.csv
    ├── eda
    │   └── Dataset_Statistics.ipynb
    ├── feature-engineering
    │   ├── FeatureEngineering.ipynb
    │   ├── ReviewSampling.ipynb
    │   └── sentiment
    │       ├── run_sentiment.py
    │       ├── sentiment_analyser.py
    │       └── Sample1kTesting.ipynb
    └── model
        ├── CollaborativeBasedFiltering.ipynb
        └── ContentBasedFiltering_Final.ipynb
```

## Environment Set-up

1. Create and activate a Python virtual environment (macOS):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install typical dependencies used in the notebooks:
```bash
pip install pandas numpy scikit-learn tqdm vaderSentiment googletrans==4.0.0-rc1 httpx langdetect emoji
```

## Reproduce Results

- Open and run notebooks in `src/feature-engineering` to build and inspect features.
- Run sentiment processing script:
```bash
# you may customise # workers (total 3 hrs est. for 4 workers)
python src/feature-engineering/sentiment/run_sentiment.py --input english_reviews.csv --output custome_file_name --workers 4
```
- Explore modeling notebooks in `src/model` for content-based and collaborative filtering experiments.