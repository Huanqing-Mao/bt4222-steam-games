# BT4222 Project Source Code

End-to-end workflow for BT4222 Group 9’s Steam games recommender: data wrangling, sentiment analysis, feature engineering, and recommender models. 

> Sentiment Analysis was built and tested with Python 3.10/3.11 on macOS due to the long processing time, which exceeds the maximum inactivity window on Google Colab.

## Table of Contents
[Dataset Access](#dataset-access)  
[Repository Layout](#repository-layout)  
[Getting Started](#getting-started)  
[Running the Workflow](#running-the-workflow)  
[Key Notebooks & Scripts](#key-notebooks--scripts)  
[Reproducing Results](#reproducing-results)  
[Project Notes](#project-notes)  

## Dataset Access

Raw CSVs are hosted outside the repository because of size limits. Download them from the shared drive and place them under `src/datasets`:

- [BT4222 Group 9 Datasets (Google Drive)](https://drive.google.com/drive/u/2/folders/1Hb68lcNCnkOnZKOcPtUTXIt3K4XvaMcb)
- Keep filenames unchanged; the notebooks expect the naming convention shown in the repository layout.

## Repository Layout
```text
.
├── README.md
└── src
    ├── datasets/                # Raw + intermediate CSVs (download separately)
    ├── eda/
    │   └── Dataset_Statistics.ipynb
    ├── feature-engineering/
    │   ├── FeatureEngineering.ipynb
    │   ├── ReviewSampling.ipynb
    │   └── sentiment/
    │       ├── run_sentiment.py
    │       ├── sentiment_analyser.py
    │       └── Sample1kTesting.ipynb
    └── model/
        ├── CollaborativeBasedFiltering.ipynb
        └── ContentBasedFiltering_Final.ipynb
```

## Environment Set-up

## Getting Started

1. **Set up a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install core dependencies**  
   ```bash
   pip install pandas numpy scikit-learn tqdm vaderSentiment googletrans==4.0.0-rc1 httpx langdetect emoji yfinance
   ```

3. **Download datasets** into `src/datasets` using the link above. Double-check paths inside the notebooks if you organise datasets differently.

## Running the Workflow

- **Exploratory analysis**: Start with `src/eda/Dataset_Statistics.ipynb` to understand the raw data distributions and missingness.
- **Feature engineering**:
  - Use `src/feature-engineering/ReviewSampling.ipynb` to downsample reviews for manageable experiments.
  - Run `src/feature-engineering/FeatureEngineering.ipynb` to build player-level and game-level features, including price, engagement, and textual indicators.
- **Sentiment enrichment**:
  - Execute the sentiment script for large-scale processing:
    ```bash
    python src/feature-engineering/sentiment/run_sentiment.py \
      --input english_reviews.csv \
      --output sentiment_reviews_full.csv \
      --workers 4
    ```
    Adjust `--workers` to match your CPU (3–4 workers ≈ 3 hours on ~30k reviews). For quick tests, use `english_reviews_1k.csv`.
- **Modelling**:
  - `src/model/ContentBasedFiltering_Final.ipynb` builds similarity-based recommenders using engineered features.
  - `src/model/CollaborativeBasedFiltering.ipynb` experiments with matrix factorisation and neighbourhood-based approaches.

## Key Notebooks & Scripts

- **Dataset_Statistics.ipynb** – descriptive stats, missing value checks, and sanity checks for the raw tables.
- **FeatureEngineering.ipynb** – merges purchases, prices, and reviews into analytical tables, exporting `2_price_features.csv`, `3_purchase_features.csv`, etc.
- **ReviewSampling.ipynb** – sampling workflows for balanced sentiment analysis experiments, which includes language detection and removal of non-english reviews.
- **run_sentiment.py** – multi-processing of sentiment analysis; wraps `sentiment_analyser.py` (mandatory to run locally).
- **Sample1kTesting.ipynb** – tests for the sentiment pipeline on the 1k review subset.
- **ContentBasedFiltering_Final.ipynb / CollaborativeBasedFiltering.ipynb** – evaluate recommenders, calibrate hyperparameters, and export recommendation lists (e.g., `recommendations_for_all_players.csv`).

## Reproducing Results

1. Download the full dataset bundle and place it in `src/datasets`.
2. Run the sentiment script (or use the precomputed `sentiment_reviews_18oct.csv` if available).
3. Execute feature-engineering notebooks to regenerate intermediate CSVs.
4. Open modelling notebooks to rebuild final recommendation outputs. Results are cached to `src/datasets`.

## Project Notes

- The repository does not contain the datasets due to their large sizes. You can always download via the link provided above.