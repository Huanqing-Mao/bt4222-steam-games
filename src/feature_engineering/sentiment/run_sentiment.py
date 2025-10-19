# run.py
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sentiment_analyser import SentimentAnalyser  # your class file

# ---------------- Worker Function ----------------
def worker(df_chunk, worker_id):
    """
    Each process initializes its own SentimentAnalyser and computes scores
    for its chunk of the dataframe with progress bar.
    """
    analyser = SentimentAnalyser()
    df_chunk = df_chunk.copy()

    # local tqdm bar for this worker
    tqdm.pandas(desc=f"Worker {worker_id+1}")

    df_chunk["sentiment_score"] = df_chunk["review"].progress_apply(analyser.compute_combined_score)
    return df_chunk


# ---------------- Parallel Runner ----------------
def compute_scores_parallel(df, num_workers=4):
    """
    Split the dataset into chunks, process each chunk in a separate process,
    and recombine results preserving original order.
    """
    df = df.dropna()
    df = df.reset_index(drop=True)
    chunks = np.array_split(df, num_workers)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, chunk, i) for i, chunk in enumerate(chunks)]

        # global progress bar across chunks
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing all chunks"):
            results.append(f.result())

    # combine results in original order
    combined = pd.concat(results).sort_index()
    return combined


# ---------------- Main Entry ----------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Compute sentiment scores in parallel.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output CSV")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes")
    args = parser.parse_args()

    print(f"📂 Loading dataset from {args.input} ...")
    df = pd.read_csv(args.input)

    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column")

    print(f"🚀 Computing sentiment scores using {args.workers} workers...")
    result_df = compute_scores_parallel(df, num_workers=args.workers)

    print(f"💾 Saving results to {args.output} ...")
    result_df.to_csv(args.output, index=False)
    print("✅ Done! Sentiment scores computed and saved.")
