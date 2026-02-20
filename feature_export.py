#!/usr/bin/env python3
"""
Compute ML features using MongoDB aggregations and export to disk.

Creates/updates:
- user_features collection
- movie_features collection
Exports:
- features/ratings_features.csv
- features/ratings_features.parquet (optional, if pyarrow installed)
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import pandas as pd
from pymongo import MongoClient


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pipeline_movie_features() -> List[Dict[str, Any]]:
    return [
        {
            "$group": {
                "_id": "$movieId",
                "movie_rating_count": {"$sum": 1},
                "movie_rating_avg": {"$avg": "$rating"},
                "movie_rating_std_approx": {"$stdDevPop": "$rating"},
                "movie_rating_min": {"$min": "$rating"},
                "movie_rating_max": {"$max": "$rating"},
                "movie_last_ts": {"$max": "$timestamp"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "movieId": "$_id",
                "movie_rating_count": 1,
                "movie_rating_avg": 1,
                "movie_rating_std_approx": 1,
                "movie_rating_min": 1,
                "movie_rating_max": 1,
                "movie_last_ts": 1,
            }
        },
    ]


def pipeline_user_features() -> List[Dict[str, Any]]:
    return [
        {
            "$group": {
                "_id": "$userId",
                "user_rating_count": {"$sum": 1},
                "user_rating_avg": {"$avg": "$rating"},
                "user_rating_std_approx": {"$stdDevPop": "$rating"},
                "user_rating_min": {"$min": "$rating"},
                "user_rating_max": {"$max": "$rating"},
                "user_last_ts": {"$max": "$timestamp"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "userId": "$_id",
                "user_rating_count": 1,
                "user_rating_avg": 1,
                "user_rating_std_approx": 1,
                "user_rating_min": 1,
                "user_rating_max": 1,
                "user_last_ts": 1,
            }
        },
    ]


def pipeline_labeled_examples(limit: int = 0) -> List[Dict[str, Any]]:
    p: List[Dict[str, Any]] = [
        {
            "$lookup": {
                "from": "user_features",
                "localField": "userId",
                "foreignField": "userId",
                "as": "u",
            }
        },
        {"$unwind": "$u"},
        {
            "$lookup": {
                "from": "movie_features",
                "localField": "movieId",
                "foreignField": "movieId",
                "as": "m",
            }
        },
        {"$unwind": "$m"},
        {
            "$lookup": {
                "from": "movies",
                "localField": "movieId",
                "foreignField": "movieId",
                "as": "mv",
            }
        },
        {"$unwind": "$mv"},
        {
            "$project": {
                "_id": 0,
                "userId": 1,
                "movieId": 1,
                "timestamp": 1,
                "rating": 1,
                "user_rating_count": "$u.user_rating_count",
                "user_rating_avg": "$u.user_rating_avg",
                "user_rating_std_approx": "$u.user_rating_std_approx",
                "movie_rating_count": "$m.movie_rating_count",
                "movie_rating_avg": "$m.movie_rating_avg",
                "movie_rating_std_approx": "$m.movie_rating_std_approx",
                "genres": "$mv.genres",
            }
        },
    ]
    if limit and limit > 0:
        p.append({"$limit": limit})
    return p


def upsert_collection(db, target_collection: str, docs: List[Dict[str, Any]], key: str) -> None:
    col = db[target_collection]
    col.create_index(key, unique=True)
    if not docs:
        return
    # Replace all docs (simple for small datasets)
    col.delete_many({})
    col.insert_many(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="movielens")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for exported examples (0 = all)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    print("Computing movie_features...")
    movie_feats = list(db["ratings"].aggregate(pipeline_movie_features(), allowDiskUse=True))
    upsert_collection(db, "movie_features", movie_feats, key="movieId")

    print("Computing user_features...")
    user_feats = list(db["ratings"].aggregate(pipeline_user_features(), allowDiskUse=True))
    upsert_collection(db, "user_features", user_feats, key="userId")

    print("Creating labeled examples dataset...")
    examples = list(db["ratings"].aggregate(pipeline_labeled_examples(limit=args.limit), allowDiskUse=True))
    df = pd.DataFrame(examples)

    # Basic cleanup for ML:
    # Convert genres string to simple counts as a baseline feature
    # (More advanced: multi-hot encode genres; we do it in notebook)
    df["genres_len"] = df["genres"].fillna("").apply(lambda s: len(str(s).split("|")) if s else 0)

    csv_path = os.path.join(args.out_dir, "ratings_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}  rows={len(df)} cols={len(df.columns)}")

    # Parquet optional (good for ML engineering)
    try:
        pq_path = os.path.join(args.out_dir, "ratings_features.parquet")
        df.to_parquet(pq_path, index=False)
        print(f"Wrote: {pq_path}")
    except Exception as e:
        print("Parquet export skipped (install pyarrow). Reason:", e)


if __name__ == "__main__":
    main()
