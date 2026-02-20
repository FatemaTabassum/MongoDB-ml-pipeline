#!/usr/bin/env python3
"""
Load MovieLens (ml-latest-small) CSV files into MongoDB.
Creates collections:
- movies
- ratings
- tags
- links

Also creates useful indexes for ML / aggregation workloads.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List

import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _chunked(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    n = len(df)
    for i in range(0, n, chunk_size):
        yield df.iloc[i: i + chunk_size]


def _upsert_many(col, docs: List[Dict], key_field: str, chunk_size: int = 2000) -> None:
    """Upsert docs by key_field using bulk writes (efficient + idempotent)."""
    ops: List[UpdateOne] = []
    for d in docs:
        if key_field not in d:
            raise ValueError(f"Document missing key_field={key_field}: {d}")
        ops.append(UpdateOne({key_field: d[key_field]}, {"$set": d}, upsert=True))

    if not ops:
        return

    for i in range(0, len(ops), chunk_size):
        batch = ops[i: i + chunk_size]
        try:
            col.bulk_write(batch, ordered=False)
        except BulkWriteError as e:
            print("BulkWriteError:", e.details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ml-latest-small", help="Path to MovieLens folder")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017", help="MongoDB connection URI")
    parser.add_argument("--db", default="movielens", help="Database name")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for inserts")
    args = parser.parse_args()

    movies_path = os.path.join(args.data_dir, "movies.csv")
    ratings_path = os.path.join(args.data_dir, "ratings.csv")
    tags_path = os.path.join(args.data_dir, "tags.csv")
    links_path = os.path.join(args.data_dir, "links.csv")

    print("Reading CSVs...")
    movies_df = _read_csv(movies_path)
    ratings_df = _read_csv(ratings_path)
    tags_df = _read_csv(tags_path) if os.path.exists(tags_path) else pd.DataFrame()
    links_df = _read_csv(links_path) if os.path.exists(links_path) else pd.DataFrame()

    # Basic cleaning / typing
    # movies_df["movieId"] = preprocess(movies_df["movieId"])
    print(movies_df["movieId"][0].dtype)
    movies_df["movieId"] = movies_df["movieId"].astype(int)
    print(movies_df["movieId"][0].dtype)

    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(int)
    if "timestamp" in ratings_df.columns:
        ratings_df["timestamp"] = ratings_df["timestamp"].astype(int)

    if not tags_df.empty:
        tags_df["userId"] = tags_df["userId"].astype(int)
        tags_df["movieId"] = tags_df["movieId"].astype(int)
        if "timestamp" in tags_df.columns:
            tags_df["timestamp"] = tags_df["timestamp"].astype(int)

    if not links_df.empty:
        links_df["movieId"] = links_df["movieId"].astype(int)

    print("Connecting to MongoDB...")
    client = MongoClient(args.mongo_uri)
    print("Available DBs:", client.list_database_names())

    db = client[args.db]
    print("Connected to database:", args.db)
    print("Existing collections:", db.list_collection_names())

    # Collections (MongoDB handles)
    c_movies = db["movies"]
    c_ratings = db["ratings"]
    c_tags = db["tags"]
    c_links = db["links"]

    print("Collections handles:", c_movies.full_name, c_ratings.full_name, c_tags.full_name, c_links.full_name)

    # ---- Load movies ----
    print("Upserting movies...")
    movie_docs = movies_df.to_dict(orient="records")
    _upsert_many(c_movies, movie_docs, key_field="movieId", chunk_size=args.chunk_size)

    # ---- Load ratings ----
    print("Inserting ratings (append-only; safe to re-run)...")
    ratings_df = ratings_df.copy()
    ratings_df["_rid"] = (
        ratings_df["userId"].astype(str)
        + "_"
        + ratings_df["movieId"].astype(str)
        + "_"
        + ratings_df["timestamp"].astype(str)
    )

    c_ratings.create_index("_rid", unique=True)

    for part in _chunked(ratings_df, args.chunk_size):
        if part.empty:
            continue
        docs = part.to_dict(orient="records")
        ops = [UpdateOne({"_rid": d["_rid"]}, {"$setOnInsert": d}, upsert=True) for d in docs]
        try:
            c_ratings.bulk_write(ops, ordered=False)
        except BulkWriteError:
            # duplicates are okay if rerun
            pass

    # ---- Load tags ----
    if not tags_df.empty:
        print("Inserting tags...")
        tags_df = tags_df.copy()
        tags_df["_tid"] = (
            tags_df["userId"].astype(str)
            + "_"
            + tags_df["movieId"].astype(str)
            + "_"
            + tags_df["timestamp"].astype(str)
            + "_"
            + tags_df["tag"].astype(str)
        )
        c_tags.create_index("_tid", unique=True)
        for part in _chunked(tags_df, args.chunk_size):
            if part.empty:
                continue
            docs = part.to_dict(orient="records")
            ops = [UpdateOne({"_tid": d["_tid"]}, {"$setOnInsert": d}, upsert=True) for d in docs]
            try:
                c_tags.bulk_write(ops, ordered=False)
            except BulkWriteError:
                pass

    # ---- Load links ----
    if not links_df.empty:
        print("Upserting links...")
        link_docs = links_df.to_dict(orient="records")
        _upsert_many(c_links, link_docs, key_field="movieId", chunk_size=args.chunk_size)

    print("Creating indexes...")
    c_movies.create_index("movieId", unique=True)
    c_movies.create_index("genres")
    c_ratings.create_index([("userId", 1), ("timestamp", -1)])
    c_ratings.create_index([("movieId", 1), ("timestamp", -1)])
    c_ratings.create_index([("userId", 1), ("movieId", 1)])
    if not tags_df.empty:
        c_tags.create_index([("movieId", 1), ("timestamp", -1)])
        c_tags.create_index([("tag", 1)])

    print(f"Done. Database='{args.db}' populated.")
    print("Collections now:", db.list_collection_names())
    print("Counts:",
          "movies=", c_movies.count_documents({}),
          "ratings=", c_ratings.count_documents({}),
          "tags=", c_tags.count_documents({}) if "tags" in db.list_collection_names() else 0,
          "links=", c_links.count_documents({}) if "links" in db.list_collection_names() else 0
          )


if __name__ == "__main__":
    main()
