# MongoDB Schema Design -- MovieLens ML Pipeline

End-to-end ML-engineering style project:
**MovieLens CSV → MongoDB → Aggregation-based feature engineering → Export → ML baseline**


## 1. Project Overview

This project implements an end-to-end ML data pipeline:

CSV → MongoDB (NoSQL) → Aggregation-based Feature Engineering → ML Model

This project demonstrates:

- NoSQL reference schema design
- Aggregation pipelines as feature engineering
- Reproducible export for ML training


The schema is intentionally designed to:

-   Support scalable event ingestion
-   Enable efficient aggregation pipelines
-   Avoid duplication
-   Prevent data leakage
-   Align with ML feature engineering workflows

------------------------------------------------------------------------


## 3. Dataset
Download **MovieLens (ml-latest-small)** from GroupLens:
- https://grouplens.org/datasets/movielens/

Place files at:
```
`data/ml-latest-small/movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`
```

------------------------------------------------------------------------

## 4. Setup

### Create environment and run the scrpits as following
```bash
uv init mongodb-ml-pipeline
uv add pymongo pandas scikit-learn pyarrow
uv run data_ingestion.py
uv run feature_export.py
train and evaluate inside ml_model.ipynb
```
------------------------------------------------------------------------

## 5. Data Modeling Strategy

MongoDB allows two main modeling approaches:

-   Embedded documents
-   Referenced collections

For this project, a referenced schema design was chosen.

### Why Referenced?

-   Ratings are event logs (growing over time)
-   Embedding would create extremely large documents
-   Aggregations would be inefficient
-   Updates would require rewriting large documents
-   ML feature engineering requires flexible joins

This design mirrors real-world production systems.

------------------------------------------------------------------------

## 6. Collections

### movies (Static / Master Data)

Stores movie metadata.

Example:
```json
{
    "movieId" : 1,
    "title" : "Toy Story (1995)",
    "genres" : "Adventure | Animation | Children | Comedy | Fantasy
}
```
<!-- { "movieId": 1, "title": "Toy Story (1995)", "genres":
"Adventure\|Animation\|Children\|Comedy\|Fantasy" } -->

Characteristics: - Small dataset (\~9k rows) - Rarely changes - One
document per movieId - Unique index on movieId

------------------------------------------------------------------------

### ratings (Event / Log Data)

Stores user--movie interactions.

Example:

```json
{
    "_rid": "1_1_964982703",
    "userId": 1,
    "movieId": 1,
    "rating": 4.0,
    "timestamp": 964982703
}
```

Characteristics: - Append-only - Large dataset (\~100k+ rows) - Multiple
ratings per movie - Multiple ratings per user - Time-based interactions

Synthetic Key:

\_rid = userId_movieId_timestamp

Ensures idempotent ingestion and prevents duplicates.

------------------------------------------------------------------------

## 7. Indexing Strategy

movies: - movieId (unique) - genres

movies:  
- movieId (unique)
- genres

ratings: 
- \_rid (unique synthetic key for idempotent ingestion)
- (userId, timestamp DESC)
- (movieId, timestamp DESC)
- (userId, movieId)

tags:  
- \_tid (unique synthetic key for idempotent ingestion)
- (movieId, timestamp DESC)
- tag

links:  
- No explicit indexes are created by the ingestion script.
- Documents are upserted using movieId as the lookup key.

------------------------------------------------------------------------

## 8. Aggregation Support for ML

Schema supports:

-   \$group
-   \$lookup
-   \$unwind
-   \$match
-   \$project

Example movie feature aggregation:

$group: {  _id: "$movieId", avg_rating: {$avg: "$rating"}, rating_count:
{\$sum: 1} }

```json
{
    "$group": {
        "_id": "$movieId",
        "avg_rating": { "$avg": "$rating" },
        "rating_count": { "$sum": 1 }
    }
}
```

------------------------------------------------------------------------

## 9. Architecture Flow
```text
MovieLens CSV
        ↓
Pandas DataFrame
        ↓
MongoDB (movies, ratings, tags, links)
        ↓
Aggregation Pipelines
        ↓
user_features + movie_features
        ↓
Feature Export (CSV / Parquet)
        ↓
ML Training (RandomForest baseline)
```

------------------------------------------------------------------------

## 10. Explanation

I used a referenced data model separating static movie metadata from
dynamic rating events. Ratings are modeled as append-only logs with a
synthetic unique key to ensure idempotent ingestion. Indexes were
created based on aggregation patterns used for ML feature engineering.

