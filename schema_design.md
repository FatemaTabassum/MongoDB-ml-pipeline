# MongoDB Schema Design (MovieLens)

This project models MovieLens as a **reference-based** schema to support analytics and ML feature engineering.

## Collections

### 1. movies collection
**Key:** `movieId` (unique)
Example:  
```json
{
  "movieId": 1,
  "title": "Toy Story (1995)",
  "genres": "Adventure|Animation|Children|Comedy|Fantasy"
}
```
### 2. ratings collection
***Composite Natural Key***
Ratings are uniquely identified by:  
userId + movieId + timestamp. 
To enforce idempotency, ingestion creates a synthetic field:  
***Synthetic key***
```
_rid = "userId_movieId_timestamp"
```
Example:  
```json
{
  "userId": 1,
  "movieId": 296,
  "rating": 4.0,
  "timestamp": 1147880044,
  "_rid": "1_296_1147880044"
}
```
### 3. tags collection
***Purpose***. 
Stores user-generated tags for movies.  

***Synthetic Unique Key***
```
_tid = userId_movieId_timestamp_tag
```
Example:    
```json
{
  "userId": 2,
  "movieId": 60756,
  "tag": "dark comedy",
  "timestamp": 1445714994,
  "_tid": "2_60756_1445714994_dark comedy"
}
```
### 4. Links collection
***Purpose***
Maps MovieLens movieId to external datasets.  
***primary key*** 
movieId (unique)
```json
{
  "movieId": 1,
  "imdbId": 114709,
  "tmdbId": 862
}
```
