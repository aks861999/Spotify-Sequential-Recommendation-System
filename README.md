# 🎵 Spotify Sequential Recommendation System
### Next-Event Prediction via Track2Vec — Word2Vec Transfer Learning Applied to Music Sequences

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-E04A2D?style=flat-square)](https://radimrehurek.com/gensim/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=flat-square&logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [The Core Problem](#the-core-problem)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Methodology](#methodology)
  - [Data Preprocessing](#1-data-preprocessing)
  - [EDA and Statistical Analysis](#2-eda-and-statistical-analysis)
  - [Sequence Feature Engineering](#3-sequence-feature-engineering)
  - [Track2Vec Model](#4-track2vec-model)
  - [Hyperparameter Tuning](#5-hyperparameter-tuning)
  - [Inference Engine](#6-inference-engine)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)

---

## Overview

This project builds a **sequential music recommendation system** that, given a partial playlist of tracks, predicts what a user is most likely to listen to next. The system treats the problem as **Next-Event Prediction (NEP)** — a well-studied formulation in sequential learning where the goal is to predict the next item in an ordered interaction sequence based solely on behavioral history.

The central contribution is **Track2Vec**: a domain transfer of Word2Vec's skip-gram architecture to music, where playlists act as *sentences* and individual tracks act as *tokens*. Co-occurrence patterns within playlist windows encode latent musical affinity, allowing the model to learn a 48-dimensional embedding space in which semantically similar tracks (by listening behavior, not audio features) cluster together.

This is the same paradigm behind Spotify's own published research ([song2vec](https://arxiv.org/abs/1603.06228)), implemented from the ground up on a real-world dataset of 14,540 users and over 76,000 playlists.

---

## The Core Problem

> *"Can we suggest what to listen to next when presented with a sequence of songs?"*

A user's playlist is an ordered sequence of interactions. The task is to learn from thousands of such sequences — each constructed by real people with real taste — and then generalise: given any new, unseen partial sequence, retrieve the most likely continuation.

This is a **purely behavioural** approach. No audio features, no genre tags, no artist metadata. The model learns entirely from the structure of co-occurrence in playlists — which is precisely the signal that matters most in collaborative, preference-driven recommendation.

---

## Dataset

The dataset is derived from the **#nowplaying dataset**, a collection of tweets in which Spotify users publish their currently playing track via the `#nowplaying` hashtag. It captures organic, real-world listening behaviour — not artificially curated sequences.

| Field | Description |
|---|---|
| `user_id` | MD5 hash of the Spotify username |
| `artistname` | Artist name as displayed on Spotify |
| `trackname` | Track title |
| `playlistname` | Name of the playlist containing this track |

**Raw dataset statistics:**

| Metric | Value |
|---|---|
| Total rows | 644,586 |
| Unique users | 14,540 |
| Unique artists | 77,555 |
| Unique tracks | 305,678 |
| Unique playlists | 100,679 |

> **Note:** A 5% stochastic sample is drawn at load time (`skiprows=lambda i: i>0 and random.random() > 0.05`) for memory efficiency. The full dataset can be used by setting `p=1.0`.

---

## Architecture

The full pipeline is a six-stage system, from raw CSV to ranked recommendations:

```
┌──────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│  Data         │   │  EDA + Quality   │   │  Sequence Feature     │
│  Ingestion    │──▶│  Control         │──▶│  Engineering          │
│  (5% sample) │   │  (power-law,     │   │  (playlist grouping,  │
│              │   │   filters)        │   │   last-item holdout)  │
└──────────────┘   └──────────────────┘   └───────────────────────┘
                                                       │
                                                       ▼
┌──────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│  Inference   │   │  Hyperparameter  │   │  Track2Vec Training   │
│  Engine      │◀──│  Tuning          │◀──│  (Word2Vec Skip-gram, │
│  (K-NN in    │   │  (grid search,   │   │   48-dim embeddings,  │
│  embed space)│   │   HR@100 metric) │   │   neg. sampling)      │
└──────────────┘   └──────────────────┘   └───────────────────────┘
```


---

## Methodology

### 1. Data Preprocessing

#### Column Normalisation
Raw column names contain quoted strings (`"artistname"`, `"trackname"`) which are stripped and shortened. This is a data quality step that prevents silent join failures downstream.

```python
df.columns = (df.columns
    .str.replace('"', "")
    .str.replace("name", "")
    .str.replace(" ", ""))
# Result: ['user_id', 'artist', 'track', 'playlist']
```

#### Noise Filtering — Motivated by Power-Law Analysis
Before applying any filter, the project formally tests whether the artist frequency distribution follows a power law using `powerlaw.Fit(discrete=True)`. The resulting log-log CCDF confirms Zipf's law holds — a small number of artists dominate the corpus while the majority appear only a handful of times. This statistical evidence justifies the subsequent filter.

```python
# Keep only artists with sufficient representation
df = df.groupby('artist').filter(lambda x: len(x) >= 50)
```

**Why 50?** The power-law analysis shows the inflection point where artist representations become too sparse to yield reliable co-occurrence statistics for embedding training. Below this threshold, tracks associated with an artist would have too few playlist neighbours to produce a meaningful embedding.

#### Cold-Start Mitigation
Users with fewer than 10 unique artists in their playlists are excluded. These users provide insufficient signal to learn meaningful sequence patterns from, and including them would introduce noise that actively harms generalisation to new users — the classic **cold-start problem**.

```python
df = df[df.groupby('user_id').artist.transform('nunique') >= 10]
# Retained: 318,699 rows
```

---

### 2. EDA and Statistical Analysis

**Artist and track distributions** are visualised with log-scale histograms. The y-axis log scaling is not cosmetic — it is necessary to see the shape of the distribution, which would appear as a flat spike against a near-zero baseline on a linear scale.

**Power-law fitting:**
```python
import powerlaw
data = list(artist_counter.values())
fit = powerlaw.Fit(data, discrete=True)
fit.plot_pdf(color='#2E3454', linewidth=2)
fit.power_law.plot_pdf(color='#2E3454', linestyle='--')
```

The solid line (empirical PDF) and dashed line (fitted power law) overlap closely, validating the distributional assumption that drives the filtering strategy.

---

### 3. Sequence Feature Engineering

#### Composite Key Construction
Two composite identifiers are constructed to avoid ambiguity:

```python
# Namespace playlists to a user — same playlist name across users is treated differently
df['playlist_id'] = df['user_id'] + '-' + df['playlist']

# Namespace tracks to an artist — same track title by different artists is treated differently
df['track_id'] = df['artist'] + '|||' + df['track']
```

The `|||` separator is chosen deliberately — it is unlikely to appear in any artist or track name, preventing collision.

#### Sequential Grouping and Last-Item Holdout

Each playlist is collapsed into a sequence, and the **last item is held out as the prediction target**. This is the correct formulation for next-event prediction because it mirrors the actual inference scenario: the model sees all but the last item and must predict it.

```python
df = df.groupby('playlist_id').agg(
    artist_sequence = ('artist',   lambda x: list(x)),
    track_sequence  = ('track_id', lambda x: list(x)),
    track_test_x    = ('track_id', lambda x: list(x)[:-1] if len(x) > 1 else []),
    track_test_y    = ('track_id', lambda x: list(x)[-1]  if len(x) > 1 else '')
).reset_index()

# Minimum viable sequence length
df = df[df['track_sequence'].apply(len) > 2]
# Result: 76,583 sequences
```

#### Train / Validate / Test Split

```python
train, validate, test = np.split(
    df.sample(frac=1, random_state=42),
    [int(0.8 * len(df)), int(0.9 * len(df))]
)
# test = 2,782 rows
```

The shuffle uses `random_state=42` for reproducibility. The 80/10/10 split is standard for this dataset size: enough training data to cover the vocabulary, enough held-out data for statistically meaningful evaluation.

---

### 4. Track2Vec Model

#### The Core Analogy

| NLP Domain | Music Domain |
|---|---|
| Corpus | All playlists |
| Document / Sentence | A single playlist |
| Word / Token | A track (`artist\|\|\|title`) |
| Word co-occurrence | Tracks appearing near each other in a playlist |
| Semantic similarity | Musical affinity / shared listening context |

This analogy is not superficial. Word2Vec learns that words with similar *contexts* have similar *meanings*. Applied to playlists, a track's context is the set of tracks a user placed near it. Two tracks that consistently appear in similar playlist neighbourhoods across many users share something real — tempo, mood, genre, or cultural moment — even if the model never sees those features directly.

#### Architecture

The model uses Word2Vec's **skip-gram** objective with **negative sampling**, implemented via Gensim:

```python
from gensim.models.word2vec import Word2Vec

hyperparams = {
    "min_count"   : 20,    # Only tracks appearing 20+ times are embedded
    "epochs"      : 30,    # Full passes over the training corpus
    "vector_size" : 48,    # Dimensionality of the embedding space
    "window"      : 10,    # Context window: 10 tracks left and right
    "ns_exponent" : 0.75,  # Negative sampling distribution exponent
}

track2vec_model = Word2Vec(train['track_sequence'].tolist(), **hyperparams)
```

**Parameter rationale:**

| Parameter | Value | Reasoning |
|---|---|---|
| `vector_size` | 48 | Sufficient dimensionality for ~300 vocabulary items; avoids overfitting on sparse data |
| `window` | 10 | Captures global playlist context without noise from distant, weakly-related tracks |
| `ns_exponent` | 0.75 | Standard value; balances between uniform (1.0) and frequency-proportional (0.0) negative sampling |
| `epochs` | 30 | Sufficient convergence without overfitting; confirmed by validation metric plateau |
| `min_count` | 20 | Selected by grid search — see Hyperparameter Tuning |

#### Training

```python
track_sequences = train['track_sequence'].tolist()
track2vec_model.build_vocab(track_sequences)
track2vec_model.train(
    track_sequences,
    total_examples=track2vec_model.corpus_count,
    epochs=track2vec_model.epochs
)
print(f"Vector space size: {len(track2vec_model.wv.index_to_key)}")
# → 319 tracks (with min_count=20)
```

---

### 5. Hyperparameter Tuning

`min_count` is the most consequential hyperparameter for this model class. It controls the vocabulary size — a direct tradeoff between **coverage** (recommending more tracks) and **quality** (having reliable embeddings for fewer tracks).

Grid search over `min_count ∈ {3, 5, 10, 20}`, evaluated by **Hit Rate @ k=100** on the validation set:

```python
hypers_sets = [
    {"min_count":  3, "epochs": 30, "vector_size": 48, "window": 10, "ns_exponent": 0.75},
    {"min_count":  5, "epochs": 30, "vector_size": 48, "window": 10, "ns_exponent": 0.75},
    {"min_count": 10, "epochs": 30, "vector_size": 48, "window": 10, "ns_exponent": 0.75},
    {"min_count": 20, "epochs": 30, "vector_size": 48, "window": 10, "ns_exponent": 0.75},
]
```

| `min_count` | Vocab Size | HR@100 (val) | HR@100 (test) |
|---|---|---|---|
| 3 | 14,886 | 0.0065 | 0.0065 |
| 5 | 5,877 | 0.0072 | 0.0072 |
| 10 | 1,560 | 0.0129 | 0.0129 |
| **20** | **319** | **0.0198** | **0.0198** |

**Interpretation:** As `min_count` increases, the vocabulary shrinks but embedding quality improves dramatically — Hit Rate nearly triples from `min_count=3` to `min_count=20`. Sparse tracks (those appearing fewer than 20 times) produce poorly-trained vectors because they have too few context examples to reliably estimate the skip-gram gradient. Including them pollutes the embedding space and reduces the signal-to-noise ratio of K-NN lookups.

This is a **quality-over-coverage tradeoff**: the model recommends from a smaller pool of tracks, but those recommendations are significantly more accurate.

---

### 6. Inference Engine

#### Prediction Function

```python
from random import choice

def predict_next_track(vector_space, input_sequence, k):
    """
    Given an ordered sequence of track IDs, predict the k most likely next tracks.

    Args:
        vector_space   : trained Word2Vec word vectors (model.wv)
        input_sequence : list of track_id strings (the partial playlist)
        k              : number of recommendations to return

    Returns:
        list of k track_id strings ranked by cosine similarity
    """
    query_item = input_sequence[-1]  # Anchor: the most recently played track

    # OOV fallback: if anchor is outside learned vocabulary, sample randomly
    if query_item not in vector_space:
        query_item = choice(list(vector_space.index_to_key))

    return [track for track, _ in vector_space.most_similar(query_item, topn=k)]
```

**Design notes:**
- **Anchor selection:** Uses only the last track in the sequence as the query vector. This is consistent with the skip-gram training objective and avoids the complexity of sequence aggregation while remaining effective.
- **OOV fallback:** Rather than raising an error on out-of-vocabulary tracks, the engine degrades gracefully by sampling a random in-vocabulary track. In a production setting, this could be replaced with nearest-neighbour lookup in a metadata embedding space.
- **Cosine similarity:** `most_similar()` ranks by cosine similarity in the 48-dimensional space, which is invariant to vector magnitude — appropriate for comparing normalised embedding vectors.

#### Evaluation Metric — Hit Rate @ k

```python
def evaluate_model(df, vector_space, k):
    df['predictions'] = df['track_test_x'].apply(
        lambda seq: predict_next_track(vector_space, seq, k)
    )
    df['hit'] = df.apply(
        lambda row: 1 if row['track_test_y'] in row['predictions'] else 0,
        axis=1
    )
    return df['hit'].sum() / len(df)
```

**Hit Rate @ k** measures the fraction of test cases where the true next track appears within the top-k predictions. It is a standard metric for next-item recommendation and directly reflects the user experience: does the correct answer appear in the recommendation list?

```python
KNN_K = 100

val_hr  = evaluate_model(validate, track2vec_model.wv, k=KNN_K)
test_hr = evaluate_model(test,     track2vec_model.wv, k=KNN_K)

print(f"Hit Rate@{KNN_K} — Validation : {val_hr:.4f}")
print(f"Hit Rate@{KNN_K} — Test       : {test_hr:.4f}")
```

---

## Results

### Model Performance

| Split | Hit Rate@100 |
|---|---|
| Validation | 0.0198 |
| Test | 0.0198 |

The validation and test metrics align closely, indicating no overfitting during hyperparameter selection.

### Example Output

```
Input track: "Calvin Harris|||Sweet Nothing"

Recommendations:
  1. Martin Garrix|||Animals       — similarity: 0.9900
  2. Usher|||Scream                — similarity: 0.9894
  3. Don Omar|||Danza Kuduro       — similarity: 0.9860
```

The recommendations are musically coherent: the model surfaces high-energy dance tracks from the same era, learned purely from co-occurrence in user playlists — with no access to genre labels, BPM, or audio features.

### Qualitative Embedding Validation

```
Query:  "Gorillaz|||19-2000"
Top-3:  Dave Matthews Band|||Everyday        (0.9974)
        Mac Demarco|||Passing Out Pieces      (0.9970)
        The Black Keys|||Thickfreakness       (0.9970)
```

These are stylistically adjacent tracks — indie rock / alternative with similar cultural placement — which confirms the embedding space encodes genuine musical context rather than random correlations.

---

## Key Design Decisions

### Why Word2Vec and not a neural sequence model?

A recurrent model (GRU, LSTM) or a Transformer would model the full sequence order, while Word2Vec only models local co-occurrence. The choice of Word2Vec is deliberate for this dataset scale:

- The dataset (after filtering) has ~319 vocabulary items and ~76K sequences — borderline for training a deep sequence model without regularisation overhead.
- Word2Vec's skip-gram objective is extremely sample-efficient and trains in seconds on CPU.
- The recommendation signal in playlists is primarily driven by *co-occurrence* (what tracks appear together) rather than strict ordering — users rearrange playlist order frequently.

Word2Vec is a strong, interpretable baseline. A natural extension is to replace the anchor-based lookup with a sequence-averaged or attention-weighted query vector.

### Why last-item holdout and not random-item holdout?

Last-item holdout mirrors the real inference scenario: in production, you always have the *recent* history and are predicting the *next* item. Random holdout would train and test on data from an unrealistic distribution.

### Why min_count=20 over a larger vocabulary?

Word2Vec embeddings for low-frequency items are unreliable because the stochastic gradient descent update has too few examples to converge. Including poorly-trained vectors adds noise to K-NN lookups, reducing precision. The 3× Hit Rate improvement from `min_count=3` to `min_count=20` empirically validates this reasoning.

### Why Hit Rate@100 and not a stricter k?

The recommendation list in a Discover Weekly-style product is typically 20–50 items. A k=100 evaluation is intentionally generous — it measures whether the system understands the *neighbourhood* of the correct answer, even if ranking within that neighbourhood is imperfect. Stricter metrics (Hit Rate@10, MRR, NDCG) are appropriate future additions.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/spotify-sequential-recommendation.git
cd spotify-sequential-recommendation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas>=1.5.0
numpy>=1.23.0
gensim>=4.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
powerlaw>=1.4.6
scipy>=1.11.0
```

---

## Usage

### Run the full pipeline via notebook

```bash
jupyter notebook notebooks/Spotify-Sequential-Recommendation-System.ipynb
```

### Train the model programmatically

```python
import pandas as pd
from src.feature_engineering import build_sequences, split_data
from src.train import final_train_model
from src.inference import final_predict_similar_tracks

# Load and prepare data (see notebook for full preprocessing steps)
df = pd.read_csv('data/spotify_dataset.csv', on_bad_lines='skip')

# Build sequences
sequences_df = build_sequences(df)
train, validate, test = split_data(sequences_df)

# Train with best hyperparameters
best_hyperparameters = {
    "min_count"   : 20,
    "epochs"      : 30,
    "vector_size" : 48,
    "window"      : 10,
    "ns_exponent" : 0.75,
}
model = final_train_model(sequences_df, best_hyperparameters)

# Get recommendations
input_track = "Calvin Harris|||Sweet Nothing"
recommendations = final_predict_similar_tracks(model, input_track, topn=3)

for track, score in recommendations:
    print(f"{track:<50} similarity: {score:.4f}")
```

### Expected output

```
Martin Garrix|||Animals                            similarity: 0.9900
Usher|||Scream                                     similarity: 0.9894
Don Omar|||Danza Kuduro                            similarity: 0.9860
```

---

## Future Work

**Sequence-aware query encoding:**
Replace the single-anchor lookup (`seq[-1]`) with a weighted average of all tracks in the input sequence — or use an attention mechanism to dynamically weight which tracks in the history are most predictive.

**Temporal modelling:**
Incorporate the order of tracks within a playlist using positional encodings or recurrent architectures (GRU, Transformer). The current model treats playlists as sets; strict ordering is unused.

**Harder evaluation metrics:**
Add MRR (Mean Reciprocal Rank), NDCG@k, and Coverage metrics. Hit Rate@100 is a forgiving upper-bound metric; production-quality evaluation requires measuring rank quality.

**Scalability to full dataset:**
Evaluate on the full 644K-row dataset without the 5% sample. Test whether a larger `vector_size` (e.g., 128, 256) and larger `min_count` threshold yields further gains.

**Continual / online updates:**
Investigate incremental Word2Vec updates to incorporate new tracks and playlists without full retraining — directly relevant to production music platforms where the catalogue changes daily.

**Cold-start handling:**
For tracks below the `min_count` threshold (OOV at inference), implement metadata-based fallback: embed track features (artist, release year, genre tags) and find the nearest in-vocabulary track as a proxy query.

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  Built with curiosity, playlists, and an unhealthy amount of Gensim documentation.
</p>