# 🎬 Movie Review Star Rating Prediction

**Large-Scale Classical ML Pipeline for 1–5 Star Review Classification**

This repository contains two fully streaming, memory-efficient machine learning pipelines designed to predict **1–5 star ratings** for **1.9M+ movie reviews** using **only classical machine learning methods** (no deep learning, no boosting).

Both pipelines operate under **8GB RAM constraints**, using optimized text vectorization, extensive feature engineering, and multi-epoch SGD training.

### 📌 Final Leaderboard Scores

| Script                  | Validation Accuracy |
| ----------------------- | ------------------- |
| **`movie_main.py`**     | **0.65634**         |
| **`movie_baseline.py`** | **0.65584**         |

---

# 📁 Repository Files

### **`movie_main.py`**

A compact but optimized end-to-end pipeline with:

* Hashed word n-grams (1–2)
* Optional char n-grams
* Numeric linguistic features
* User/Product target encoding
* Balanced class weights
* Streaming SGD training

### **`movie_baseline.py`**

A more flexible and tuneable version supporting:

* Word trigrams
* ElasticNet/L1/L2 penalties
* Multi-epoch streaming
* Confidence/probability outputs
* Enhanced training options

---

# 📊 Data Science Workflow

Below is a detailed explanation of everything done across both scripts, following the required six-step structure.

---

# 1. **Data Exploration**

Because the dataset contains **1.7M training rows** and **212k test rows**, full in-RAM EDA is not possible on an 8GB system. Instead, exploration was done in **streamed batches**.

Key observations:

### **Dataset Structure**

* **Id** — unique identifier
* **Summary** — short headline
* **Text** — full review text
* **UserId** — reviewer identifier
* **ProductId** — movie/product identifier
* **Score** — 1–5 stars (target; missing for rows to predict)

### **Findings**

* **Highly skewed distribution**: most ratings are 4 or 5
  → requires **balanced class weights**
* **UserId and ProductId repeat many times**
  → strong potential for **target encoding**
* **Text varies significantly in length**
  → numeric text features help separate classes
* **Some fields are missing**
  → pipeline must gracefully handle NaNs
* **Text contains many informal patterns** (elongations, punctuation bursts, capitalization)
  → char n-grams + numeric features helpful

These insights shaped our feature engineering strategy.

---

# 2. **Feature Extraction / Engineering**

The model uses **three complementary feature families**, combined into a single sparse matrix:

---

## A. **Hashed Text Vectorization (Main Signal)**

Implemented using `HashingVectorizer` for scalability:

* **Word n-grams**

  * baseline: (1, 2)
  * advanced: (1, 3)
* **Character n-grams** (3–5 grams, optional)
  These capture stylistic patterns, elongated words, misspellings, emotional emphasis, etc.

### Why Hashing?

* No vocabulary stored → O(1) memory
* Supports **524K–1M+ features**
* Enables full streaming

---

## B. **Custom Numeric Linguistic Features**

Extracted from each review:

* log(# characters)
* log(# words)
* count(!)
* count(?)
* ratio of uppercase letters
* ratio of digits
* elongated words (“soooo”, “yessss”)
* sentences
* small **positive/negative cue dictionaries**

  * captures sentiment polarity

These stabilize prediction boundaries between adjacent star ratings (1 vs 2, 4 vs 5).

---

## C. **User & Product Target Encoding**

A major gain in performance came from encoding:

* **User behavior priors**
* **Product average rating priors**

Formula:

[
\text{encoded value} = \frac{\text{sum} + m \cdot \text{global mean}}{\text{count} + m}
]

with smoothing `m = 50` to avoid overfitting.

This feature captures tendencies such as:

* Users who consistently give low ratings
* Products that are universally loved or disliked

No leakage occurs because *only labeled rows* are used to compute encodings.

---

# 3. **Model Creation and Assumptions**

We use:

### ✔ `SGDClassifier` with `loss="log_loss"` (logistic regression)

Because:

* Supports **partial_fit()** → streaming
* Works on **huge sparse matrices**
* Produces **probabilities + confidence**
* Highly scalable to >1M features
* Converges well with multiple epochs

### Model Assumptions

* Linearity in the transformed feature space is sufficient
* Large, diverse n-gram sets capture sentiment and semantics
* User/Product priors meaningfully shift probability mass
* Rating prediction is a multiclass classification task (5 classes)
* Balanced class weights correct skewed label distribution

---

# 4. **Model Tuning**

### Techniques that improved performance:

#### **1. Increasing Hashed Feature Space**

* 262K → 524K → 1M features
* Improves text resolution
* Higher collision resistance

#### **2. Char n-grams**

* Help with:

  * emphasis
  * elongated words
  * misspellings
* Especially useful in 1-star and 5-star extremes

#### **3. Word Trigrams**

Adds contextual sentiment (“not very good”, “one of the best”).

#### **4. Multi-Epoch Streaming**

Repeat streaming training several times:

```bash
--epochs 2
```

#### **5. Regularization Tuning**

* L2 (stable baseline)
* ElasticNet (useful when high n-grams included)

#### **6. Balanced Class Weights**

Prevents overprediction of 4 and 5 stars.

---

# 5. **Model Evaluation / Performance**

Evaluation metric:

### ✔ **Classification Accuracy**

Matching the official Kaggle scoring.

### **Final Validation Scores**

| Script                  | Accuracy    | Notes                                 |
| ----------------------- | ----------- | ------------------------------------- |
| **`movie_main.py`**     | **0.65634** | Best performing version               |
| **`movie_baseline.py`** | **0.65584** | Slightly behind but more configurable |

### Why the scores plateau?

Given:

* No deep learning
* No boosting
* No embeddings
* Heavy memory constraints

~0.656 is a strong performance for classical ML on 1.9M+ noisy text reviews.

---

# 6. **Struggles / Issues / Open Questions**

### **A. Severe Memory Constraints (8GB RAM)**

* Cannot load dataset fully
* Cannot use TF-IDF vocabulary
* Cannot use logistic regression from sklearn (too heavy)
* Must avoid dense matrices at all costs
  → HashingVectorizer + streaming was essential

### **B. Long Training Time**

Streaming across 1.7M rows × multiple epochs is slow.

---

# 🚀 Running the Scripts

## Example: `movie_main.py`

```bash
python movie_main.py \
  --train data/train.csv \
  --test data/test.csv \
  --sample data/sample.csv \
  --out submission.csv \
  --features 524288 \
  --chunksize 40000 \
  --use_char \
  --val_frac 0.05
```

## Example: `movie_baseline.py`

```bash
python movie_baseline.py \
  --train data/train.csv \
  --test data/test.csv \
  --sample data/sample.csv \
  --out submission_final.csv \
  --features 524288 \
  --chunksize 40000 \
  --use_char \
  --word_trigrams \
  --epochs 2 \
  --alpha 1e-5 \
  --with_confidence preds_conf_final.csv
```

---

# 📦 Output Files

* `submission.csv` — Kaggle-ready file (Id, Score)
* Optional:

  * `preds_conf.csv` — per-review confidence scores
  * `preds_prob.csv` — per-class probabilities

---

# 🏁 Final Note

This project demonstrates that **classical machine learning**, when engineered properly, can scale to millions of samples and deliver competitive accuracy—without deep learning, GPUs, or high-memory hardware.
