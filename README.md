# 🎬 Movie Review Star Rating Prediction  
**Classical ML Pipeline for Large-Scale Text Classification**

This repository contains two end-to-end machine learning pipelines (`movie_main.py` and `movie_baseline.py`) designed to **predict 1–5 star ratings** for over **1.9 million movie reviews** using **only classical ML methods**.  
The solution is fully streaming-based and optimized for **low-RAM environments (8GB)** while still achieving strong performance through **feature engineering, hashed text vectors, multi-epoch SGD training, and target encoding**.

---

# 📂 Repository Overview

### **`movie_baseline.py`**
A strong, optimized baseline that uses:
- Hashed **word n-grams** (1–2 or 1–3)
- Optional **character n-grams**
- Numeric linguistic features (lengths, punctuation, sentiment cues)
- **Target-encoded UserId & ProductId**
- Balanced class weights  
- Streaming training via `SGDClassifier`  

### **`movie_main.py`**
A more compact version of the above with the same core ideas, used for simpler experiment runs.

---

# 🧠 Project Summary

The task is to **predict the missing review scores** in the provided Kaggle-style dataset.  
You may use **any classical ML method**, except:
❌ neural networks  
❌ gradient boosting / XGBoost / LightGBM  
✔ allowed: LR, SVM, Naive Bayes, SGD, linear models, etc.

We build an efficient and scalable model that:
- Handles **1.7M training samples** without loading the full dataset into memory
- Extracts **rich text features**
- Learns **behavioral priors** for users/products
- Supports **probabilities & confidence estimates**
- Produces Kaggle-ready submission files

---

# 🏗️ Model Pipeline (In Depth)

This project follows a robust classical ML pipeline tailored for massive text data.

---

## 1. **Data Exploration**

Key observations from the dataset:
- Reviews contain **Summary**, **Text**, and metadata fields.
- Some **Score** values are missing → these form the prediction set.
- Large skew in rating distribution (many 4s and 5s) → requires class balancing.
- UserId/ProductId provide valuable historical signal.

Because of dataset size, exploration is streamed in parts.

---

## 2. **Feature Extraction & Engineering**

We combine three independent feature families into one large sparse matrix:

---

### **A. Text Features (Hashed Vectorization)**  
Using `HashingVectorizer` for memory safety:

| Feature | Description | Notes |
|--------|-------------|-------|
| Word unigrams & bigrams *(baseline)* | Captures semantics | stopwords removed |
| Word unigrams → trigrams *(final model)* | Boosts sentiment/context | slightly heavier |
| Character 3–5 grams *(optional)* | Captures misspellings, emphasis | improves generalization |
| Normalized TF–IDF style vectors | Via L2 norm | streaming-friendly |

No vocabulary is stored → safe for 1M+ features.

---

### **B. Numeric Text Features (Custom Engineered)**  

Extracted per review:

- log( # characters )
- log( # words )
- # exclamation marks
- # question marks
- ratio of uppercase letters
- ratio of digits
- elongated word count (“soooo”)
- sentence count
- tiny sentiment dictionary counts (positive/negative cues)

These improve rating separation significantly, especially between:
- 1–2 star reviews (negative cues)
- 4–5 star reviews (positive cues)

---

### **C. User & Product Target Encoding**

A powerful classical-ML trick:

- For each UserId/ProductId:
  - Compute smoothed mean rating:  
    **(sum + global_mean * m) / (count + m)**  
- Adds two numeric features:
  - *User mean rating prior*
  - *Product mean rating prior*

This captures tendencies like:
- Harsh reviewers
- Overrated or poorly rated products

No leakage occurs:  
Only labeled rows are used for computing priors.

---

# 3. **Model Creation**

We train a **logistic regression model** using **SGDClassifier**:

- `loss="log_loss"` → supports probabilities
- Supports **partial_fit()** → required for streaming
- High-dimensional sparse input is handled efficiently
- Balanced `class_weight` ensures fairness across stars

Penalty options:
- `l2` (default)
- `elasticnet` (if l1_ratio is set)
- `l1` (if l1_ratio ≥ 1)

---

# 4. **Model Training (Streaming & Multi-Epoch)**

### Why streaming?
- Training data = **1.7M reviews**
- Not possible to load into RAM
- We process chunks sequentially: `chunksize=40000`

### Training Loop
1. Read chunk of labeled rows  
2. Convert to engineered features  
3. `partial_fit()` the classifier  
4. Repeat for all chunks  
5. Optionally restart (`epochs > 1`) for better convergence

This mimics stochastic gradient descent over multiple epochs without exceeding RAM limits.

---

# 5. **Validation & Evaluation**

Validation is optional and hash-based:

- A review goes into validation if  
  `hash(Id + seed) < val_frac`

This ensures:
- Deterministic split
- No memory blowups
- No need to store arrays

Evaluation Metric:
### ✔ **Classification Accuracy**  
Same metric used on Kaggle.

Best validation accuracy achieved:
