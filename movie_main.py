#!/usr/bin/env python3
"""
Usage (example)
$ python movie_main.py \
    --train data/train.csv \
    --test data/test.csv \
    --sample data/sample.csv \
    --out submission.csv \
    --features 262144 \
    --chunksize 40000 \
    --use_char \
    --val_frac 0.05 \
    --with_confidence preds_with_confidence.csv \
    --with_probabilities preds_with_probabilities.csv \
    --user_prod_encoding on
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import re

import numpy as np
import pandas as pd
from scipy import sparse as sp

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

RANDOM_STATE = 42
CLASSES = np.array([1, 2, 3, 4, 5], dtype=int)

# Tiny, embedded sentiment cue lists (kept minimal for speed)
POS_WORDS = set(
    "great excellent amazing love loved awesome wonderful perfect fantastic favorite outstanding superb"
    .split()
)
NEG_WORDS = set(
    "bad terrible awful hate hated worst poor disappointing boring annoying broken useless"
    .split()
)

# -------------------------
# Helpers
# -------------------------

def _resolve(p: Path) -> Path:
    p = Path(os.path.expanduser(str(p)))
    if p.suffix == "":
        for suf in (".csv", ".csv.gz"):
            if (p.with_suffix(suf)).exists():
                return p.with_suffix(suf)
    return p

def read_data_paths(train_path: Path, test_path: Path, sample_path: Path) -> Tuple[Path, Path, Path]:
    train_path = _resolve(train_path)
    test_path = _resolve(test_path)
    sample_path = _resolve(sample_path)
    for p in (train_path, test_path, sample_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
    return train_path, test_path, sample_path

def basic_clean_series(s: pd.Series) -> pd.Series:
    # lowercase, squash whitespace
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

def make_text_column_df(df: pd.DataFrame) -> pd.Series:
    summary = basic_clean_series(df.get("Summary", pd.Series([""] * len(df))))
    text = basic_clean_series(df.get("Text", pd.Series([""] * len(df))))
    return (summary + " . " + text).str.strip()

def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.maximum(den, 1.0)
    return num / den

def add_text_numeric_features(df: pd.DataFrame) -> sp.csr_matrix:
    txt = make_text_column_df(df)

    len_chars = txt.str.len().to_numpy(dtype=np.float32, na_value=0.0)
    len_words = txt.str.split().str.len().to_numpy(dtype=np.float32, na_value=0.0)
    num_ex = txt.str.count(r"!").to_numpy(dtype=np.float32)
    num_qm = txt.str.count(r"\?").to_numpy(dtype=np.float32)
    num_caps = txt.str.count(r"[A-Z]").to_numpy(dtype=np.float32)  # uppercase before lowercasing
    num_digits = txt.str.count(r"\d").to_numpy(dtype=np.float32)
    num_elong = txt.str.count(r"([a-z])\1{2,}").to_numpy(dtype=np.float32)  # e.g., "soooo"
    num_sent = txt.str.count(r"[.!?]+").to_numpy(dtype=np.float32)

    # Fast-ish sentiment cue counts via regex alternations
    if POS_WORDS:
        pos_pat = r"\b(" + "|".join(re.escape(w) for w in POS_WORDS) + r")\b"
        pos_counts = txt.str.count(pos_pat, flags=0).to_numpy(dtype=np.float32)
    else:
        pos_counts = np.zeros_like(len_chars)

    if NEG_WORDS:
        neg_pat = r"\b(" + "|".join(re.escape(w) for w in NEG_WORDS) + r")\b"
        neg_counts = txt.str.count(neg_pat, flags=0).to_numpy(dtype=np.float32)
    else:
        neg_counts = np.zeros_like(len_chars)
    
    feat = np.vstack([
        np.log1p(len_chars),
        np.log1p(len_words),
        np.log1p(num_ex),
        np.log1p(num_qm),
        _ratio(num_caps, len_chars),
        _ratio(num_digits, len_chars),
        np.log1p(num_elong),
        np.log1p(num_sent),
        np.log1p(pos_counts),
        np.log1p(neg_counts),
    ]).T

    return sp.csr_matrix(feat)

def build_vectorizers(n_features: int, use_char: bool):
    hv_word = HashingVectorizer(
        n_features=(n_features if not use_char else n_features // 2),
        alternate_sign=False,
        analyzer="word",
        ngram_range=(1, 2),
        norm="l2",
        stop_words="english",
    )
    hv_char = None
    if use_char:
        hv_char = HashingVectorizer(
            n_features=n_features // 2,
            alternate_sign=False,
            analyzer="char",
            ngram_range=(3, 5),
            norm="l2",
        )
    return hv_word, hv_char

def vectorize_text_series(series: pd.Series, hv_word, hv_char=None) -> sp.csr_matrix:
    Xw = hv_word.transform(series)
    if hv_char is None:
        return Xw
    Xc = hv_char.transform(series)
    return sp.hstack([Xw, Xc], format="csr")

# -------------------------
# First-pass stats (weights, encodings)
# -------------------------

def compute_balanced_class_weights(train_csv: Path, chunksize: int) -> Dict[int, float]:
    counts = {int(c): 0 for c in CLASSES}
    for chunk in pd.read_csv(train_csv, usecols=["Score"], chunksize=chunksize):
        y = chunk["Score"].dropna().astype(int).to_numpy()
        if y.size == 0:
            continue
        vals, freq = np.unique(y, return_counts=True)
        for v, f in zip(vals, freq):
            if int(v) in counts:
                counts[int(v)] += int(f)
    total = sum(counts.values()) or 1
    n_classes = len(CLASSES)
    return {c: (total / (n_classes * counts[c])) if counts[c] > 0 else 1.0 for c in counts}

def compute_user_product_encoding(train_csv: Path, chunksize: int, m: float = 50.0) -> Tuple[Dict[str, tuple], Dict[str, tuple], float]:
    """Return (user_map, prod_map, global_mean).
    Store tuples (sum_scores, count). Use smoothed mean: (sum + global_mean*m) / (count + m).
    Only from labeled rows -> no leakage.
    """
    user_sum: Dict[str, float] = {}
    user_cnt: Dict[str, int] = {}
    prod_sum: Dict[str, float] = {}
    prod_cnt: Dict[str, int] = {}
    total_sum = 0.0
    total_cnt = 0

    cols = ["UserId", "ProductId", "Score"]
    for chunk in pd.read_csv(train_csv, usecols=cols, chunksize=chunksize):
        sub = chunk[chunk["Score"].notna()]
        if sub.empty:
            continue
        y = sub["Score"].astype(float).to_numpy()
        total_sum += float(y.sum())
        total_cnt += int(y.size)

        for uid, s in zip(sub["UserId"].fillna("<na>").astype(str), y):
            user_sum[uid] = user_sum.get(uid, 0.0) + float(s)
            user_cnt[uid] = user_cnt.get(uid, 0) + 1

        for pid, s in zip(sub["ProductId"].fillna("<na>").astype(str), y):
            prod_sum[pid] = prod_sum.get(pid, 0.0) + float(s)
            prod_cnt[pid] = prod_cnt.get(pid, 0) + 1

    global_mean = float(total_sum / total_cnt) if total_cnt > 0 else 3.0
    user_map = {k: (user_sum[k], user_cnt[k]) for k in user_sum}
    prod_map = {k: (prod_sum[k], prod_cnt[k]) for k in prod_sum}
    return user_map, prod_map, global_mean

def encode_user_product(df: pd.DataFrame, user_map: Dict[str, tuple], prod_map: Dict[str, tuple], global_mean: float, m: float = 50.0) -> sp.csr_matrix:
    uid = df.get("UserId", pd.Series(["<na>"] * len(df))).fillna("<na>").astype(str)
    pid = df.get("ProductId", pd.Series(["<na>"] * len(df))).fillna("<na>").astype(str)
    u_feat = np.empty(len(df), dtype=np.float32)
    p_feat = np.empty(len(df), dtype=np.float32)
    for i, u in enumerate(uid):
        s, c = user_map.get(u, (0.0, 0))
        u_feat[i] = (s + global_mean * m) / (c + m)
    for i, p in enumerate(pid):
        s, c = prod_map.get(p, (0.0, 0))
        p_feat[i] = (s + global_mean * m) / (c + m)
    return sp.csr_matrix(np.vstack([u_feat, p_feat]).T)

# -------------------------
# Validation split (optional)
# -------------------------

def choose_validation_mask(ids: np.ndarray, frac: float, seed: int) -> np.ndarray:
    import hashlib
    if frac <= 0:
        return np.zeros_like(ids, dtype=bool)
    vals = []
    for rid in ids:
        h = hashlib.md5(f"{int(rid)}_{seed}".encode()).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        vals.append(val)
    vals = np.asarray(vals)
    return vals < frac

def stream_validate_accuracy(train_csv: Path, clf: SGDClassifier, hv_word, hv_char, chunksize: int,
                             val_frac: float, val_seed: int,
                             user_map: Optional[Dict[str, tuple]],
                             prod_map: Optional[Dict[str, tuple]],
                             global_mean: float) -> Optional[float]:
    if val_frac <= 0:
        return None
    total = 0
    correct = 0
    reader = pd.read_csv(train_csv, usecols=["Id", "Summary", "Text", "Score", "UserId", "ProductId"], chunksize=chunksize)
    for chunk in reader:
        labeled = chunk[chunk["Score"].notna()].copy()
        if labeled.empty:
            continue
        mask_val = choose_validation_mask(labeled["Id"].to_numpy(), val_frac, val_seed)
        sub = labeled.loc[mask_val]
        if sub.empty:
            continue
        y_true = sub["Score"].astype(int).to_numpy()
        text = make_text_column_df(sub)
        X_text = vectorize_text_series(text, hv_word, hv_char)
        X_num = add_text_numeric_features(sub)
        X_enc = encode_user_product(sub, user_map or {}, prod_map or {}, global_mean)
        X = sp.hstack([X_text, X_num, X_enc], format="csr")
        y_pred = clf.predict(X).astype(int)
        total += y_true.size
        correct += int((y_true == y_pred).sum())
        del chunk, labeled, sub, y_true, text, X_text, X_num, X_enc, X, y_pred
        gc.collect()
    if total == 0:
        return None
    return correct / total

# -------------------------
# Streaming train / predict
# -------------------------

def stream_train(train_csv: Path, clf: SGDClassifier, hv_word, hv_char, chunksize: int,
                 val_frac: float, val_seed: int,
                 user_map: Optional[Dict[str, tuple]], prod_map: Optional[Dict[str, tuple]], global_mean: float) -> None:
    reader = pd.read_csv(train_csv, usecols=["Id", "Summary", "Text", "Score", "UserId", "ProductId"], chunksize=chunksize)
    first = True
    processed = 0
    for chunk in reader:
        labeled = chunk[chunk["Score"].notna()].copy()
        if labeled.empty:
            continue
        # If we're doing validation, exclude the held-out rows from training
        if val_frac > 0:
            mask_val = choose_validation_mask(labeled["Id"].to_numpy(), val_frac, val_seed)
            labeled = labeled.loc[~mask_val]
        if labeled.empty:
            continue

        y = labeled["Score"].astype(int).to_numpy()
        text = make_text_column_df(labeled)
        X_text = vectorize_text_series(text, hv_word, hv_char)
        X_num = add_text_numeric_features(labeled)
        X_enc = encode_user_product(labeled, user_map or {}, prod_map or {}, global_mean)
        X = sp.hstack([X_text, X_num, X_enc], format="csr")

        if first:
            clf.partial_fit(X, y, classes=CLASSES)
            first = False
        else:
            clf.partial_fit(X, y)

        processed += len(labeled)
        if processed % (chunksize * 5) == 0:
            print(f"  trained on {processed} labeled rows…", flush=True)

        del chunk, labeled, y, text, X_text, X_num, X_enc, X
        gc.collect()

def stream_predict_for_ids(train_csv: Path, ids: np.ndarray, clf: SGDClassifier, hv_word, hv_char, chunksize: int,
                           want_proba: bool,
                           user_map: Optional[Dict[str, tuple]], prod_map: Optional[Dict[str, tuple]], global_mean: float
                           ) -> Dict[int, Dict[str, object]]:
    """Return map: Id -> {'score': int, 'conf': float, 'proba': Optional[np.ndarray]}"""
    id_set = set(int(x) for x in ids.tolist())
    result: Dict[int, Dict[str, object]] = {}
    reader = pd.read_csv(train_csv, usecols=["Id", "Summary", "Text", "UserId", "ProductId"], chunksize=chunksize)
    for chunk in reader:
        sub = chunk[chunk["Id"].isin(id_set)]
        if sub.empty:
            continue
        text = make_text_column_df(sub)
        X_text = vectorize_text_series(text, hv_word, hv_char)
        X_num = add_text_numeric_features(sub)
        X_enc = encode_user_product(sub, user_map or {}, prod_map or {}, global_mean)
        X = sp.hstack([X_text, X_num, X_enc], format="csr")

        pred = clf.predict(X).astype(int)
        if want_proba and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
            maxp = proba.max(axis=1)
        else:
            proba = None
            maxp = np.full(pred.shape[0], np.nan, dtype=float)

        for i, rid in enumerate(sub["Id"].to_numpy()):
            result[int(rid)] = {
                "score": int(pred[i]),
                "conf": float(maxp[i]),
                "proba": (proba[i].copy() if proba is not None else None),
            }

        del chunk, sub, text, X_text, X_num, X_enc, X, pred
        if want_proba and proba is not None:
            del proba, maxp
        gc.collect()
    return result

# -------------------------
# Main
# -------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--sample", default="sample.csv")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--features", type=int, default=2**18, help="hash features (reduce for low RAM)")
    parser.add_argument("--chunksize", type=int, default=50000, help="rows per chunk")
    parser.add_argument("--use_char", action="store_true", help="add char 3-5grams (memory heavier)")
    parser.add_argument("--with_confidence", default=None, help="optional CSV with Id,Score,Confidence")
    parser.add_argument("--with_probabilities", default=None, help="optional CSV with Id and per-class probabilities")
    parser.add_argument("--val_frac", type=float, default=0.0, help="hold out fraction of labeled rows for validation (0 disables)")
    parser.add_argument("--val_seed", type=int, default=123, help="seed for validation split")
    parser.add_argument("--user_prod_encoding", choices=["on", "off"], default="on", help="enable target/freq encoding for UserId/ProductId")
    args = parser.parse_args(argv)

    # Auto-redirect to ./data if present
    if args.train == "train.csv" and Path("data/train.csv").exists():
        args.train = "data/train.csv"
    if args.test == "test.csv" and Path("data/test.csv").exists():
        args.test = "data/test.csv"
    if args.sample == "sample.csv" and Path("data/sample.csv").exists():
        args.sample = "data/sample.csv"

    train_csv, test_csv, sample_csv = read_data_paths(Path(args.train), Path(args.test), Path(args.sample))

    # Build vectorizers
    hv_word, hv_char = build_vectorizers(args.features, args.use_char)

    # First pass: class weights and (optionally) user/product encodings
    print("Computing class weights…", flush=True)
    weights = compute_balanced_class_weights(train_csv, chunksize=max(200000, args.chunksize))

    user_map: Dict[str, tuple] = {}
    prod_map: Dict[str, tuple] = {}
    global_mean: float = 3.0
    if args.user_prod_encoding == "on":
        print("Computing user/product encodings…", flush=True)
        user_map, prod_map, global_mean = compute_user_product_encoding(
            train_csv, chunksize=max(200000, args.chunksize)
        )

    # Classifier (logistic => predict_proba available)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        random_state=RANDOM_STATE,
        class_weight=weights,
    )

    print("Training (streaming)…", flush=True)
    stream_train(
        train_csv, clf, hv_word, hv_char, args.chunksize,
        val_frac=args.val_frac, val_seed=args.val_seed,
        user_map=(user_map if args.user_prod_encoding == "on" else None),
        prod_map=(prod_map if args.user_prod_encoding == "on" else None),
        global_mean=global_mean,
    )

    # Optional validation accuracy using the same metric as eval (accuracy)
    if args.val_frac > 0:
        print("Evaluating on validation split…", flush=True)
        val_acc = stream_validate_accuracy(
            train_csv, clf, hv_word, hv_char, args.chunksize,
            args.val_frac, args.val_seed,
            user_map=(user_map if args.user_prod_encoding == "on" else None),
            prod_map=(prod_map if args.user_prod_encoding == "on" else None),
            global_mean=global_mean,
        )
        if val_acc is not None:
            print(f"Validation accuracy: {val_acc:.4f}")

    print("Preparing prediction ids…", flush=True)
    test_df = pd.read_csv(test_csv)
    ids = test_df["Id"].to_numpy()

    print("Predicting (streaming)…", flush=True)
    want_proba = bool(args.with_confidence or args.with_probabilities)
    pred_map = stream_predict_for_ids(
        train_csv, ids, clf, hv_word, hv_char, args.chunksize, want_proba,
        user_map=(user_map if args.user_prod_encoding == "on" else None),
        prod_map=(prod_map if args.user_prod_encoding == "on" else None),
        global_mean=global_mean,
    )

    # Build Kaggle submission aligning to sample.csv order
    sample = pd.read_csv(sample_csv)
    sub = sample[["Id"]].copy()
    sub["Score"] = [int(pred_map.get(int(i), {"score": 3})["score"]) for i in sub["Id"].to_numpy()]
    sub["Score"] = sub["Score"].astype(int)
    out_path = Path(args.out)
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(sub)} rows.")

    # Optional: write confidence file
    if args.with_confidence:
        conf_rows = []
        for i in sample["Id"].to_numpy():
            rec = pred_map.get(int(i))
            if rec is None:
                conf_rows.append({"Id": int(i), "Score": 3, "Confidence": np.nan})
            else:
                conf_rows.append({"Id": int(i), "Score": int(rec["score"]), "Confidence": float(rec["conf"])})
        pd.DataFrame(conf_rows).to_csv(args.with_confidence, index=False)
        print(f"Wrote confidence file: {args.with_confidence}")

    # Optional: write per-class probabilities (prob_1..prob_5)
    if args.with_probabilities:
        prob_rows = []
        for i in sample["Id"].to_numpy():
            rec = pred_map.get(int(i))
            row: Dict[str, object] = {"Id": int(i)}
            if rec is None or rec["proba"] is None:
                for c in CLASSES:
                    row[f"prob_{c}"] = np.nan
            else:
                # Map clf.classes_ -> 1..5 columns
                class_to_prob = {int(clf.classes_[k]): float(rec["proba"][k]) for k in range(len(clf.classes_))}
                for c in CLASSES:
                    row[f"prob_{c}"] = class_to_prob.get(int(c), 0.0)
            prob_rows.append(row)
        pd.DataFrame(prob_rows).to_csv(args.with_probabilities, index=False)
        print(f"Wrote probabilities file: {args.with_probabilities}")

if __name__ == "__main__":
    sys.exit(main())
