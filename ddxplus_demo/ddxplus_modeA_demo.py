#!/usr/bin/env python3
"""
DDXPlus — Mode-A differential engine demo.

What this does
--------------
1. ESTIMATE (training):  count -> Laplace-smoothed pathology priors p(d) and
   per-evidence likelihoods p(e | d) over the 1.3M DDXPlus patients.
2. INFER (mode A):  feed each TEST patient's *complete* evidence record at once
   into a naive-Bayes inverter -> a dense posterior over the 49 pathologies =
   the predicted differential.
3. SCORE:  DDR / DDP / DDF1 against the dataset's ground-truth differential,
   using the *exact* definition from the official repo
   (mila-iqia/ddxplus  code/aarlc/ddxplus_code/metrics.py):
       gt_mask   = gt_differential  > 0.01
       pred_mask = predicted_diff   > 0.01
       DDR  = |gt_mask & pred_mask| / |gt_mask|     (per patient, then averaged)
       DDP  = |gt_mask & pred_mask| / |pred_mask|
       DDF1 = 2*DDP*DDR / (DDP + DDR)

Mode A == "feed all evidences, no question loop". It isolates whether the
estimated tables are correct, decoupled from any question-selection policy.

Usage
-----
  # Offline correctness proof (no data / no network needed): builds a synthetic
  # dataset in the exact DDXPlus schema whose ground-truth differential IS the
  # true Bayes posterior, so a correct pipeline must score DDR/DDP ~ 1.0.
  python ddxplus_modeA_demo.py --selftest

  # Real data: point at a folder holding the official DDXPlus release files
  #   release_evidences.json, release_conditions.json,
  #   release_train_patients.csv, release_test_patients.csv
  python ddxplus_modeA_demo.py --data-dir /path/to/ddxplus \
         --max-train 1000000 --max-test 50000
"""

import argparse
import ast
import json
import os
import sys
import time
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Parsing helpers
# --------------------------------------------------------------------------- #
def parse_list(cell):
    """EVIDENCES / DIFFERENTIAL_DIAGNOSIS are stringified python lists."""
    if isinstance(cell, (list, tuple)):
        return cell
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return ast.literal_eval(cell)


def load_release(data_dir):
    """Return (pathologies, severity, train_df, test_df) from official files."""
    with open(os.path.join(data_dir, "release_conditions.json")) as f:
        conditions = json.load(f)
    # release_conditions.json may be a dict keyed by name, or a list of objects
    if isinstance(conditions, dict):
        items = list(conditions.values())
    else:
        items = conditions

    def cname(c):
        return (
            c.get("condition_name") or c.get("cond-name-eng") or c.get("cond-name-fr")
        )

    pathologies = sorted(cname(c) for c in items)
    severity = {cname(c): c.get("severity") for c in items}

    def read_patients(*names):
        for n in names:
            p = os.path.join(data_dir, n)
            if os.path.exists(p):
                return pd.read_csv(p)
        raise FileNotFoundError(f"none of {names} found in {data_dir}")

    train = read_patients("release_train_patients.csv", "release_train_patients.zip")
    test = read_patients("release_test_patients.csv", "release_test_patients.zip")
    return pathologies, severity, train, test


# --------------------------------------------------------------------------- #
#  Estimation  (the entire "training" = counting + smoothing)
# --------------------------------------------------------------------------- #
def estimate(df, pathologies, alpha=1.0):
    """Estimate priors p(d) and per-token likelihoods p(token present | d).

    A 'token' is one entry of the EVIDENCES list: either a binary code 'E_x'
    or a categorical/multi value 'E_x_@_value'. Each distinct token is treated
    as its own binary feature (present / absent) -- which is exactly how the
    dataset encodes them, and matches the conditional-independence factorisation
    used to *generate* the data.
    """
    D = len(pathologies)
    pidx = {p: i for i, p in enumerate(pathologies)}

    paths = df["PATHOLOGY"].map(pidx)
    if paths.isna().any():
        bad = df["PATHOLOGY"][paths.isna()].unique()[:5]
        raise ValueError(f"PATHOLOGY values not in conditions.json: {bad}")
    paths = paths.to_numpy()

    ev_lists = df["EVIDENCES"].map(parse_list).tolist()

    # vocab
    vocab = {}
    for evs in ev_lists:
        for t in evs:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)

    Nd = np.zeros(D)
    for d in paths:
        Nd[d] += 1

    # cnt[d, token] via scatter-add
    rows, cols = [], []
    for d, evs in zip(paths, ev_lists):
        for t in evs:
            rows.append(d)
            cols.append(vocab[t])
    cnt = np.zeros((D, V))
    if rows:
        np.add.at(cnt, (np.asarray(rows), np.asarray(cols)), 1.0)

    prior = (Nd + alpha) / (Nd.sum() + alpha * D)
    # binary-style smoothing: 2 outcomes (present / absent)
    p_present = (cnt + alpha) / (Nd[:, None] + 2 * alpha)
    p_present = np.clip(p_present, 1e-9, 1 - 1e-9)

    LP_present = np.log(p_present)
    LP_absent = np.log(1.0 - p_present)
    base = np.log(prior) + LP_absent.sum(axis=1)  # [D]

    return dict(
        pidx=pidx,
        vocab=vocab,
        prior=prior,
        LP_present=LP_present,
        LP_absent=LP_absent,
        base=base,
    )


# --------------------------------------------------------------------------- #
#  Mode-A inference: feed ALL evidences at once
# --------------------------------------------------------------------------- #
def predict_posterior(tokens, model):
    vocab, LP_present, LP_absent, base = (
        model["vocab"],
        model["LP_present"],
        model["LP_absent"],
        model["base"],
    )
    cols = [vocab[t] for t in tokens if t in vocab]
    logp = base.copy()
    if cols:
        cols = np.asarray(cols)
        logp = logp + (LP_present[:, cols] - LP_absent[:, cols]).sum(axis=1)
    logp -= logp.max()
    post = np.exp(logp)
    post /= post.sum()
    return post


# --------------------------------------------------------------------------- #
#  Scoring  (matches official metrics.py exactly)
# --------------------------------------------------------------------------- #
def evaluate(df, model, pathologies, tres=0.01, max_n=None):
    pidx = model["pidx"]
    D = len(pathologies)
    if max_n:
        df = df.head(max_n)

    ddr_l, ddp_l, ddf1_l, gtpa_l = [], [], [], []
    skipped = 0
    for _, row in df.iterrows():
        gt_pairs = parse_list(row["DIFFERENTIAL_DIAGNOSIS"])
        if not gt_pairs:
            skipped += 1
            continue
        gt = np.zeros(D)
        ok = True
        for name, prob in gt_pairs:
            if name not in pidx:  # differential names must be known
                ok = False
                break
            gt[pidx[name]] = prob
        if not ok:
            skipped += 1
            continue

        post = predict_posterior(parse_list(row["EVIDENCES"]), model)

        gt_mask, pred_mask = gt > tres, post > tres
        inter = np.logical_and(gt_mask, pred_mask).sum()
        r = inter / max(1, gt_mask.sum())
        p = inter / max(1, pred_mask.sum())
        f1 = (2 * p * r) / (p + r + 1e-10)
        ddr_l.append(r)
        ddp_l.append(p)
        ddf1_l.append(f1)

        true_d = pidx.get(row["PATHOLOGY"])
        if true_d is not None:
            gtpa_l.append(float(post[true_d] > tres))

    return dict(
        n=len(ddr_l),
        skipped=skipped,
        DDR=float(np.mean(ddr_l)),
        DDP=float(np.mean(ddp_l)),
        DDF1=float(np.mean(ddf1_l)),
        GTPA=float(np.mean(gtpa_l)),
    )


# --------------------------------------------------------------------------- #
#  Synthetic self-test: data generated by naive Bayes => gt diff = true posterior
# --------------------------------------------------------------------------- #
def make_synthetic(n_patients, D=10, V=60, seed=0, n_train_ratio=0.7):
    rng = np.random.default_rng(seed)
    pathologies = [f"path_{i}" for i in range(D)]
    # distinct-ish likelihoods so diseases are separable
    true_p = rng.beta(0.4, 0.4, size=(D, V))  # in (0,1), pushed to extremes
    true_prior = rng.dirichlet(np.ones(D) * 2.0)
    log_prior = np.log(true_prior)
    LPp, LPa = np.log(true_p), np.log(1 - true_p)

    rows = []
    for _ in range(n_patients):
        d = rng.choice(D, p=true_prior)
        present = rng.random(V) < true_p[d]  # full evidence vector
        tokens = [f"E_{j}" for j in range(V) if present[j]]
        # TRUE Bayes posterior given the full evidence vector = ground-truth diff
        logpost = log_prior + (np.where(present[None, :], LPp, LPa)).sum(axis=1)
        logpost -= logpost.max()
        post = np.exp(logpost)
        post /= post.sum()
        diff = [[pathologies[i], float(post[i])] for i in range(D)]
        rows.append(
            dict(
                AGE=30,
                SEX="F",
                PATHOLOGY=pathologies[d],
                EVIDENCES=str(tokens),
                DIFFERENTIAL_DIAGNOSIS=str(diff),
                INITIAL_EVIDENCE=tokens[0] if tokens else "",
            )
        )
    df = pd.DataFrame(rows)
    cut = int(len(df) * n_train_ratio)
    severity = {p: 3 for p in pathologies}
    return pathologies, severity, df.iloc[:cut].copy(), df.iloc[cut:].copy()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="folder with official DDXPlus release files")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="run offline synthetic correctness proof",
    )
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=20000)
    ap.add_argument("--synthetic-n", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.time()
    if args.selftest:
        print("== SELF-TEST (synthetic, gt differential == true Bayes posterior) ==")
        pathologies, severity, train, test = make_synthetic(args.synthetic_n)
    elif args.data_dir:
        print(f"== REAL DATA: {args.data_dir} ==")
        pathologies, severity, train, test = load_release(args.data_dir)
    else:
        sys.exit("provide --selftest or --data-dir")

    if args.max_train:
        train = train.head(args.max_train)
    print(f"pathologies={len(pathologies)}  train={len(train)}  test={len(test)}")

    model = estimate(train, pathologies, alpha=args.alpha)
    print(f"estimated: vocab={len(model['vocab'])} tokens  ({time.time() - t0:.1f}s)")

    res = evaluate(test, model, pathologies, max_n=args.max_test)
    print(
        f"\nevaluated {res['n']} patients (skipped {res['skipped']})  "
        f"[{time.time() - t0:.1f}s total]"
    )
    print("-" * 44)
    print(f"  DDR  (recall)    = {res['DDR'] * 100:6.2f} %")
    print(f"  DDP  (precision) = {res['DDP'] * 100:6.2f} %")
    print(f"  DDF1             = {res['DDF1'] * 100:6.2f} %")
    print(f"  GTPA (true patho in diff) = {res['GTPA'] * 100:6.2f} %")
    print("-" * 44)
    if args.selftest:
        ok = res["DDR"] > 0.95 and res["DDP"] > 0.95
        print(
            "SELF-TEST",
            "PASS ✓" if ok else "FAIL ✗",
            "(expect DDR & DDP > 95% when data is naive-Bayes generated)",
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
