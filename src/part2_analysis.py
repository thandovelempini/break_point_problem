"""
Part 2 analyses: serve predictability and pressure.

Predictability is measured with Shannon entropy of serve direction (wide / body / T):
  0 bits    = always the same spot
  1.58 bits = a perfectly even three-way split (the maximum)

"""

import re
import unicodedata

import numpy as np
import pandas as pd

MIN_FIRST_SERVES = 300      # players need this many charted first serves for their own row
MIN_BP_SERVES = 20          # per player or court or serve number, to test pressure
N_DRAWS = 500               # random draws for the pressure test
SEED = 42
DIRECTIONS = ["wide", "body", "T"]
MAX_ENTROPY = np.log2(3)


# Helpers

def entropy(counts):
    counts = np.asarray(counts, float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=-1)


def direction_counts(df):
    return df["direction"].value_counts().reindex(DIRECTIONS, fill_value=0).to_numpy()


def conditional_entropy(df):
    h, n = 0.0, 0
    for _, court in df.groupby("court"):
        h += entropy(direction_counts(court)) * len(court)
        n += len(court)
    return h / n if n else np.nan


def normalise_name(name):
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z ]", " ", name.lower().replace("-", " "))
    return re.sub(r"\s+", " ", name).strip()


def pct(x):
    return f"{x:.1%}"


# 1. Direction mix

def direction_mix(serves):
    mix = (serves.groupby(["serve_number", "court", "is_break_point"])["direction"]
           .value_counts(normalize=True).unstack().reindex(columns=DIRECTIONS).reset_index())
    mix["n"] = serves.groupby(["serve_number", "court", "is_break_point"]).size().to_numpy()
    mix["entropy"] = entropy(mix[DIRECTIONS].to_numpy())
    return mix


def points_won_by_direction(serves):
    landed = serves[~serves["fault"]]
    t = landed.groupby(["serve_number", "direction"])["server_won_point"].agg(n="size", points_won="mean")
    return t.reindex(DIRECTIONS, level="direction").reset_index()

# 2. Player predictability

def player_entropy(serves):
    rows = []
    for player, df in serves.groupby("server"):
        first, second = df[df["serve_number"] == 1], df[df["serve_number"] == 2]
        if len(first) < MIN_FIRST_SERVES:
            continue
        mix = direction_counts(first) / len(first)

        # Points won on serve: one row per point (the first-serve row carries the result)
        rows.append({
            "player": player,
            "matches_charted": df["match_id"].nunique(),
            "first_serves": len(first),
            "second_serves": len(second),
            "first_entropy": conditional_entropy(first),
            "first_wide_t_entropy": conditional_entropy(first[first["direction"] != "body"]),
            "second_entropy": conditional_entropy(second) if len(second) >= 100 else np.nan,
            "first_wide": mix[0], "first_body": mix[1], "first_T": mix[2],
            "serve_points_won": first["server_won_point"].mean(),
        })
    t = pd.DataFrame(rows)
    t["first_predictability"] = 1 - t["first_entropy"] / MAX_ENTROPY   # 0 = unreadable, 1 = one spot
    return t.sort_values("first_entropy").reset_index(drop=True)


# 3. Pressure: do players get more predictable on break points?

def pressure_test(serves, label="all directions"):
    # Compare each player's break-point entropy with random samples of her own serves

    rng = np.random.default_rng(SEED)
    code = serves["direction"].map({d: i for i, d in enumerate(DIRECTIONS)}).to_numpy()
    serves = serves.assign(code=code)

    cells, null_rows = [], []
    for (player, court, number), df in serves.groupby(["server", "court", "serve_number"]):
        n_bp = int(df["is_break_point"].sum())
        if n_bp < MIN_BP_SERVES or len(df) - n_bp < 5 * n_bp:
            continue
        h_bp = entropy(np.bincount(df.loc[df["is_break_point"], "code"], minlength=3))
        all_codes = df["code"].to_numpy()
        # N_DRAWS random subsets of size n_bp, drawn without replacement.
        idx = np.argsort(rng.random((N_DRAWS, len(all_codes))), axis=1)[:, :n_bp]
        draw_counts = np.stack([(all_codes[idx] == k).sum(axis=1) for k in range(3)], axis=1)
        h_draws = entropy(draw_counts)
        cells.append({"player": player, "court": court, "serve_number": number,
                      "bp_serves": n_bp, "bp_entropy": h_bp, "expected_entropy": h_draws.mean(),
                      "effect": h_bp - h_draws.mean()})
        null_rows.append(h_draws - h_draws.mean())

    cells = pd.DataFrame(cells)
    null = np.stack(null_rows)                          # cells x draws
    w = cells["bp_serves"].to_numpy()

    def summarise(mask, which):
        ww = w[mask]
        obs = np.average(cells.loc[mask, "effect"], weights=ww)
        null_mean = (null[mask] * ww[:, None]).sum(axis=0) / ww.sum()
        return {"directions": label, "serve": which, "cells": int(mask.sum()), "players": cells.loc[mask, "player"].nunique(),
                "bp_serves": int(ww.sum()), "effect_bits": obs,
                "no_effect_low": np.percentile(null_mean, 2.5), "no_effect_high": np.percentile(null_mean, 97.5),
                "p_value_two_sided": (np.sum(np.abs(null_mean) >= abs(obs)) + 1) / (N_DRAWS + 1),
                "share_more_predictable": (cells.loc[mask, "effect"] < 0).mean()}

    summary = pd.DataFrame([
        summarise(np.ones(len(cells), bool), "all"),
        summarise((cells["serve_number"] == 1).to_numpy(), "first"),
        summarise((cells["serve_number"] == 2).to_numpy(), "second"),
    ])
    per_player = cells.groupby("player").apply(
        lambda d: pd.Series({"bp_serves": d["bp_serves"].sum(),
                             "pressure_effect": np.average(d["effect"], weights=d["bp_serves"])}),
        include_groups=False).reset_index()
    return summary, cells, per_player



# 4. Link to results: do predictable servers win less?

def entropy_vs_results(players, part1_players=None):
    t = players.copy()
    if part1_players is not None:
        p1 = part1_players[["player", "hold_rate", "service_games"]].assign(key=lambda d: d["player"].map(normalise_name))
        t = t.assign(key=t["player"].map(normalise_name)).merge(p1.drop(columns="player"), on="key", how="left")
        t = t.drop(columns="key")

    rng = np.random.default_rng(SEED)
    rows = []
    for measure in ["first_entropy", "first_body", "first_wide_t_entropy"]:
        for outcome in ["serve_points_won", "hold_rate"]:
            if outcome not in t:
                continue
            d = t[[measure, outcome]].dropna()
            r = d[measure].corr(d[outcome], method="spearman")
            boot = [d.iloc[rng.integers(0, len(d), len(d))].corr(method="spearman").iloc[0, 1] for _ in range(1000)]
            rows.append({"measure": measure, "outcome": outcome, "players": len(d), "spearman_r": r,
                         "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows), t