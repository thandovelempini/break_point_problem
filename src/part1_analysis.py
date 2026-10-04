"""
Part 1 analyses: the four questions, plus shared helpers
"""

import numpy as np
import pandas as pd

MIN_SERVICE_GAMES = 100   # players need this many to get their own row in the player table
N_BOOTSTRAP = 500         # resamples for the momentum confidence interval
SEED = 42



# Helpers

def wilson(successes, n, z=1.96):
    """95% Wilson interval for a proportion. Works on scalars or arrays."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    p = np.divide(successes, n, out=np.full_like(n, np.nan), where=n > 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(df, by, success_col):
    t = df.groupby(by)[success_col].agg(n="size", successes="sum").reset_index()
    t["rate"] = t["successes"] / t["n"]
    t["ci_low"], t["ci_high"] = wilson(t["successes"], t["n"])
    return t


def pct(x):
    return f"{x:.1%}"



# 1. Hold rates

def hold_rates(games):
    svc = games[~games["is_tiebreak"]].copy()
    svc["held"] = svc["held"].astype(bool)
    overall = rate_table(svc.assign(all="All"), "all", "held")
    by_surface = rate_table(svc, "surface", "held")
    by_year = rate_table(svc, "year", "held")
    return svc, overall, by_surface, by_year



# 2. Break-back

def break_back(svc):
    """After a player breaks, her next game (same set) is on her own serve.
    Break-back = she loses that game.

    Two baselines, each built from her service games that did not come straight
    after a break:
      - career --> her break-against rate across all Slam matches in the data
      - same match --> her break-against rate in the rest of that match, which also
        accounts for how well she was playing that day and how good the opponent was
    """
    g = svc.copy()
    nxt = g.groupby(["match_id", "set_no"]).shift(-1)
    g["next_server"] = nxt["server"]
    g["next_held"] = nxt["held"]

    broke = g[(~g["held"]) & (g["next_server"] == g["returner"])].copy()
    broke["player"] = broke["returner"]
    broke["broken_back"] = ~broke["next_held"].astype(bool)

    # Service games that did not come straight after the server's own break
    after = broke[["match_id", "set_no", "game_no"]].assign(game_no=lambda d: d["game_no"] + 1, after_break=True)
    normal = svc.merge(after, on=["match_id", "set_no", "game_no"], how="left")
    normal = normal[normal["after_break"].isna()]

    career = normal.groupby("server")["held"].agg(["size", "sum"])
    career = (1 - career["sum"] / career["size"]).rename("career_broken_rate")
    in_match = normal.groupby(["match_id", "server"])["held"].agg(["size", "sum"])
    in_match = (1 - in_match["sum"] / in_match["size"]).rename("match_broken_rate")

    broke = broke.join(career, on="player").join(in_match, on=["match_id", "player"])

    rows = []
    for label, col in [("career", "career_broken_rate"), ("same match", "match_broken_rate")]:
        b = broke[broke[col].notna()]
        n, observed, expected = len(b), b["broken_back"].mean(), b[col].mean()
        se = np.sqrt((b[col] * (1 - b[col])).sum()) / n
        rows.append({
            "baseline": label, "breaks_followed_by_serve": n,
            "break_back_rate": observed, "expected_rate": expected,
            "difference": observed - expected,
            "ci_low": observed - expected - 1.96 * se, "ci_high": observed - expected + 1.96 * se,
        })
    summary = pd.DataFrame(rows)

    by_surface = broke.groupby("surface").agg(
        n=("broken_back", "size"), break_back_rate=("broken_back", "mean"),
        expected_career=("career_broken_rate", "mean"), expected_same_match=("match_broken_rate", "mean")).reset_index()

    per_player = broke.groupby("player").agg(
        breaks=("broken_back", "size"), broken_back=("broken_back", "sum"),
        expected_rate=("career_broken_rate", "first")).reset_index()
    per_player["break_back_rate"] = per_player["broken_back"] / per_player["breaks"]
    return summary, by_surface, per_player


# 3. Break value: does the first player to break win the set?

def break_value(games):
    g = games.copy()
    # Completed sets only --> winner reached 6+ games with a 2-game lead, or won a tiebreak
    wins = g.groupby(["match_id", "set_no", "game_winner"]).size().unstack(fill_value=0)
    sets = g.groupby(["match_id", "set_no"]).agg(
        set_winner=("game_winner", "last"), had_tiebreak=("is_tiebreak", "any"),
        surface=("surface", "first")).reset_index()
    top = wins.max(axis=1).rename("w_games")
    second = wins.apply(lambda r: r.nlargest(2).iloc[-1] if (r > 0).sum() > 1 else 0, axis=1).rename("l_games")
    sets = sets.join(top, on=["match_id", "set_no"]).join(second, on=["match_id", "set_no"])
    sets = sets[(sets["w_games"] >= 6) & ((sets["w_games"] - sets["l_games"] >= 2) | sets["had_tiebreak"])]

    breaks = g[(~g["is_tiebreak"]) & (g["held"] == False)]  # noqa: E712
    first = breaks.groupby(["match_id", "set_no"]).head(1)[["match_id", "set_no", "game_no", "returner"]]
    first = first.rename(columns={"returner": "first_breaker", "game_no": "first_break_game"})

    sets = sets.merge(first, on=["match_id", "set_no"], how="left")
    sets["any_break"] = sets["first_breaker"].notna()
    sets["first_breaker_won"] = sets["first_breaker"] == sets["set_winner"]

    broken_sets = sets[sets["any_break"]]
    overall = rate_table(broken_sets.assign(all="All"), "all", "first_breaker_won")
    by_surface = rate_table(broken_sets, "surface", "first_breaker_won")
    no_break_share = 1 - sets["any_break"].mean()
    return overall, by_surface, no_break_share, len(sets)


# 4. Momentum: same score, different path

TWO_PATH_STATES = ["15-15", "30-15", "15-30", "30-30", "40-15", "15-40", "40-30", "30-40", "40-40"]


def momentum(points):
    """Does winning the last point make the server more likely to win the next?

    A simple "after a win vs after a loss" comparison is biased: strong servers
    string wins together just by being strong. So compare servers at the same
    score who got there by different paths
    """

    p = points[~points["is_tiebreak"]].copy()
    p = p.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

    gk = [p["match_id"], p["set_no"], p["game_no"]]
    prev_won = p.groupby(gk)["server_won_point"].shift(1)

    won = p["server_won_point"].astype(int)
    srv_pts = won.groupby(gk).cumsum() - won          
    ret_pts = (1 - won).groupby(gk).cumsum() - (1 - won)
    names = ["0", "15", "30", "40"]

    def label(s, r):
        if s >= 3 and r >= 3:
            return "40-40" if s == r else ("AD-40" if s > r else "40-AD")
        return f"{names[min(s, 3)]}-{names[min(r, 3)]}"

    p["state"] = [label(s, r) for s, r in zip(srv_pts.to_numpy(), ret_pts.to_numpy())]
    p["prev_won"] = prev_won
    pairs = p[p["prev_won"].notna() & p["state"].isin(TWO_PATH_STATES)].copy()
    pairs["prev_won"] = pairs["prev_won"].astype(bool)

    # Per state: P(win next | arrived by winning) - P(win next | arrived by losing)
    by_state = pairs.groupby(["state", "prev_won"])["server_won_point"].agg(["size", "mean"]).unstack()
    by_state.columns = [f"{a}_{'after_win' if b else 'after_loss'}" for a, b in by_state.columns]
    by_state = by_state.rename(columns={"size_after_win": "n_after_win", "size_after_loss": "n_after_loss",
                                        "mean_after_win": "p_win_after_win", "mean_after_loss": "p_win_after_loss"})
    by_state["difference"] = by_state["p_win_after_win"] - by_state["p_win_after_loss"]
    by_state["n"] = by_state["n_after_win"] + by_state["n_after_loss"]
    by_state = by_state.reindex(TWO_PATH_STATES).reset_index()

    def weighted_effect(counts):
        # counts: array [state, path(0=loss,1=win), (n, wins)] --> weighted mean difference
        n, w = counts[..., 0], counts[..., 1]
        rate = np.divide(w, n, out=np.zeros_like(w, dtype=float), where=n > 0)
        diff = rate[:, 1] - rate[:, 0]
        weight = n.sum(axis=1)
        return (diff * weight).sum() / weight.sum()

    # Match x state x path counts for the bootstrap
    pairs["s_idx"] = pairs["state"].map({s: i for i, s in enumerate(TWO_PATH_STATES)})
    m_idx, match_ids = pd.factorize(pairs["match_id"])
    arr = np.zeros((len(match_ids), len(TWO_PATH_STATES), 2, 2))
    np.add.at(arr, (m_idx, pairs["s_idx"].to_numpy(), pairs["prev_won"].to_numpy().astype(int), 0), 1)
    np.add.at(arr, (m_idx, pairs["s_idx"].to_numpy(), pairs["prev_won"].to_numpy().astype(int), 1),
              pairs["server_won_point"].to_numpy().astype(float))

    observed = weighted_effect(arr.sum(axis=0))
    rng = np.random.default_rng(SEED)
    boot = np.array([
        weighted_effect(np.tensordot(rng.poisson(1.0, len(match_ids)), arr, axes=(0, 0)))
        for _ in range(N_BOOTSTRAP)
    ])
    se = boot.std(ddof=1)
    z = observed / se

    summary = pd.DataFrame([{
        "point_pairs_compared": len(pairs),
        "momentum_effect": observed,
        "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
        "z_score": z,
        "bootstrap_resamples": N_BOOTSTRAP,
    }])
    return summary, by_state


# Player table (for the dashboard)

def player_table(svc, bb_players):
    t = rate_table(svc, "server", "held").rename(columns={
        "server": "player", "n": "service_games", "successes": "holds", "rate": "hold_rate",
        "ci_low": "hold_ci_low", "ci_high": "hold_ci_high"})
    surf = svc.pivot_table(index="server", columns="surface", values="held", aggfunc="mean")
    surf.columns = [f"hold_rate_{c.lower()}" for c in surf.columns]
    t = t.join(surf, on="player")
    t = t.merge(bb_players[["player", "breaks", "broken_back", "break_back_rate"]], on="player", how="left")
    t = t[t["service_games"] >= MIN_SERVICE_GAMES]
    return t.sort_values("service_games", ascending=False).reset_index(drop=True)

