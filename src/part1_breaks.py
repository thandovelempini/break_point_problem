"""
The Break Point Problem - Part 1: Do breaks matter?

  1. Hold rates    - overall, by surface, by year, and per player
  2. Break-back    - after breaking, is a player more likely to be broken straight back?
  3. Break value   - how often does the first player to break go on to win the set?
  4. Momentum      - does winning the last point make winning the next one more likely?

"""

from pathlib import Path

import pandas as pd

from part1_analysis import (MIN_SERVICE_GAMES, break_back, break_value, hold_rates, momentum, pct,
                            player_table)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def load():
    matches = pd.read_csv(DATA / "slam_matches_wta.csv")
    games = pd.read_csv(DATA / "slam_games_wta.csv")
    points = pd.read_csv(DATA / "slam_points_wta.csv")

    points = points.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)
    games = games.merge(matches[["match_id", "year", "slam", "surface", "round"]], on="match_id", how="left")
    games = games.sort_values(["match_id", "set_no", "game_no"]).reset_index(drop=True)
    return matches, games, points


def main():
    print("Loading clean Slam data...")
    matches, games, points = load()

    print("1/4 Hold rates...")
    svc, hold_all, hold_surf, hold_year = hold_rates(games)

    print("2/4 Break-back...")
    bb, bb_surf, bb_players = break_back(svc)

    print("3/4 Break value...")
    bv_all, bv_surf, no_break_share, n_sets = break_value(games)

    print("4/4 Momentum...")
    mom, mom_states = momentum(points)

    players = player_table(svc, bb_players)

    hold_surf.to_csv(DATA / "part1_hold_by_surface.csv", index=False)
    hold_year.to_csv(DATA / "part1_hold_by_year.csv", index=False)
    bb.to_csv(DATA / "part1_break_back.csv", index=False)
    bb_surf.to_csv(DATA / "part1_break_back_by_surface.csv", index=False)
    bv_surf.to_csv(DATA / "part1_break_value_by_surface.csv", index=False)
    mom.to_csv(DATA / "part1_momentum.csv", index=False)
    mom_states.to_csv(DATA / "part1_momentum_by_score.csv", index=False)
    players.to_csv(DATA / "part1_players.csv", index=False)

    # Findings 
    h, v, m = hold_all.iloc[0], bv_all.iloc[0], mom.iloc[0]
    print("\n=== Part 1 findings ===")
    print(f"\n1. HOLD RATE: {pct(h['rate'])} of {h['n']:,.0f} service games "
          f"(95% CI {pct(h['ci_low'])}-{pct(h['ci_high'])})")
    for _, r in hold_surf.iterrows():
        print(f"   {r['surface']:<6} {pct(r['rate'])}  (n={r['n']:,})")

    b0 = bb.iloc[0]
    print(f"\n2. BREAK-BACK: after breaking, players were broken straight back {pct(b0['break_back_rate'])} "
          f"of the time (n={b0['breaks_followed_by_serve']:,.0f}). Compared with how often they're normally broken:")
    for _, b in bb.iterrows():
        print(f"   vs {b['baseline']:<10} {pct(b['expected_rate'])} expected -> {b['difference']:+.1%} "
              f"(95% CI {b['ci_low']:+.1%} to {b['ci_high']:+.1%})")

    print(f"\n3. BREAK VALUE: the first player to break won the set {pct(v['rate'])} of the time "
          f"(n={v['n']:,} sets; {pct(no_break_share)} of {n_sets:,} completed sets had no break)")
    for _, r in bv_surf.iterrows():
        print(f"   {r['surface']:<6} {pct(r['rate'])}")

    print(f"\n4. MOMENTUM: at the same score, servers who had just won the last point won the next")
    print(f"   {m['momentum_effect']:+.1%} more often than those who had just LOST it "
          f"(95% CI {m['ci_low']:+.1%} to {m['ci_high']:+.1%}, {m['point_pairs_compared']:,.0f} points)")
    for _, r in mom_states.iterrows():
        print(f"   {r['state']:<6} after win {pct(r['p_win_after_win'])}  after loss {pct(r['p_win_after_loss'])}"
              f"  -> {r['difference']:+.1%}  (n={r['n']:,.0f})")

    print(f"\nPlayer table: {len(players)} players with {MIN_SERVICE_GAMES}+ service games -> part1_players.csv")


if __name__ == "__main__":
    main()

"""
Findings:

1. Hold rate = 66%:
- 63% on clay
- 66% on hard courts
- 69% on grass

2. There's no "letdown" after breaking:
- Players get broken straight back 31% of the time
- Compared with their career rate that looks like a big dip (−3.3%), but most of it is because the player who 
breaks is usually the one playing better that day
- Against her serving in the same match, the effect nearly vanishes (−0.6%)

3. Breaks decide sets:
- The first player to break wins the set 79% of the time, on every surface

4. Momentum isn't real at the point level:
- Comparing servers at the same score (e.g. 30-30) who got there by winning vs losing the last point, 
the difference is 0.0% (95% CI −0.3% to +0.4%) across 320k points

"""