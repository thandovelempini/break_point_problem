"""
The Break Point Problem - Part 2: Why do serves get broken?

  1. Direction mix - where players serve, by serve number, court and pressure
  2. Predictability - each player's serve entropy, given the court
  3. Pressure - do players get more predictable on break points?
  4. Results - do more predictable servers win fewer points and hold less? 
  (overall entropy, body share, and wide-vs-T entropy)

"""

from pathlib import Path

import pandas as pd

from part2_analysis import (MAX_ENTROPY, MIN_FIRST_SERVES, direction_mix, entropy_vs_results, pct,
                            player_entropy, points_won_by_direction, pressure_test)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def main():
    print("Loading charted serves...")
    serves = pd.read_csv(DATA / "charting_serves_wta.csv", low_memory=False)
    part1_file = DATA / "part1_players.csv"
    part1_players = pd.read_csv(part1_file) if part1_file.exists() else None
    if part1_players is None:
        print("  (part1_players.csv not found - run part1_breaks.py first to link to Slam hold rates)")

    print("1/4 Direction mix...")
    mix = direction_mix(serves)
    by_direction = points_won_by_direction(serves)

    print("2/4 Player predictability...")
    players = player_entropy(serves)

    print("3/4 Pressure test (about a minute)...")
    pressure, pressure_cells, pressure_players = pressure_test(serves)
    wt = serves[serves["direction"] != "body"]
    pressure_wt, pressure_cells_wt, _ = pressure_test(wt, label="wide vs T")
    pressure = pd.concat([pressure, pressure_wt], ignore_index=True)
    pressure_cells = pd.concat([pressure_cells.assign(directions="all directions"),
                                pressure_cells_wt.assign(directions="wide vs T")], ignore_index=True)
    players = players.merge(pressure_players, on="player", how="left")

    print("4/4 Link to results...")
    links, players = entropy_vs_results(players, part1_players)

    mix.to_csv(DATA / "part2_direction_mix.csv", index=False)
    by_direction.to_csv(DATA / "part2_points_won_by_direction.csv", index=False)
    players.to_csv(DATA / "part2_players.csv", index=False)
    pressure.to_csv(DATA / "part2_pressure.csv", index=False)
    pressure_cells.to_csv(DATA / "part2_pressure_by_cell.csv", index=False)
    links.to_csv(DATA / "part2_entropy_vs_results.csv", index=False)

    # Findings 

    print("\n=== Part 2 findings ===")
    print(f"\n1. DIRECTION MIX ({len(serves):,} charted serves; wide / body / T)")
    for (number, court, bp), r in mix.set_index(["serve_number", "court", "is_break_point"]).iterrows():
        label = f"{'1st' if number == 1 else '2nd'} serve, {court:<5} {'break point' if bp else 'other':<11}"
        print(f"   {label} {pct(r['wide'])} / {pct(r['body'])} / {pct(r['T'])}   entropy {r['entropy']:.3f}  (n={r['n']:,.0f})")
    print("   Points won when the serve lands in:")
    for _, r in by_direction.iterrows():
        print(f"   {'1st' if r['serve_number'] == 1 else '2nd'} serve {r['direction']:<5} {pct(r['points_won'])}  (n={r['n']:,})")

    print(f"\n2. PREDICTABILITY: {len(players)} players with {MIN_FIRST_SERVES}+ charted first serves. "
          f"First-serve entropy given court (max {MAX_ENTROPY:.2f} bits):")
    print(f"   median {players['first_entropy'].median():.3f}, "
          f"range {players['first_entropy'].min():.3f} to {players['first_entropy'].max():.3f}")
    show = ["player", "first_serves", "first_entropy", "first_wide", "first_body", "first_T"]
    print("   Most predictable:")
    print(players.nsmallest(5, "first_entropy")[show].to_string(index=False, float_format="%.3f"))
    print("   Least predictable:")
    print(players.nlargest(5, "first_entropy")[show].to_string(index=False, float_format="%.3f"))

    print("\n3. PRESSURE: break-point entropy vs same-size random samples of each player's own serves")
    print("   (same player, same court, same serve; negative = more predictable on break points)")
    for _, r in pressure.iterrows():
        print(f"   {r['directions']:<14} {r['serve']:<6} {r['effect_bits']:+.3f} bits  (no-effect range {r['no_effect_low']:+.3f} to "
              f"{r['no_effect_high']:+.3f}, p = {r['p_value_two_sided']:.3f}; {r['players']} players, "
              f"{r['bp_serves']:,} break-point serves; {pct(r['share_more_predictable'])} of cells more predictable)")

    print("\n4. RESULTS: Spearman correlation of each first-serve measure with...")
    names = {"first_entropy": "overall entropy", "first_body": "body share", "first_wide_t_entropy": "wide-vs-T entropy"}
    for _, r in links.iterrows():
        print(f"   {names[r['measure']]:<18} vs {r['outcome']:<17} r = {r['spearman_r']:+.2f}  (95% CI {r['ci_low']:+.2f} to {r['ci_high']:+.2f}, "
              f"{r['players']} players)")


if __name__ == "__main__":
    main()

"""
Findings:

1. The body serve is the weak spot:
- First serves that land in win 66% of points when aimed wide or down the T, but only 57% to the body

2. 'Predictable' servers actually win more, but entropy measure is misleading:
- Overall entropy is mostly measuring how often a player goes to the body
- The low-entropy players (Safarova, Henin, Davenport...) rarely serve to the body at all (6–11% of first serves), 
which is why they win more

3. Pressure doesn't really change anything:
- On break points, players become slightly more predictable (−0.015 bits, about 1% of the scale).
- That's statistically detectable across 41k break-point serves but tiny in practice, 
and only about half of players show it
- Conclusion is that players serve almost the same way under pressure

4. The true predictability effect is wide vs T:
- Leave out body serves and the relationship flips --> 
players who mix wide and T more evenly win more serve points (r = +0.21) and hold serve more often (r = +0.19)
- Thus, "being hard to read helps", and it only shows up once you separate the two effects

"""