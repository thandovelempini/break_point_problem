"""
The Break Point Problem - export for Tableau
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "tableau"


def key(name):
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", name.lower().replace("-", " "))).strip()


def wilson(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


# 1. Players

def build_players(serves):
    p1 = pd.read_csv(DATA / "part1_players.csv")
    p2 = pd.read_csv(DATA / "part2_players.csv").drop(columns=["hold_rate", "service_games"], errors="ignore")
    lookup = pd.read_csv(DATA / "player_lookup.csv", dtype=str)[["name", "player_id", "hand", "dob", "ioc"]]

    years = serves.groupby("server")["year"].agg(first_year_charted="min", last_year_charted="max")
    p2 = p2.join(years, on="player")

    p1["key"], p2["key"] = p1["player"].map(key), p2["player"].map(key)
    t = p1.merge(p2.drop(columns="player"), on="key", how="outer")
    t["player"] = t["player"].fillna(t["key"].map(p2.set_index("key")["player"]))
    t = t.drop(columns="key")

    t["in_slam_data"] = t["service_games"].notna()
    t["in_charting_data"] = t["first_serves"].notna()
    t = t.merge(lookup.drop_duplicates("name").rename(columns={"name": "player", "ioc": "country"}),
                on="player", how="left")
    t = t.rename(columns={"first_T": "first_t"})
    front = ["player", "player_id", "country", "hand", "dob", "in_slam_data", "in_charting_data"]
    return t[front + [c for c in t.columns if c not in front]].sort_values("player").reset_index(drop=True)



# 2. Serve directions 

def build_serve_directions(serves):
    s = serves.assign(
        serve=serves["serve_number"].map({1: "First", 2: "Second"}),
        situation=np.where(serves["is_break_point"], "Break point", "Other points"),
        landed=~serves["fault"],
        won_when_landed=~serves["fault"] & serves["server_won_point"],
    )
    t = (s.groupby(["server", "year", "surface", "serve", "court", "situation", "direction"], dropna=False)
         .agg(serves=("direction", "size"), landed=("landed", "sum"), won_when_landed=("won_when_landed", "sum"))
         .reset_index().rename(columns={"server": "player"}))
    t["court"] = t["court"].str.capitalize()
    return t



# 3. Player card (bio details)

WTA_DIR = ROOT / "data" / "raw" / "tennis-sackmann-archive" / "wta"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COUNTRIES = {
    "AND": ("Andorra", "ad"), "ARG": ("Argentina", "ar"), "ARM": ("Armenia", "am"), "AUS": ("Australia", "au"),
    "AUT": ("Austria", "at"), "BEL": ("Belgium", "be"), "BLR": ("Belarus", "by"), "BRA": ("Brazil", "br"),
    "BUL": ("Bulgaria", "bg"), "CAN": ("Canada", "ca"), "CHN": ("China", "cn"), "COL": ("Colombia", "co"),
    "CRO": ("Croatia", "hr"), "CZE": ("Czechia", "cz"), "DEN": ("Denmark", "dk"), "EGY": ("Egypt", "eg"),
    "ESP": ("Spain", "es"), "EST": ("Estonia", "ee"), "FRA": ("France", "fr"), "GBR": ("Great Britain", "gb"),
    "GEO": ("Georgia", "ge"), "GER": ("Germany", "de"), "GRE": ("Greece", "gr"), "HUN": ("Hungary", "hu"),
    "INA": ("Indonesia", "id"), "ISR": ("Israel", "il"), "ITA": ("Italy", "it"), "JPN": ("Japan", "jp"),
    "KAZ": ("Kazakhstan", "kz"), "LAT": ("Latvia", "lv"), "LUX": ("Luxembourg", "lu"), "MEX": ("Mexico", "mx"),
    "MNE": ("Montenegro", "me"), "NED": ("Netherlands", "nl"), "NZL": ("New Zealand", "nz"),
    "PAR": ("Paraguay", "py"), "PHI": ("Philippines", "ph"), "POL": ("Poland", "pl"), "PUR": ("Puerto Rico", "pr"),
    "ROU": ("Romania", "ro"), "RSA": ("South Africa", "za"), "RUS": ("Russia", "ru"), "SLO": ("Slovenia", "si"),
    "SRB": ("Serbia", "rs"), "SUI": ("Switzerland", "ch"), "SVK": ("Slovakia", "sk"), "SWE": ("Sweden", "se"),
    "THA": ("Thailand", "th"), "TPE": ("Chinese Taipei", "tw"), "TUN": ("Tunisia", "tn"), "TUR": ("Türkiye", "tr"),
    "UKR": ("Ukraine", "ua"), "USA": ("United States", "us"), "UZB": ("Uzbekistan", "uz"),
}


def nice_date(yyyymmdd, day=True):
    s = str(yyyymmdd)
    if len(s) < 8 or not s[:8].isdigit():
        return None
    month = MONTHS[int(s[4:6]) - 1]
    return f"{int(s[6:8])} {month} {s[:4]}" if day else f"{month} {s[:4]}"


def add_bio(players):
    wta = pd.read_csv(WTA_DIR / "wta_players.csv", dtype=str).drop_duplicates("player_id")
    names = wta.set_index("player_id")[["name_first", "name_last", "height"]]
    t = players.join(names, on="player_id")
    split = t["player"].str.split(" ", n=1)
    t["first_name"] = t["name_first"].fillna(split.str[0])
    t["last_name"] = t["name_last"].fillna(split.str[1])
    t["country_name"] = t["country"].map(lambda c: COUNTRIES.get(c, (c, None))[0])
    t["flag_url"] = t["country"].map(
        lambda c: f"https://flagcdn.com/w80/{COUNTRIES[c][1]}.png" if c in COUNTRIES else None)
    return t.drop(columns=["name_first", "name_last"]).rename(columns={"height": "height_cm"})


def build_player_card(players):
    ids = set(players["player_id"].dropna())
    ranks = pd.concat(
        [pd.read_csv(f, dtype={"player": str, "ranking_date": str}, usecols=["ranking_date", "rank", "player"])
         for f in sorted(WTA_DIR.glob("wta_rankings_*.csv"))], ignore_index=True)
    ranks = ranks[ranks["player"].isin(ids)].sort_values(["player", "rank", "ranking_date"])
    best = ranks.drop_duplicates("player").set_index("player")            # first date at her best rank
    latest_date = ranks["ranking_date"].max()
    latest = ranks[ranks["ranking_date"] == latest_date].set_index("player")["rank"]
    today = pd.Timestamp.today()

    rows = []
    for _, p in players.iterrows():
        pid = p["player_id"]

        def add(order, field, value):
            if value is not None and value == value:  # skip blanks / NaN
                rows.append({"player": p["player"], "sort_order": order, "field": field, "value": value})

        born = pd.to_datetime(str(p["dob"])[:8], format="%Y%m%d", errors="coerce") if pd.notna(p["dob"]) else pd.NaT
        if pd.notna(born):
            age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            add(1, "Born", f"{nice_date(p['dob'])} (age {age})")
        if pd.notna(p.get("height_cm")):
            add(2, "Height", f"{int(float(p['height_cm'])) / 100:.2f} m")
        add(3, "Plays", {"R": "Right-handed", "L": "Left-handed"}.get(p["hand"]))
        if pid in best.index:
            add(4, "Career-high ranking", f"#{int(best.loc[pid, 'rank'])} ({nice_date(best.loc[pid, 'ranking_date'], day=False)})")
        if pid in latest.index:
            add(5, "Latest ranking", f"#{int(latest.loc[pid])} ({nice_date(latest_date)})")
    return pd.DataFrame(rows, columns=["player", "sort_order", "field", "value"])



# 4. Findings 

def build_findings():
    rows = []

    def add(section, finding, category, value, n=None, low=None, high=None, unit="share",
            interval="95% CI", order=None):
        rows.append({"section": section, "finding": finding, "category": str(category), "value": value,
                     "interval_low": low, "interval_high": high,
                     "interval_type": interval if low is not None else None,
                     "n": n, "unit": unit, "sort_order": order if order is not None else len(rows)})

    # Part 1 
    for fname, finding in [("part1_hold_by_surface.csv", "Hold rate"),
                           ("part1_break_value_by_surface.csv", "First to break wins the set")]:
        t = pd.read_csv(DATA / fname)
        n, s = t["n"].sum(), t["successes"].sum()
        lo, hi = wilson(s, n)
        add("Part 1: Do breaks matter?", finding, "All surfaces", s / n, n, lo, hi)
        for _, r in t.iterrows():
            add("Part 1: Do breaks matter?", finding, r["surface"], r["rate"], r["n"], r["ci_low"], r["ci_high"])

    for _, r in pd.read_csv(DATA / "part1_hold_by_year.csv").iterrows():
        add("Part 1: Do breaks matter?", "Hold rate by year", int(r["year"]), r["rate"], r["n"], r["ci_low"], r["ci_high"])

    for _, r in pd.read_csv(DATA / "part1_break_back.csv").iterrows():
        if r["baseline"] == "career":
            add("Part 1: Do breaks matter?", "Break-back rate", "Observed", r["break_back_rate"],
                r["breaks_followed_by_serve"])
        add("Part 1: Do breaks matter?", "Break-back rate", f"Expected ({r['baseline']})", r["expected_rate"],
            r["breaks_followed_by_serve"])
        add("Part 1: Do breaks matter?", "Break-back vs expected", r["baseline"].capitalize(), r["difference"],
            r["breaks_followed_by_serve"], r["ci_low"], r["ci_high"])

    m = pd.read_csv(DATA / "part1_momentum.csv").iloc[0]
    add("Part 1: Do breaks matter?", "Momentum effect", "All scores", m["momentum_effect"],
        m["point_pairs_compared"], m["ci_low"], m["ci_high"])
    for _, r in pd.read_csv(DATA / "part1_momentum_by_score.csv").iterrows():
        add("Part 1: Do breaks matter?", "Momentum effect by score", r["state"], r["difference"], r["n"])

    # Part 2 
    for _, r in pd.read_csv(DATA / "part2_points_won_by_direction.csv").iterrows():
        serve = "First" if r["serve_number"] == 1 else "Second"
        add("Part 2: Why do serves get broken?", "Points won when the serve lands in",
            f"{serve} serve, {r['direction']}", r["points_won"], r["n"])

    for _, r in pd.read_csv(DATA / "part2_pressure.csv").iterrows():
        add("Part 2: Why do serves get broken?", "Break-point change in serve entropy",
            f"{r['directions'].capitalize()}, {r['serve']} serves", r["effect_bits"], r["bp_serves"],
            r["no_effect_low"], r["no_effect_high"], unit="bits", interval="no-effect range")

    labels = {"first_entropy": "Overall entropy", "first_body": "Body share",
              "first_wide_t_entropy": "Wide-vs-T entropy",
              "serve_points_won": "points won on serve", "hold_rate": "Slam hold rate"}
    for _, r in pd.read_csv(DATA / "part2_entropy_vs_results.csv").iterrows():
        add("Part 2: Why do serves get broken?", "Correlation with results",
            f"{labels[r['measure']]} vs {labels[r['outcome']]}", r["spearman_r"], r["players"],
            r["ci_low"], r["ci_high"], unit="r")

    return pd.DataFrame(rows)



# 5. Score grid (heat map)

SCORE_NAMES = ["0", "15", "30", "40"]
SCORE_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def score_label(srv, ret):
    # points won before this point -> ("30", "40"); every deuce counts as 40-40
    if srv >= 3 and ret >= 3:
        return ("40", "40") if srv == ret else (("AD", "40") if srv > ret else ("40", "AD"))
    return SCORE_NAMES[min(srv, 3)], SCORE_NAMES[min(ret, 3)]


def build_score_grid():
    # One row per score per surface: how often the server wins the next point,
    # and how often she holds the game from there. Tiebreaks left out.
    pts = pd.read_csv(DATA / "slam_points_wta.csv",
                      usecols=["match_id", "set_no", "game_no", "point_no", "server_won_point", "is_tiebreak"])
    pts = pts[~pts["is_tiebreak"]].sort_values(["match_id", "set_no", "game_no", "point_no"])
    games = pd.read_csv(DATA / "slam_games_wta.csv", usecols=["match_id", "set_no", "game_no", "held"])
    games = games.dropna(subset=["held"])                       # tiebreak games have no "held"
    games["held"] = games["held"].astype(str).eq("True")
    surface = pd.read_csv(DATA / "slam_matches_wta.csv", usecols=["match_id", "surface"])

    won = pts["server_won_point"].astype(int)
    keys = [pts["match_id"], pts["set_no"], pts["game_no"]]
    srv = won.groupby(keys).cumsum() - won                 # server points won BEFORE this point
    ret = (1 - won).groupby(keys).cumsum() - (1 - won)     # returner points won before this point
    labels = [score_label(a, b) for a, b in zip(srv.to_numpy(), ret.to_numpy())]
    pts["server_score"] = [a for a, _ in labels]
    pts["returner_score"] = [b for _, b in labels]
    pts = (pts.merge(games, on=["match_id", "set_no", "game_no"], how="inner")
              .merge(surface, on="match_id", how="left"))
    pts = pd.concat([pts, pts.assign(surface="All surfaces")], ignore_index=True)

    cell = ["surface", "server_score", "returner_score"]
    t = pts.groupby(cell).agg(points=("server_won_point", "size"),
                              points_won=("server_won_point", "sum")).reset_index()
    reached = pts.drop_duplicates(cell + ["match_id", "set_no", "game_no"])
    hold = reached.groupby(cell)["held"].agg(games_reached="size", games_held="sum").reset_index()
    t = t.merge(hold, on=cell)

    t["point_win_rate"] = t["points_won"] / t["points"]
    t["point_win_ci_low"], t["point_win_ci_high"] = wilson(t["points_won"], t["points"])
    t["hold_rate"] = t["games_held"] / t["games_reached"]
    t["hold_ci_low"], t["hold_ci_high"] = wilson(t["games_held"], t["games_reached"])
    t["server_order"] = t["server_score"].map(SCORE_ORDER)
    t["returner_order"] = t["returner_score"].map(SCORE_ORDER)
    t["score"] = t["server_score"] + "-" + t["returner_score"]
    t["is_break_point"] = (((t["returner_score"] == "40") & (t["server_order"] < 3))
                           | (t["returner_score"] == "AD"))
    front = ["surface", "score", "server_score", "returner_score", "server_order", "returner_order", "is_break_point"]
    return t[front + [c for c in t.columns if c not in front]].sort_values(
        ["surface", "server_order", "returner_order"]).reset_index(drop=True)



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading...")
    serves = pd.read_csv(DATA / "charting_serves_wta.csv", low_memory=False)
    serves["year"] = pd.to_numeric(serves["match_id"].str[:4], errors="coerce").astype("Int64")
    serves["surface"] = serves["surface"].where(serves["surface"].isin(["Hard", "Clay", "Grass", "Carpet"]), "Unknown")

    players = add_bio(build_players(serves))
    card = build_player_card(players)
    directions = build_serve_directions(serves)
    findings = build_findings()
    grid = build_score_grid()

    players.to_csv(OUT / "players.csv", index=False)
    directions.to_csv(OUT / "serve_directions.csv", index=False)
    findings.to_csv(OUT / "findings.csv", index=False)
    card.to_csv(OUT / "player_card.csv", index=False)
    grid.to_csv(OUT / "score_grid.csv", index=False)

    print(f"players.csv           {len(players):>7,} rows  ({players['in_slam_data'].sum()} in Slam data, "
          f"{players['in_charting_data'].sum()} in charting data, "
          f"{(players['in_slam_data'] & players['in_charting_data']).sum()} in both)")
    print(f"serve_directions.csv  {len(directions):>7,} rows  ({directions['serves'].sum():,} serves)")
    print(f"findings.csv          {len(findings):>7,} rows")
    print(f"player_card.csv       {len(card):>7,} rows  (bio lines for {card['player'].nunique()} players)")
    print(f"score_grid.csv        {len(grid):>7,} rows  (18 scores x {grid['surface'].nunique()} surface options, for the heat map)")
    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()