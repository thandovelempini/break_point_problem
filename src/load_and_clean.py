import re
import unicodedata
from pathlib import Path
 
import numpy as np
import pandas as pd
 
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
 
ARCHIVE_DIR = RAW / "tennis-sackmann-archive"               # from Hugging Face
SLAM_DIR = ARCHIVE_DIR / "slam_pointbypoint"                # *-matches.csv, *-points.csv
WTA_DIR = ARCHIVE_DIR / "wta"                               # wta_players.csv, wta_matches_YYYY.csv
ATP_DIR = ARCHIVE_DIR / "atp"                               # only needed if INCLUDE_ATP_BASELINE
CHARTING_DIR = RAW / "tennis_MatchChartingProject"          # charting-w-*.csv, from GitHub
 
 
def check_folders() -> None:
    """Stop early with a clear message if a data folder is missing."""
    missing = [d for d in (SLAM_DIR, WTA_DIR, CHARTING_DIR) if not d.exists()]
    if missing:
        raise FileNotFoundError("Missing data folder(s):\n  " + "\n  ".join(map(str, missing))
                                + "\nSee README.md for which files to download and where to put them.")
 
INCLUDE_ATP_BASELINE = False
 
SLAM_SURFACE = {"ausopen": "Hard", "usopen": "Hard", "frenchopen": "Clay", "wimbledon": "Grass"}
STANDARD_SCORES = {"0", "15", "30", "40", "AD"}
SERVE_DIRECTION = {"4": "wide", "5": "body", "6": "T"}
ROUND_FROM_DIGIT = {"1": "R128", "2": "R64", "3": "R32", "4": "R16", "5": "QF", "6": "SF", "7": "F"}
 
 
def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z ]", " ", name.lower().replace("-", " "))
    return re.sub(r"\s+", " ", name).strip()
 
 
 
# Grand Slam point-by-point
 
def tour_from_match_num(match_num: str) -> str | None:
    s = str(match_num)
    if s.startswith("2") or s.startswith("WS"):
        return "WTA"
    if s.startswith("1") or s.startswith("MS"):
        return "ATP"
    return None
 
 
def load_slam() -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_tours = {"WTA", "ATP"} if INCLUDE_ATP_BASELINE else {"WTA"}
    point_cols = [
        "match_id", "SetNo", "GameNo", "PointNumber", "PointServer", "PointWinner",
        "P1Score", "P2Score",
    ]
    matches, points = [], []
    for mfile in sorted(SLAM_DIR.glob("*-matches.csv")):
        pfile = mfile.with_name(mfile.name.replace("-matches", "-points"))
        if not pfile.exists():
            continue
        m = pd.read_csv(mfile)
        m["tour"] = m["match_num"].map(tour_from_match_num)
        m = m[m["tour"].isin(keep_tours)]
        if m.empty:
            continue  
        p = pd.read_csv(pfile, usecols=lambda c: c in point_cols, low_memory=False, dtype={"P1Score": str, "P2Score": str})
        p = p[p["match_id"].isin(m["match_id"])]
        matches.append(m[["match_id", "year", "slam", "tour", "match_num", "player1", "player2"]])
        points.append(p)
 
    matches = pd.concat(matches, ignore_index=True)
    points = pd.concat(points, ignore_index=True)
 
    points = points[points["PointServer"].isin([1, 2])].copy()
    matches = matches[matches["match_id"].isin(points["match_id"])].copy()
    matches["surface"] = matches["slam"].map(SLAM_SURFACE)
    matches["round"] = matches["match_num"].astype(str).str[-3].map(ROUND_FROM_DIGIT)
    matches = matches.drop(columns="match_num")
    return matches, points
 
 
TOUR_SLAM_NAME = {"australian open": "ausopen", "roland garros": "frenchopen",
                  "wimbledon": "wimbledon", "us open": "usopen"}
 
 
def name_keys(name: str) -> tuple[str, str]:
    tokens = normalise_name(name).split()
    if len(tokens) < 2:
        return "", tokens[0] if tokens else ""
    return f"{tokens[0][0]} {' '.join(tokens[1:])}", tokens[-1]
 
 
def load_tour_slams(tour: str, years) -> pd.DataFrame:
    folder = WTA_DIR if tour == "WTA" else ATP_DIR
    frames = []
    for y in years:
        f = folder / f"{tour.lower()}_matches_{y}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, dtype={"winner_id": str, "loser_id": str})
        t = t[t["tourney_level"] == "G"].copy()
        t["year"] = y
        t["slam"] = t["tourney_name"].str.lower().map(TOUR_SLAM_NAME)
        t["tour"] = tour
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
  
    for side in ("winner", "loser"):
        t[f"{side}_k1"] = t[f"{side}_name"].map(lambda n: name_keys(n)[0])
        t[f"{side}_k2"] = t[f"{side}_name"].map(lambda n: name_keys(n)[1])
    return t
 
 
def match_to_tour(matches: pd.DataFrame) -> pd.DataFrame:
    tour = pd.concat([load_tour_slams(t, sorted(matches["year"].unique()))
                      for t in matches["tour"].unique()], ignore_index=True)
    m = matches.copy()
    for p in ("player1", "player2"):
        m[f"{p}_k1"] = m[p].map(lambda n: name_keys(n)[0])
        m[f"{p}_k2"] = m[p].map(lambda n: name_keys(n)[1])
 
    linked = []
    remaining_m, remaining_t = m, tour
    for k in ("k1", "k2"):
        remaining_m = remaining_m.assign(pair=[tuple(sorted(x)) for x in zip(remaining_m[f"player1_{k}"], remaining_m[f"player2_{k}"])])
        remaining_t = remaining_t.assign(pair=[tuple(sorted(x)) for x in zip(remaining_t[f"winner_{k}"], remaining_t[f"loser_{k}"])])
        keys = ["tour", "year", "slam", "pair"]
        uniq_m = remaining_m[~remaining_m.duplicated(keys, keep=False)]
        uniq_t = remaining_t[~remaining_t.duplicated(keys, keep=False)]
        hit = uniq_m.merge(uniq_t, on=keys, suffixes=("", "_t"))
       
        hit["p1_is_winner"] = hit[f"player1_{k}"] == hit[f"winner_{k}"]
        linked.append(hit)
        remaining_m = remaining_m[~remaining_m["match_id"].isin(hit["match_id"])]
        remaining_t = remaining_t[~remaining_t.set_index(keys).index.isin(hit.set_index(keys).index)]
 
 
    remaining_t = remaining_t.reset_index(drop=True).assign(t_row=lambda d: d.index)
    m_long = pd.concat([
        remaining_m.assign(m_side="player1", key=remaining_m["player1_k2"]),
        remaining_m.assign(m_side="player2", key=remaining_m["player2_k2"]),
    ])
    t_long = pd.concat([
        remaining_t.assign(t_side="winner", key=remaining_t["winner_k2"]),
        remaining_t.assign(t_side="loser", key=remaining_t["loser_k2"]),
    ])
    cand = m_long.merge(t_long[["tour", "year", "slam", "round", "key", "t_side", "t_row"]],
                        on=["tour", "year", "slam", "round", "key"])
    cand = cand[cand.groupby("match_id")["t_row"].transform("nunique") == 1]
    cand = cand.drop_duplicates("match_id")
    cand = cand[~cand["t_row"].duplicated(keep=False)]
    hit3 = cand[["match_id", "m_side", "t_side", "t_row"]].merge(remaining_m, on="match_id").merge(
        remaining_t.drop(columns=["tour", "year", "slam", "round"]), on="t_row", suffixes=("", "_t"))
    hit3["p1_is_winner"] = (hit3["m_side"] == "player1") == (hit3["t_side"] == "winner")
    linked.append(hit3)
 
    hit = pd.concat(linked, ignore_index=True)
    w = hit["p1_is_winner"]
    info = pd.DataFrame({
        "match_id": hit["match_id"],
        "player1": hit["winner_name"].where(w, hit["loser_name"]),
        "player2": hit["loser_name"].where(w, hit["winner_name"]),
        "player1_id": hit["winner_id"].where(w, hit["loser_id"]),
        "player2_id": hit["loser_id"].where(w, hit["winner_id"]),
        "player1_rank": hit["winner_rank"].where(w, hit["loser_rank"]),
        "player2_rank": hit["loser_rank"].where(w, hit["winner_rank"]),
        "score": hit["score"],
        "winner": hit["winner_name"],
    })
 
    out = matches.rename(columns={"player1": "player1_raw", "player2": "player2_raw"}).merge(info, on="match_id", how="left")
    out["player1"] = out["player1"].fillna(out["player1_raw"])
    out["player2"] = out["player2"].fillna(out["player2_raw"])
    print(f"  linked {out['player1_id'].notna().sum():,} of {len(out):,} Slam matches to tour results")
    return out
 
 
def clean_slam(matches: pd.DataFrame, points: pd.DataFrame):
    matches = match_to_tour(matches)
    points = points.merge(matches[["match_id", "player1", "player2", "tour"]], on="match_id")
    points["PointNumber"] = pd.to_numeric(points["PointNumber"], errors="coerce")
    points = points.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)
 
    s1 = points["PointServer"] == 1
    points["server"] = points["player1"].where(s1, points["player2"])
    points["returner"] = points["player2"].where(s1, points["player1"])
    points["server_won_point"] = points["PointWinner"] == points["PointServer"]
 
    # The GameWinner and BreakPoint columns are empty for the Australian Open and French Open
    # 2018-2021, so game winners and break points are worked out from the points themselves
  
    game_keys = [points["match_id"], points["SetNo"], points["GameNo"]]
    p1_before = points.groupby(game_keys)["P1Score"].shift(1).fillna("0")
    p2_before = points.groupby(game_keys)["P2Score"].shift(1).fillna("0")
    srv_before = p1_before.where(s1, p2_before)
    ret_before = p2_before.where(s1, p1_before)
    points["is_break_point"] = ((ret_before == "40") & srv_before.isin(["0", "15", "30"])) | (ret_before == "AD")
 
    nonstd = ~points["P1Score"].isin(STANDARD_SCORES) | ~points["P2Score"].isin(STANDARD_SCORES)
    points["is_tiebreak"] = nonstd.groupby(game_keys).transform("any")
    points.loc[points["is_tiebreak"], "is_break_point"] = False
 
    g = points.groupby(["match_id", "SetNo", "GameNo"], sort=False)
    games = g.agg(
        tour=("tour", "first"),
        server=("server", "first"),
        returner=("returner", "first"),
        server_num=("PointServer", "first"),
        is_tiebreak=("is_tiebreak", "first"),
        points_played=("PointNumber", "size"),
        break_points_faced=("is_break_point", "sum"),
        game_winner_num=("PointWinner", "last"),
    ).reset_index()
    games = games[games["game_winner_num"].isin([1, 2])]
    games["game_winner"] = np.where(games["game_winner_num"] == games["server_num"], games["server"], games["returner"])
    games["held"] = (games["game_winner_num"] == games["server_num"]).astype("boolean")
    games.loc[games["is_tiebreak"], "held"] = pd.NA
    games = games.drop(columns=["server_num", "game_winner_num"])
 
    fallback = games.drop_duplicates("match_id", keep="last").set_index("match_id")["game_winner"]
    missing = matches["winner"].isna()
    matches.loc[missing, "winner"] = matches.loc[missing, "match_id"].map(fallback)
    matches["winner_source"] = "tour results"
    matches.loc[missing, "winner_source"] = "last game (unmatched)"
 
    points = points[[
        "match_id", "tour", "SetNo", "GameNo", "PointNumber", "server", "returner",
        "server_won_point", "is_break_point", "is_tiebreak", "P1Score", "P2Score",
    ]].rename(columns={"SetNo": "set_no", "GameNo": "game_no", "PointNumber": "point_no",
                       "P1Score": "p1_score", "P2Score": "p2_score"})
    games = games.rename(columns={"SetNo": "set_no", "GameNo": "game_no"})
    return matches, games, points
 
 
 
FAULT = re.compile(r"[0-9]\+?[nwdxge!V]")  
 
 
def parse_serve(code) -> tuple[str | None, bool | None]:
    if not isinstance(code, str) or not code:
        return None, None
    code = code.lstrip("c")  
    if not code or not code[0].isdigit():
        return None, None  
    return SERVE_DIRECTION.get(code[0]), bool(FAULT.fullmatch(code))
 
 
def score_state(pts: str):
    idx = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
    try:
        a, b = str(pts).split("-")
    except ValueError:
        return None, None, False
    if a in idx and b in idx:
        s, r = idx[a], idx[b]
        is_bp = (r == 3 and s < 3) or r == 4
        return s, r, is_bp
    if a.isdigit() and b.isdigit(): 
        return int(a), int(b), False
    return None, None, False
 
 
def load_charting() -> pd.DataFrame:
    cm = pd.read_csv(CHARTING_DIR / "charting-w-matches.csv", encoding="latin-1", dtype=str)
    cm["date"] = pd.to_datetime(cm["Date"], format="%Y%m%d", errors="coerce")
    cm = cm[["match_id", "Player 1", "Player 2", "date", "Tournament", "Round", "Surface"]].rename(
        columns={"Player 1": "player1", "Player 2": "player2", "Tournament": "tournament",
                 "Round": "round", "Surface": "surface"})
 
    pts = pd.concat(
        [pd.read_csv(f, encoding="latin-1", low_memory=False, dtype=str)
         for f in sorted(CHARTING_DIR.glob("charting-w-points-*.csv"))],
        ignore_index=True,
    )
    pts = pts.merge(cm, on="match_id", how="inner")
 
    s1 = pts["Svr"] == "1"
    pts["server"] = pts["player1"].where(s1, pts["player2"])
    pts["returner"] = pts["player2"].where(s1, pts["player1"])
    pts["server_won_point"] = pts["PtWinner"] == pts["Svr"]
 
    state = pts["Pts"].map(score_state)
    pts["is_break_point"] = state.str[2]
    total = state.str[0] + state.str[1]
    pts["court"] = total.map(lambda t: None if pd.isna(t) else ("deuce" if t % 2 == 0 else "ad"))
 
    first = pts["1st"].map(parse_serve)
    second = pts["2nd"].map(parse_serve)
    base = ["match_id", "date", "tournament", "round", "surface", "server", "returner",
            "Pt", "court", "is_break_point", "server_won_point"]
    s1_rows = pts[base].assign(serve_number=1, direction=first.str[0], fault=first.str[1])
    has_2nd = pts["2nd"].notna() & (pts["2nd"] != "")
    s2_rows = pts.loc[has_2nd, base].assign(
        serve_number=2, direction=second[has_2nd].str[0], fault=second[has_2nd].str[1])
 
    serves = pd.concat([s1_rows, s2_rows], ignore_index=True)
    serves = serves[serves["direction"].notna()]  
    serves = serves.rename(columns={"Pt": "point_no"})
    serves["point_no"] = pd.to_numeric(serves["point_no"], errors="coerce")
    return serves.sort_values(["match_id", "point_no", "serve_number"]).reset_index(drop=True)
 
# Player lookup across sources
 
def build_player_lookup(slam_matches: pd.DataFrame, serves: pd.DataFrame) -> pd.DataFrame:
    wta = pd.read_csv(WTA_DIR / "wta_players.csv", dtype=str)
    wta["key"] = (wta["name_first"].fillna("") + " " + wta["name_last"].fillna("")).map(normalise_name)
    n_ids = wta.groupby("key")["player_id"].transform("nunique")
    by_name = wta[n_ids == 1].drop_duplicates("key").set_index("key")["player_id"]
    ambiguous = set(wta.loc[n_ids > 1, "key"])
 
    slam = pd.concat([
        slam_matches[["player1", "player1_id"]].set_axis(["name", "player_id"], axis=1),
        slam_matches[["player2", "player2_id"]].set_axis(["name", "player_id"], axis=1),
    ]).assign(source="slam")
    chart = pd.DataFrame({"name": pd.concat([serves["server"], serves["returner"]]).unique(), "source": "charting"})
    names = pd.concat([slam, chart], ignore_index=True).dropna(subset=["name"])
 
    lookup = names.groupby("name").agg(
        source=("source", lambda s: ",".join(sorted(set(s)))),
        player_id=("player_id", "first"), 
    ).reset_index()
    lookup["name_key"] = lookup["name"].map(normalise_name)
    no_id = lookup["player_id"].isna()
    lookup.loc[no_id, "player_id"] = lookup.loc[no_id, "name_key"].map(by_name)
 
    lookup["match_status"] = "matched"
    lookup.loc[lookup["player_id"].isna(), "match_status"] = "unmatched"
    lookup.loc[lookup["player_id"].isna() & lookup["name_key"].isin(ambiguous), "match_status"] = "ambiguous"
 
    info = wta.drop_duplicates("player_id").set_index("player_id")[["hand", "dob", "ioc"]]
    return lookup.join(info, on="player_id")
 
 
def main():
    check_folders()
    OUT.mkdir(parents=True, exist_ok=True)
 
    print("Loading Grand Slam point-by-point...")
    slam_matches, slam_points = load_slam()
    slam_matches, slam_games, slam_points = clean_slam(slam_matches, slam_points)
 
    print("Loading Match Charting Project...")
    serves = load_charting()
 
    print("Building player lookup...")
    lookup = build_player_lookup(slam_matches, serves)
 
    slam_matches.to_csv(OUT / "slam_matches_wta.csv", index=False)
    slam_games.to_csv(OUT / "slam_games_wta.csv", index=False)
    slam_points.to_csv(OUT / "slam_points_wta.csv", index=False)
    serves.to_csv(OUT / "charting_serves_wta.csv", index=False)
    lookup.to_csv(OUT / "player_lookup.csv", index=False)
 
    # Summary 
    std = slam_games[~slam_games["is_tiebreak"].astype(bool)]
    print("\nSummary")
    print(f"Slam matches: {len(slam_matches):,}  ({slam_matches['year'].min()}-{slam_matches['year'].max()})"
          f"  | linked to tour results: {slam_matches['player1_id'].notna().mean():.1%}")
    print(f"Slam points:  {len(slam_points):,}")
    print(f"Slam games:   {len(slam_games):,}  | hold rate {std['held'].astype(float).mean():.1%}")
    print("Hold rate by surface:")
    by_surf = std.merge(slam_matches[["match_id", "surface"]], on="match_id")
    print(by_surf.groupby("surface")["held"].apply(lambda s: s.astype(float).mean()).round(3).to_string())
    print(f"\nCharted serves: {len(serves):,} from {serves['match_id'].nunique():,} matches")
    print(serves.groupby("serve_number")["direction"].value_counts(normalize=True).round(3).unstack().to_string())
    print(f"Break-point serves: {serves['is_break_point'].sum():,}")
    print(f"\nPlayers: {len(lookup):,} | " + ", ".join(f"{k} {v}" for k, v in lookup["match_status"].value_counts().items()))
    print(f"\nFiles written to {OUT}")
 
 
if __name__ == "__main__":
    main()
 
"""
Findings:
- Hold rate of 66% (101k service games) --> Women hold serve about 2/3 of the time at the Slams, 
so breaks happen roughly once every 3 service games (thus, "serve isn't king" premise is confirmed)
- Clay is lowest (63%) and grass highest (69%) --> slower courts blunt the serve 
- 2nd serves go to the body 45% of the time, against 20% for 1st serves --> Players play it safe when they can't 
afford a fault (pressure makes serves more predictable)
"""
