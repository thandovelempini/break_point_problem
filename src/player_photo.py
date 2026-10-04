"""
The Break Point Problem - add player photos for Tableau.

Adds four columns to data/tableau/players.csv:
  photo_url      direct link to a 300px-wide photo (use with Tableau's Image Role)
  photo_credit   photographer / author, as required by the photo's licence
  photo_license  e.g. "CC BY-SA 4.0"
  photo_page     the photo's page on Wikimedia Commons (full licence details)

Photos come from Wikimedia Commons via each player's Wikidata entry, so they are
openly licensed - but most licences require you to credit the author, which is
why the credit columns exist. Don't swap in photos from Google or the WTA site:
those are usually copyrighted.

Each photo is also saved to images/players/<player_id>.jpg, and photo_url points at
that copy on GitHub (set GITHUB_RAW_BASE below), because Tableau can't reliably load
photos straight from Wikimedia. Push the images folder to GitHub after running.

Needs an internet connection. Results are cached in data/raw/player_photos_cache.csv
and photos already in images/players/ are skipped, so reruns are quick.

Run from the project root, AFTER export_tableau.py:  python3 src/add_player_photos.py
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAYERS = ROOT / "data" / "tableau" / "players.csv"
WTA_PLAYERS = ROOT / "data" / "raw" / "tennis-sackmann-archive" / "wta" / "wta_players.csv"
CACHE = ROOT / "data" / "raw" / "player_photos_cache.csv"

WIDTH = 300           # photo width in pixels
BATCH = 50            # Wikidata accepts up to 50 players per request
PHOTO_BATCH = 10      # Commons builds a thumbnail for each photo, so ask for fewer at a time
PAUSE = 2             # seconds between requests
MAX_TRIES = 6
# Wikimedia asks every script to identify itself, and throttles anonymous-looking ones
# harder. Replace the bracket with your GitHub repo link (or an email) for fewer 429s.
USER_AGENT = "BreakPointProblem/1.0 (tennis analytics portfolio project; python urllib)"
PHOTO_COLS = ["photo_url", "photo_credit", "photo_license", "photo_page"]

# Tableau can't reliably load photos straight from Wikimedia (it throttles them), so the
# script saves a copy of each photo in images/players/ and points photo_url at the copy
# on GitHub. Set this to YOUR repo: https://raw.githubusercontent.com/<username>/<repo>/<branch>
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/thandovelempini/break_point_problem/main"
IMAGES = ROOT / "images" / "players"


def fetch(url):
    """Download a URL, waiting and retrying when Wikimedia says 'too many requests' (429)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_TRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 503) or attempt == MAX_TRIES:
                raise
            retry_after = err.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else 15 * attempt
            print(f"\n  Wikimedia asked us to slow down - waiting {wait} s (attempt {attempt} of {MAX_TRIES})...")
            time.sleep(wait)


def get_json(url, params):
    return json.loads(fetch(f"{url}?{urllib.parse.urlencode(params)}"))


def strip_html(text):
    """Commons credits are HTML (often a link to the author's page); keep the plain text."""
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


def batches(items, size=BATCH):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def photo_filenames(wikidata_ids):
    """Wikidata ID -> Commons file name, from each entry's 'image' property (P18)."""
    found = {}
    for chunk in batches(wikidata_ids):
        data = get_json("https://www.wikidata.org/w/api.php", {
            "action": "wbgetentities", "ids": "|".join(chunk), "props": "claims", "format": "json"})
        for qid, entity in data.get("entities", {}).items():
            claims = entity.get("claims", {}).get("P18", [])
            if claims:
                value = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")
                if value:
                    found[qid] = value
        time.sleep(PAUSE)  # be polite to the API
    return found


def photo_details(filenames):
    """Commons file name -> thumbnail URL, credit, licence and page link."""
    details = {}
    chunks = list(batches(filenames, PHOTO_BATCH))
    for i, chunk in enumerate(chunks, 1):
        print(f"  photo batch {i} of {len(chunks)}", end="\r")
        titles = [f"File:{name}" for name in chunk]
        data = get_json("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
            "iiprop": "url|extmetadata", "iiurlwidth": WIDTH, "format": "json"})
        query = data.get("query", {})
        # The API may tidy titles (e.g. underscores to spaces); map them back.
        renamed = {n["to"]: n["from"] for n in query.get("normalized", [])}
        for page in query.get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("thumburl"):
                continue
            meta = info.get("extmetadata", {})
            title = renamed.get(page["title"], page["title"])
            details[title.removeprefix("File:")] = {
                "photo_url": info["thumburl"],
                "photo_credit": strip_html(meta.get("Artist", {}).get("value")) or "Unknown author",
                "photo_license": strip_html(meta.get("LicenseShortName", {}).get("value")),
                "photo_page": info.get("descriptionurl"),
            }
        time.sleep(PAUSE)
    print()
    return details


def save_photos(players):
    """Save each photo to images/players/<player_id>.<ext> and point photo_url at the GitHub copy.
    Photos already saved are skipped, so reruns are quick."""
    if "YOUR-USERNAME" in GITHUB_RAW_BASE:
        print("\nGITHUB_RAW_BASE isn't set yet, so photo_url still points at Wikimedia."
              "\nSet it near the top of this script and run again.")
        return players
    IMAGES.mkdir(parents=True, exist_ok=True)
    players = players.copy()
    players["photo_source"] = players["photo_url"]
    todo = players[players["photo_url"].notna() & players["player_id"].notna()]
    saved = 0
    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        ext = Path(urllib.parse.urlparse(row["photo_source"]).path).suffix.lower() or ".jpg"
        name = f"{row['player_id']}{ext}"
        path = IMAGES / name
        if not path.exists():
            print(f"  downloading photo {i} of {len(todo)}", end="\r")
            path.write_bytes(fetch(row["photo_source"]))
            saved += 1
            time.sleep(1)
        players.loc[idx, "photo_url"] = f"{GITHUB_RAW_BASE}/images/players/{name}"
    players.loc[players["player_id"].isna(), "photo_url"] = None
    # The photos' licences require credit wherever the files are shared (including GitHub).
    credits = todo.assign(file=[f"{pid}{Path(urllib.parse.urlparse(u).path).suffix.lower() or '.jpg'}"
                                for pid, u in zip(todo["player_id"], todo["photo_source"])])
    credits[["file", "player", "photo_credit", "photo_license", "photo_page"]].rename(columns={
        "photo_credit": "author", "photo_license": "license", "photo_page": "source"}
    ).sort_values("file").to_csv(IMAGES / "CREDITS.csv", index=False)
    print(f"\n{saved} new photos saved to {IMAGES} ({len(todo)} in total, credits in CREDITS.csv).")
    print("Now commit and push the images folder to GitHub so the links work:")
    print('  git add images && git commit -m "Add player photos" && git push')
    return players


def main():
    players = pd.read_csv(PLAYERS, dtype={"player_id": str})
    players = players.drop(columns=[c for c in PHOTO_COLS if c in players], errors="ignore")
    ids = pd.read_csv(WTA_PLAYERS, dtype=str)[["player_id", "wikidata_id"]].dropna().drop_duplicates("player_id")
    players = players.merge(ids, on="player_id", how="left")

    cache = pd.read_csv(CACHE, dtype=str) if CACHE.exists() else pd.DataFrame(columns=["wikidata_id"] + PHOTO_COLS)
    wanted = sorted(set(players["wikidata_id"].dropna()) - set(cache["wikidata_id"]))

    if wanted:
        print(f"Looking up {len(wanted)} players on Wikidata...")
        files = photo_filenames(wanted)
        print(f"  {len(files)} have a photo. Fetching photo details from Wikimedia Commons...")
        details = photo_details(sorted(set(files.values())))
        new = pd.DataFrame([{"wikidata_id": qid, **details.get(files.get(qid), {})} for qid in wanted])
        cache = pd.concat([cache, new.reindex(columns=cache.columns)], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(CACHE, index=False)
    else:
        print("All players already in the cache.")

    players = players.merge(cache, on="wikidata_id", how="left").drop(columns="wikidata_id")
    players = save_photos(players)
    players.to_csv(PLAYERS, index=False)

    charted = players[players["in_charting_data"]]
    print(f"\nPhotos found for {charted['photo_url'].notna().sum()} of {len(charted)} charted players "
          f"({players['photo_url'].notna().sum()} of {len(players)} overall).")
    print(f"Updated {PLAYERS}")


if __name__ == "__main__":
    main()