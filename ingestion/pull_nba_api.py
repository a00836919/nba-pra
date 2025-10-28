"""pull_nba_api.py

Ingest historical NBA player box scores via nba_api and persist season-level parquet files.

- Uses `LeagueGameFinder` to enumerate GAME_IDs and dates for a given season.
- Loops over those GAME_IDs and pulls `BoxScoreTraditionalV2`.
- Extracts minimal columns and saves to `/data/raw/player_game_{season_end_year}.parquet`.

Run examples:
    python -m ingestion.pull_nba_api --seasons 2023-24 2024-25
    python -m ingestion.pull_nba_api --last-n 3

Notes:
- Be respectful of the NBA Stats rate limits. Default sleep between calls is 0.6s; adjust with `--sleep`.
- Parquet requires `pyarrow`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# nba_api imports
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2

# ----------------------------------
# Logging
# ----------------------------------
logger = logging.getLogger("pull_nba_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ----------------------------------
# Constants
# ----------------------------------
RAW_DIR = os.path.join("data", "raw")
NEEDED_COLS = [
    "GAME_ID",
    "PLAYER_ID",
    "TEAM_ID",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "FGA",
    "FTA",
    "TOV",
    "GAME_DATE",
]

# ----------------------------------
# Helpers
# ----------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def season_label_to_end_year(season_label: str) -> int:
    """Convert season label like '2023-24' -> 2024 (end year) for filename.
    """
    try:
        start, end2 = season_label.split("-")
        return int("20" + end2) if len(end2) == 2 else int(end2)
    except Exception as e:
        raise ValueError(f"Unexpected season label: {season_label}") from e


def get_recent_seasons(last_n: int = 2) -> List[str]:
    """Return last_n completed/current season labels (e.g., ['2023-24','2024-25']).

    Uses NBA season convention: YYYY-YY. Assumes current date determines current season.
    """
    # Determine current season start based on Oct 1 cutoff.
    today = pd.Timestamp.today(tz="America/New_York").date()
    year = today.year
    # NBA seasons typically start in Oct. If before July, we're in season ending this calendar year.
    # Approximate: if month >= 10 -> season start is current year; else start is previous year.
    start_year = year if today.month >= 10 else year - 1
    seasons: List[str] = []
    for i in range(last_n):
        s = start_year - i
        seasons.append(f"{s}-{str((s + 1) % 100).zfill(2)}")
    return seasons


def fetch_game_index(season: str, season_type: str | None = None) -> pd.DataFrame:
    """Fetch game index for a season using LeagueGameFinder.

    Returns dataframe with at least ['GAME_ID','GAME_DATE'] unique per GAME_ID.
    """
    logger.info(f"Fetching game index for season {season} (season_type={season_type})…")
    lgf = leaguegamefinder.LeagueGameFinder(
        league_id_nullable="00",
        season_nullable=season,
        season_type_nullable=season_type,  # None -> all types
    )
    df = lgf.get_data_frames()[0]
    # Deduplicate and keep earliest date per game id (there should be one row per team per game)
    df = (
        df[["GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .groupby("GAME_ID", as_index=False)
        .agg({"GAME_DATE": "min"})
    )
    # Normalize date to YYYY-MM-DD
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    logger.info(f"Found {len(df):,} games for season {season}.")
    return df


def pull_box_scores_for_season(
    season: str,
    sleep: float = 0.6,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Loop over GAME_IDs for the season and collect player box scores.

    Returns a dataframe with minimal columns defined in NEEDED_COLS.
    """
    games_df = fetch_game_index(season)
    game_dates: Dict[str, str] = dict(zip(games_df["GAME_ID"], games_df["GAME_DATE"]))

    records: List[pd.DataFrame] = []

    for i, (game_id, game_date) in enumerate(game_dates.items(), start=1):
        # Simple retry loop for transient errors / 429s
        for attempt in range(1, max_retries + 1):
            try:
                bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                df = bs.get_data_frames()[0]
                # Select/rename minimal columns and attach date
                cols_present = [c for c in NEEDED_COLS if c in df.columns]
                df = df[[c for c in cols_present if c != "GAME_DATE"]].copy()
                df["GAME_DATE"] = pd.to_datetime(game_date)
                # Keep only required
                df = df[[c for c in NEEDED_COLS if c in df.columns]]
                records.append(df)
                if i % 100 == 0:
                    logger.info(f"Pulled {i:,}/{len(game_dates):,} games…")
                break  # success
            except Exception as e:
                wait = sleep * attempt
                logger.warning(
                    f"Error on GAME_ID={game_id} (attempt {attempt}/{max_retries}): {e}. Sleeping {wait:.2f}s"
                )
                time.sleep(wait)
                if attempt == max_retries:
                    logger.error(f"Giving up on GAME_ID={game_id} after {max_retries} attempts.")
        time.sleep(sleep)

    if not records:
        raise RuntimeError(f"No box score data collected for season {season}.")

    out = pd.concat(records, ignore_index=True)

    # Basic dtypes clean-up
    numeric_cols = ["PTS", "REB", "AST", "FGA", "FTA", "TOV"]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # GAME_DATE to date
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.date

    # Ensure only minimal cols and order
    keep = [c for c in NEEDED_COLS if c in out.columns]
    out = out[keep]

    return out


def save_season_parquet(df: pd.DataFrame, season: str, out_dir: str = RAW_DIR) -> str:
    ensure_dir(out_dir)
    end_year = season_label_to_end_year(season)
    path = os.path.join(out_dir, f"player_game_{end_year}.parquet")
    logger.info(f"Saving {len(df):,} player-game rows to {path}")
    df.to_parquet(path, index=False)
    return path


# ----------------------------------
# CLI
# ----------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull NBA box scores and persist parquet by season.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--seasons",
        nargs="+",
        help="Explicit list of season labels like 2023-24 2024-25",
    )
    group.add_argument(
        "--last-n",
        type=int,
        help="Fetch the last N seasons (including current)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Seconds to sleep between API calls (respect rate limits)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=RAW_DIR,
        help="Output directory for parquet files",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    seasons = args.seasons if args.seasons else get_recent_seasons(args.last_n)

    logger.info(f"Seasons to fetch: {seasons}")

    for season in seasons:
        df = pull_box_scores_for_season(season=season, sleep=args.sleep)
        save_season_parquet(df, season, out_dir=args.out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
