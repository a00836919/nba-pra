# ingestion/pull_nba_api.py
"""
Ingest historical NBA player box scores via nba_api and persist season-level parquet files.

- Uses `LeagueGameLog` (team logs) to enumerate Regular Season GAME_IDs and dates for a given season.
- Pulls player box scores via `BoxScoreTraditionalV3` (fallback to V2 if needed).
- Extracts minimal columns and saves to `/data/raw/player_game_{season_end_year}.parquet`.

Run:
    python -m ingestion.pull_nba_api --seasons 2023-24 2024-25
    python -m ingestion.pull_nba_api --last-n 3

Notes:
- Respect NBA Stats rate limits. Default sleep is 0.6s with small random jitter.
- Requires `pyarrow` for parquet.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
import random
from typing import Dict, List

import pandas as pd

# nba_api imports
from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoretraditionalv2,
    boxscoretraditionalv3,
)

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
    """Convert '2023-24' -> 2024 (end year) for filename."""
    try:
        start, end2 = season_label.split("-")
        return int("20" + end2) if len(end2) == 2 else int(end2)
    except Exception as e:
        raise ValueError(f"Unexpected season label: {season_label}") from e


def get_recent_seasons(last_n: int = 2) -> List[str]:
    """Return last_n season labels (e.g., ['2023-24','2024-25'])."""
    today = pd.Timestamp.today(tz="America/New_York").date()
    year = today.year
    start_year = year if today.month >= 10 else year - 1
    return [f"{s}-{str((s + 1) % 100).zfill(2)}" for s in range(start_year, start_year - last_n, -1)]


def fetch_game_index(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return unique GAME_ID and GAME_DATE for the given season & season type.

    Uses LeagueGameLog (team logs) -> unique GAME_IDs; filters out future dates.
    """
    logger.info(f"Fetching {season_type} games for season {season} via LeagueGameLog…")
    # Correct parameter names for LeagueGameLog:
    # - season
    # - season_type_all_star
    # - player_or_team_abbreviation ("T" for team logs)
    lgl = leaguegamelog.LeagueGameLog(
        league_id="00",
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",  # team logs -> one row per team per game
        sorter="DATE",
        direction="ASC",
    )
    df = lgl.get_data_frames()[0][["GAME_ID", "GAME_DATE"]].drop_duplicates()
    # Parse dates like "OCT 24, 2024" -> date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date

    # keep only games that have occurred (avoid empty box scores for future dates)
    today = pd.Timestamp.today(tz="America/New_York").date()
    df = df[df["GAME_DATE"] <= today]

    # one row per GAME_ID
    df = df.sort_values("GAME_DATE").drop_duplicates("GAME_ID", keep="first")
    logger.info(f"Found {len(df):,} {season_type} games (through {today}) for season {season}.")
    return df


def _extract_player_frame_v3(bs_v3: boxscoretraditionalv3.BoxScoreTraditionalV3) -> pd.DataFrame:
    """
    BoxScoreTraditionalV3 returns multiple result sets; find the player-level one.
    Prefer the first frame that includes PLAYER_ID & TEAM_ID.
    """
    for df in bs_v3.get_data_frames():
        if {"PLAYER_ID", "TEAM_ID"}.issubset(set(df.columns)):
            return df.copy()
    raise RuntimeError("Could not locate player-level result set in BoxScoreTraditionalV3.")


def pull_box_scores_for_season(
    season: str,
    sleep: float = 0.6,
    max_retries: int = 3,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    games_df = fetch_game_index(season, season_type=season_type)
    game_dates: Dict[str, str] = dict(zip(games_df["GAME_ID"], games_df["GAME_DATE"]))
    records: List[pd.DataFrame] = []
    pulled_ok = 0

    for i, (game_id, game_date) in enumerate(game_dates.items(), start=1):
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                # --- Try V3 ---
                bs_v3 = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
                df = _extract_player_frame_v3(bs_v3)
                if df.empty:
                    raise ValueError("V3 returned empty frame")
                df["GAME_DATE"] = pd.to_datetime(game_date)
                keep = [c for c in NEEDED_COLS if c in df.columns or c == "GAME_DATE"]
                df = df[keep]
                records.append(df)
                pulled_ok += 1
                break
            except Exception as e_v3:
                last_err = e_v3
                # --- Fallback: V2 ---
                try:
                    bs_v2 = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                    df = bs_v2.get_data_frames()[0]
                    if df.empty:
                        raise ValueError("V2 returned empty frame")
                    df["GAME_DATE"] = pd.to_datetime(game_date)
                    keep = [c for c in NEEDED_COLS if c in df.columns or c == "GAME_DATE"]
                    df = df[keep]
                    records.append(df)
                    pulled_ok += 1
                    logger.debug(f"Fell back to V2 for GAME_ID={game_id} due to: {e_v3}")
                    break
                except Exception as e_v2:
                    last_err = e_v2
                    wait = sleep * attempt
                    logger.warning(
                        f"GAME_ID={game_id} attempt {attempt}/{max_retries} failed: {e_v2}. Sleeping {wait:.2f}s"
                    )
                    time.sleep(wait)

        else:
            logger.error(f"Giving up on GAME_ID={game_id}. Last error: {last_err}")

        # polite sleep + small jitter
        time.sleep(sleep + random.uniform(0, 0.2))

        if i % 100 == 0:
            logger.info(f"Progress: {i:,}/{len(game_dates):,} games | successful pulls: {pulled_ok:,}")

    if not records:
        raise RuntimeError(f"No box score data collected for season {season} ({season_type}).")

    out = pd.concat(records, ignore_index=True)

    # Basic dtype cleanup
    for c in ["PTS", "REB", "AST", "FGA", "FTA", "TOV"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.date

    # Ensure minimal columns and order
    keep = [c for c in NEEDED_COLS if c in out.columns]
    out = out[keep]

    logger.info(f"Collected {len(out):,} player-game rows for {season} ({season_type}).")
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
    group.add_argument("--seasons", nargs="+", help="Explicit list of season labels like 2023-24 2024-25")
    group.add_argument("--last-n", type=int, help="Fetch the last N seasons (including current)")
    p.add_argument("--sleep", type=float, default=0.6, help="Seconds to sleep between API calls")
    p.add_argument("--out-dir", type=str, default=RAW_DIR, help="Output directory for parquet files")
    p.add_argument(
        "--season-type",
        type=str,
        default="Regular Season",
        choices=["Regular Season", "Playoffs", "PlayIn", "All-Star"],
        help="Which season type to pull (default: Regular Season)",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    seasons = args.seasons if args.seasons else get_recent_seasons(args.last_n)

    logger.info(f"Seasons to fetch: {seasons} ({args.season_type})")

    for season in seasons:
        df = pull_box_scores_for_season(
            season=season,
            sleep=args.sleep,
            season_type=args.season_type,
        )
        save_season_parquet(df, season, out_dir=args.out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
