from __future__ import annotations

import functools
import logging
import re

import numpy as np
import pandas as pd
from alive_progress import alive_bar

from .mods import Mods

__all__ = ["convert_csv", "read_parquet"]


def conjugate_conds(*conds):
    return functools.reduce(np.logical_and, conds)


def convert_csv(filename: str, output: str):
    with alive_bar(3, dual_line=True, title="Indexing...") as bar:
        bar.text = "-> Reading CSV, please wait..."

        df = pd.read_csv(filename, engine="pyarrow")
        df = df.astype(
            {
                "replayHash": str,
                "beatmapHash": str,
                "summary": str,
                "date": "datetime64[ns]",
                "playerName": str,
                "modsReadable": str,
                "mods": int,
                "performance-IsFC": bool,
                "performance-IsFail": bool,
                "performance-Accuracy": float,
                "performance-Score": int,
                "performance-300s": int,
                "performance-100s": int,
                "performance-50s": int,
                "performance-Misses": int,
                "performance-Geki": int,
                "performance-Katu": int,
                "performance-MaxCombo": int,
                "beatmap-Artist": str,
                "beatmap-Title": str,
                "beatmap-Version": str,  # Difficulty
                "beatmap-BPMMax": float,
                "beatmap-BPMMin": float,
                "beatmap-Id": int,
                "beatmap-SetId": int,
                "beatmap-HP": float,
                "beatmap-OD": float,
                "beatmap-AR": float,
                "beatmap-CS": float,
                "beatmap-MaxCombo": int,
                "beatmap-HitObjects": int,
                "beatmap-Circles": int,
                "beatmap-Sliders": int,
                "beatmap-Spinners": int,
                "beatmapPlay-BPMMax": float,
                "beatmapPlay-BPMMin": float,
                "beatmapPlay-HP": float,
                "beatmapPlay-OD": float,
                "beatmapPlay-AR": float,
                "beatmapPlay-CS": float,
                "osrReplayUrl": str,
            },
            errors="ignore",
        )

        bar()
        bar.text = "-> Filtering data, please wait..."

        star_rating_rx = re.compile(r"\[(?:[4-7](?:\.\d{1,2})?) ⭐]")
        invalid_usernames = ["osu!", "lazer!dance"]
        allowed_mods = Mods("HDFLNFPFSD")

        star_rating_cond = df["summary"].str.contains(
            star_rating_rx,
        )  # Filter maps to 4-7*
        username_cond = ~df["playerName"].isin(
            invalid_usernames,
        )  # Filter out invalid usernames
        obj_min_count_cond = df["beatmap-HitObjects"] >= 50  # Minimum of 50 objects
        bpm_max_cond = np.logical_and(
            15 <= df["beatmap-BPMMax"],
            df["beatmap-BPMMax"] <= 700,
        )  # Sane BPM limits
        bpm_min_cond = np.logical_and(
            15 <= df["beatmap-BPMMin"],
            df["beatmap-BPMMin"] <= 700,
        )  # Sane BPM limits
        ss_cond = df["performance-Accuracy"] == 1  # Filter to only SS scores
        mods_cond = np.logical_or(
            df["mods"] == 0,
            df["mods"] & allowed_mods.bitwise,
        )  # Filter to only NM and allowed_mods

        filter = conjugate_conds(
            star_rating_cond,
            username_cond,
            obj_min_count_cond,
            bpm_max_cond,
            bpm_min_cond,
            ss_cond,
            mods_cond,
        )
        df = df[filter]
        df = df[["replayHash", "beatmapHash"]]

        bar()
        bar.text = "Saving as parquet, please wait..."

        df.to_parquet(output)
        bar()

    logging.info(f"Indexed {len(df.index)} replays")


def read_parquet(filename: str):
    return pd.read_parquet(filename, engine="fastparquet")
