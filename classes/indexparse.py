from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

from .mods import Mods

__all__ = ["convert_csv", "read_parquet"]


def conjugate_conds(*conds):
    return functools.reduce(np.logical_and, conds)


def convert_csv(filename: str, output: str):

    df = pd.read_csv(filename, low_memory=False)
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

    star_rating_rx = re.compile(r"\[([4-7](\.\d{1,2})?) ⭐]")
    invalid_usernames = ["osu!", "lazer!dance"]
    allowed_mods = Mods("HDFLNFPFSD")

    star_rating_cond = df["summary"].str.contains(star_rating_rx)
    username_cond = ~df["playerName"].isin(invalid_usernames)
    obj_min_count_cond = df["beatmap-HitObjects"] >= 50
    bpm_max_cond = df["beatmap-BPMMax"] <= 700
    bpm_min_cond = df["beatmap-BPMMin"] >= 15
    ss_cond = df["performance-Accuracy"] == 1
    mods_cond = df["mods"] & allowed_mods.bitwise

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
    df.to_parquet(output, compression=None)


def read_parquet(filename: str):
    return pd.read_parquet(filename, engine="fastparquet")
