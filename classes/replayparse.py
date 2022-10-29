from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import slider
from alive_progress import alive_it

__all__ = ["parse_replays"]
"""
input(hitobjects): [x,y,is_visible,is_circle, is_slider, is_spinner]
target(cursor): [x,y]
x, y: [-0.5, 0.5]
"""


def parse_replays(df: pd.DataFrame):
    total, skipped = len(df.index), 0
    for replay_hash, beatmap_hash in alive_it(df.values, title="Processing replays..."):
        try:
            replay_path = f"data/raw/replays/osr/{replay_hash}.osr"
            replay = slider.Replay.from_path(replay_path, retrieve_beatmap=False)

            beatmap_path = f"data/raw/beatmaps/{beatmap_hash}.osu"
            beatmap = slider.Beatmap.from_path(beatmap_path)

            for action in replay.actions:
                x, y = action.position.x, action.position.y

        except Exception as e:
            skipped += 1
            logging.warn(f"Skipped {replay_path} ({skipped} / {total}) Reason: {e}")

    logging.info(
        f"Finished loading {total-skipped} replays. {skipped} failed to load ({skipped/total*100 :.2f}%)",
    )
