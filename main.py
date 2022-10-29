from __future__ import annotations

import logging

from classes import indexparse
from classes import logformat
from classes import replayparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logformat.LoggingFormat())
logger.addHandler(ch)


def main():
    csv_path, parquet_path = "data/raw/index.csv", "data/sanitized/index.parquet"
    indexparse.convert_csv(csv_path, parquet_path)  # Prepare clean parquet
    index = indexparse.read_parquet(parquet_path)
    replayparse.parse_replays(index)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Received keyboard interrupt.")
