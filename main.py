from __future__ import annotations

from classes import indexparse


def main():

    # Prepare clean parquet
    csv_path, parquet_path = "data/raw/index.csv", "data/sanitized/index.parquet"
    indexparse.convert_csv(csv_path, parquet_path)
    index = indexparse.read_parquet(parquet_path)
    print(index)


if __name__ == "__main__":
    main()

# data/raw/replays/osr/{replayhash}.osr
# data/raw/beatmaps/{beatmaphash}.osu
