import os
import csv
import argparse


def process_metadata(metadata_path, save_path):
    """
    Processes Mara dataset metadata by extracting filenames and retaining metadata,
    then saves the processed data to a new CSV file.

    Args:
        metadata_path (str): Path to the input metadata CSV file.
        save_path (str): Path where the processed metadata will be saved.

    Note:
        The input CSV is expected to be tab-delimited. The output CSV will be pipe-delimited.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)

    processed_data = [
        [
            p[0].split("/")[-1].split(".")[0],
            p[1],
        ]
        for p in data
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(processed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--metadata_path",
        type=str,
        default="/train_path/Mara/mara.csv",
        help="Path to metadata.csv",
    )

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="/train_path/Mara/metadata.csv",
        help="Where to save processed metadata",
    )

    args = parser.parse_args()

    process_metadata(args.metadata_path, args.save_path)
