import os
import csv
import random
import argparse


def split_ljspeech(metadata_path, save_dir, validation_size=0.1, seed=88):

    assert os.path.exists(metadata_path)

    # Read the data from the file
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        data = list(reader)

    # Shuffle the data to ensure random splitting
    random.seed(88)
    random.shuffle(data)

    # Calculate the split index
    split_index = int(len(data) * (1 - validation_size))

    # Split the data into train and validation sets
    train_set = data[:split_index]
    validation_set = data[split_index:]

    train_set_path = os.path.join(save_dir, "train_metadata.csv")
    validation_set_path = os.path.join(save_dir, "validation_metadata.csv")

    with open(train_set_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(train_set)

    with open(validation_set_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(validation_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--metadata_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "metadata.csv"),
        help="Path to metadata.csv",
    )

    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Where to save the split dataset",
    )

    args = parser.parse_args()

    split_ljspeech(args.metadata_path, args.save_dir)
