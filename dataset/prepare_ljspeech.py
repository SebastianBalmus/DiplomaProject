import os
import csv
import random
import argparse


def split_ljspeech(metadata_path, save_dir, validation_size=0.1, seed=88):
    """
    Splits the LJ Speech dataset metadata into train and validation sets.

    Args:
        metadata_path (str): Path to the metadata CSV file containing LJ Speech dataset information.
        save_dir (str): Directory where the split dataset will be saved.
        validation_size (float, optional): Fraction of the dataset to include in the validation set. Defaults to 0.1.
        seed (int, optional): Seed for random shuffling of the dataset. Defaults to 88.

    Raises:
        AssertionError: If metadata_path does not exist.

    Note:
        The metadata CSV file is expected to have entries separated by '|' delimiter.

    """
    assert os.path.exists(metadata_path)

    # Read the data from the file
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        data = list(reader)

    # Shuffle the data to ensure random splitting
    random.seed(seed)
    random.shuffle(seed)

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
