import os
import csv
import logging
import argparse
import numpy as np
from inference_handlers.Tacotron2InferenceHandler import Tacotron2InferenceHandler


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def process_data(args):
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        data = list(reader)

    # create save directory if it doesn't exist
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        os.chmod(args.save_dir, 0o775)

    tacotron2 = Tacotron2InferenceHandler(
        '/train_path/working_models/tacotron2_initial_training_modified',
        use_cuda=True
    )

    for sentence in data:
        logger.info(f"Synthesising {sentence[0]}")

        wav_path = os.path.join(args.wavs_dir, f"{sentence[0]}.wav")
        mel = tacotron2.teacher_inference(wav_path, sentence[1])

        out_path = os.path.join(args.save_dir, f"{sentence[0]}.npy")
        np.save(out_path, mel)
        logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--metadata_path",
        type=str,
        help="Path to original metadata.csv",
    )

    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        help="Where to save the mels",
    )

    parser.add_argument(
        "-w",
        "--wavs_dir",
        type=str,
        help="Where the original wavs are stored"
    )

    args = parser.parse_args()

    process_data(args)