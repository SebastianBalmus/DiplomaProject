import os
import csv
import logging
import argparse
import numpy as np
from inference_handlers.Tacotron2InferenceHandler import Tacotron2InferenceHandler


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def process_data(args):
    """
    Processes input data to generate and save mel-spectrograms using Tacotron2.

    This function reads metadata, synthesizes speech for each text entry using 
    teacher-forced inference with Tacotron2, and saves the resulting mel-spectrograms 
    to the specified directory.

    Args:
        args (argparse.Namespace): Command-line arguments including:
            - metadata_path (str): Path to the metadata CSV file containing text entries.
            - save_dir (str): Directory where the generated mel-spectrograms will be saved.
            - wavs_dir (str): Directory where the original waveform files are stored.

    Side Effects:
        - Creates the save directory if it doesn't exist.
        - Reads text entries and corresponding waveform paths from the metadata CSV.
        - Synthesizes mel-spectrograms using Tacotron2's teacher inference method.
        - Saves the generated mel-spectrograms as .npy files in the save directory.
        - Logs progress and file saving status.
    """
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        data = list(reader)

    # create save directory if it doesn't exist
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        os.chmod(args.save_dir, 0o775)

    tacotron2 = Tacotron2InferenceHandler(
        args.ckpt_path, use_cuda=True
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
        "-cp",
        "--ckpt_path",
        type=str,
        help="Path to Tacotron2 checkpoint used for inference"
    )

    parser.add_argument(
        "-w", "--wavs_dir", type=str, help="Where the original wavs are stored"
    )

    args = parser.parse_args()

    process_data(args)
