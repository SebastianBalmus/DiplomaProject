import sys
sys.path.append('..')

from fastapi import HTTPException
from inference_handlers.HiFiGanInferenceHandler import HiFiGanInferenceHandler
from inference_handlers.Tacotron2InferenceHandler import Tacotron2InferenceHandler


_CHECKPOINTS_PATHS = {
    "tacotron2_ro": "/train_path/working_models/tacotron2_ro",
    "tacotron2_en": "/train_path/working_models/tacotron2_en",
    "hifigan": "/train_path/working_models/hifigan",
    "hifigan_ft_ro": "/train_path/working_models/hifigan_ft_ro",
    "hifigan_ft_en": "/train_path/working_models/hifigan_ft_en",
}


def select_model(model_id, use_cuda):
    if model_id == "tts_en":
        tacotron2_ckpt = _CHECKPOINTS_PATHS['tacotron2_en']
        hifigan_ckpt = _CHECKPOINTS_PATHS['hifigan']

    elif model_id == "tts_ro":
        tacotron2_ckpt = _CHECKPOINTS_PATHS['tacotron2_ro']
        hifigan_ckpt = _CHECKPOINTS_PATHS['hifigan']

    elif model_id == "tts_en_ft":
        tacotron2_ckpt = _CHECKPOINTS_PATHS['tacotron2_en']
        hifigan_ckpt = _CHECKPOINTS_PATHS['hifigan_ft_en']

    elif model_id == "tts_ro_ft":
        tacotron2_ckpt = _CHECKPOINTS_PATHS['tacotron2_ro']
        hifigan_ckpt = _CHECKPOINTS_PATHS['hifigan_ft_ro']

    else:
        raise HTTPException(status_code=422, detail="No model with the provided ID was found!")

    tacotron2 = Tacotron2InferenceHandler(
        ckpt_pth=tacotron2_ckpt,
        use_cuda=use_cuda,
    )

    hifigan = HiFiGanInferenceHandler(
        ckpt_pth=hifigan_ckpt,
        use_cuda=use_cuda,
    )

    return tacotron2, hifigan