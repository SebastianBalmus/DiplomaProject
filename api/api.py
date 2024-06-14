import sys
sys.path.append('..')

import io
import os
import torch
import tempfile
import librosa
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.io import wavfile
from .model_selections import select_model
from hparams.HiFiGanHParams import HiFiGanHParams as hps
from dataset.HiFiGanDataset import HiFiGanDataset

plt.style.use('dark_background')
plt.rcParams.update({'font.size': 22})

app = FastAPI(middleware=[
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
])


class TextBody(BaseModel):
    text: str


@app.post("/infer")
def infer(model_id: str, use_cuda: bool, body: TextBody):

    tacotron2, hifigan = select_model(model_id, use_cuda)

    try:

        mel = tacotron2.infer_e2e(body.text)
        wav, sr = hifigan.infer(mel)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            wavfile.write(tmpfile.name, sr, wav.astype(np.int16))
            return FileResponse(tmpfile.name, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mel")
async def mel(wav_file: UploadFile = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    try:
        # Write the uploaded file contents to the temporary file
        temp_file.write(await wav_file.read())
        temp_file.close()

        # Load the audio data using librosa from the temporary file
        wav, sr = librosa.load(temp_file.name, sr=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading WAV file: {str(e)}")
    finally:
        os.remove(temp_file.name)  # Clean up the temporary file

    mel_spec = HiFiGanDataset.mel_spectrogram(
        torch.FloatTensor(wav).unsqueeze(0),
        n_fft=hps.n_fft,
        num_mels=hps.num_mels,
        sampling_rate=hps.sampling_rate,
        hop_size=hps.hop_size,
        win_size=hps.win_size,
        fmin=hps.fmin,
        fmax=hps.fmax_for_loss,
    )

    plt.figure(figsize=(11, 7))
    plt.imshow(mel_spec[0], aspect="auto", origin="lower", interpolation="none")
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")
