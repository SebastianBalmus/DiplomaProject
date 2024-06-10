import os
import tempfile
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from scipy.io import wavfile
from .model_selections import select_model



app = FastAPI()


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
            os.remove(tmpfile.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

