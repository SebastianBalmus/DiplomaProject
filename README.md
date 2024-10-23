# DiplomaProject

This project is my Bachelor's diploma project and it consists of a Text-to-Speech (TTS) in English and Romanian application built using Tacotron2 and HiFi-GAN.

### Features
- Fast synthesis in English and Romanian
- High-fidelity speech generation
- Easily extensible with other languages

### Requirements
- Docker
- CUDA

### Installation
1. Clone this repository.
2. Use the ```scripts/launch_container.sh``` utility to create a development environment.
3. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) and the [MARA Corpus](https://speech.utcluj.ro/marasc/). For the MARA Corpus, additional preprocessing is needed in order to create a metadata file that is identical with the LJSpeech one. Furthermore, the audios must be resampled to 22.5 kHz.
4. Train Tacotron2 and HiFi-GAN using the ```train_tacotron2.py``` and ```train_hifigan.py``` utilities.
5. For a greater fidelity audio, you can use the hifigan_fine_tuning_preprocessing.py script in order to
6. (Optional) If you want to use a web interface for inference, you can use the ```scripts/launch_frontend.sh```to launch the frontend and ```scripts/launch_backend.sh``` (from within the dev container) to launch a REST API.
7. (Optional) You can use the ```scripts/launch_jupyter_server.sh``` to launch the Jupyter Notebook interface and use the demo notebooks.


> Info: For every Bash or Python script, you can use the ```-h``` flag to see the possible arguments


### Usage
The Inference can be done in 3 distinct ways: a REST API, a web interface and using the provided Jupyter Notebooks.

