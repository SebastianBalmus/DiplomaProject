# DiplomaProject

python -W ignore::UserWarning train_tacotron2.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/metadata_test.csv -cd /train_path/tacotron2_ckpt -l /train_path/tacotron2_log

python -W ignore::UserWarning train_tacotron2.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/metadata.csv -cd /train_path/tacotron2_ckpt -l /train_path/tacotron2_log


python -W ignore::UserWarning train_hifigan.py -w /train_path/LJSpeech-1.1/wavs/ -t /train_path/LJSpeech-1.1/metadata_test.csv -v /train_path/LJSpeech-1.1/metadata_test.csv -cd /train_path/hifigan_ckpt -l /train_path/hifigan_log


python hifigan_fine_tuning_preprocessing.py -m /train_path/LJSpeech-1.1/metadata_test.csv -s /train_path/LJSpeech-1.1/mels
