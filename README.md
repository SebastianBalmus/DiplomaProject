# DiplomaProject

python -W ignore::UserWarning train_tacotron2.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/metadata.csv -cd /train_path/tacotron2_ckpt -l /train_path/tacotron2_log

python -W ignore::UserWarning train_tacotron2.py -w /train_path/Mara/wavs/ -m /train_path/Mara/metadata.csv -cd /train_path/tacotron2_ckpt/ro_ckpt -l /train_path/tacotron2_log/ro_log

python -W ignore::UserWarning train_hifigan.py -w /train_path/LJSpeech-1.1/wavs/ -t /train_path/LJSpeech-1.1/metadata.csv -v /train_path/LJSpeech-1.1/metadata_test.csv -cd /train_path/hifigan_ckpt -l /train_path/hifigan_log

python -W ignore::UserWarning train_hifigan.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/mels/ -t /train_path/LJSpeech-1.1/train_metadata.csv -v /train_path/LJSpeech-1.1/validation_metadata.csv -cd /train_path/hifigan_ckpt -l /train_path/hifigan_log --ckpt_path_generator /train_path/working_models/hifigan_initial --ckpt_path_discriminator /train_path/working_models/do_00470000 --fine_tuning True


python hifigan_fine_tuning_preprocessing.py -m /train_path/LJSpeech-1.1/metadata.csv -s /train_path/LJSpeech-1.1/mels
