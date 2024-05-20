# DiplomaProject

python train_tacotron2.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/metadata.csv -cd /train_path/tacotron2_ckpt -l /train_path/tacotron2_log



python train_tacotron2.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/metadata_test.csv -cd /train_path/tacotron2_ckpt -l /train_path/tacotron2_log



python -m torch.distributed.launch --nproc_per_node 4 train_tacotron2.py -w /train_path/LJSpeech-1.1/wavs/ -m /train_path/LJSpeech-1.1/metadata.csv -cd /train_path/tacotron2_ckpt -l /train_path/tacotron2_log